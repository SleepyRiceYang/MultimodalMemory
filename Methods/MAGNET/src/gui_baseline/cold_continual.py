"""AndroidWorld Easy MAGNET dual-memory continual learning, initialized empty.

This is an online reconstruction, not the procedural-only 50/50 adaptation.
No Amex bootstrap, no hidden task answers, no capability modifications.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from itertools import islice
from pathlib import Path

import networkx as nx
import numpy as np
import requests
import yaml
from openai import OpenAI

from .common import image_part,read,write
from .cold_memory import DualMemory,empty_memory
from .cold_vision import MiniCPMEncoder,RegionParser
from .runner import Android

ROOT=Path(__file__).resolve().parents[4]
ABSTRACT='''Abstract common high-level workflows from these SUCCESSFUL GUI demonstrations.
Use only recorded actions, observations and task goals. Replace instance-specific names,
queries, phone numbers and dates with [Parameter] slots, preserving operation semantics.
Retain different valid paths. Return JSON {"workflows":[{"title":"task category",
"actions":["reusable high-level step"]}]}. Do not invent unsupported steps.'''
FUNCTION='''Describe the function of the clicked UI element using the BEFORE screen,
the click action, the cropped element and the AFTER screen. Judge whether the click
produced a meaningful, supported UI transition. Ignore whether the entire task succeeded.
Return JSON {"valid":true/false,"function":"specific reusable functional description",
"evidence":"observed transition"}. No change or insufficient evidence means valid=false.
Do not turn the current task goal into evidence of an action effect.'''
ACTION_NAMES={'CLICK_ELEMENT','SCROLL','SWIPE','INPUT','PRESS_ENTER','PRESS_HOME','PRESS_BACK','STOP','ANSWER'}


def parse_json(text,prefix=''):
    text=text.strip()
    if text.startswith('```'):text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
    try:return json.loads(text)
    except json.JSONDecodeError:
        if not prefix:raise
        return json.loads(prefix+text)


def load_prompts():
    base=read_yaml(ROOT/'Methods/MAGNET/configs/fragments/base_modern.yaml')['PROMPTS']
    actor=read_yaml(ROOT/'Methods/MAGNET/configs/fragments/modes/coat/coat_bbox.yaml')['PROMPTS']['ACTION_PREDICT']['COAT']
    return base,actor


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def check_stop(args):
    event=getattr(args,'stop_event',None)
    if event is not None and event.is_set():raise InterruptedError('Experiment stopped; partial task is not committed')


class Model:
    def __init__(self,args):
        self.args=args;self.client=OpenAI(base_url=args.model_url,api_key='EMPTY',timeout=180,max_retries=0)
    def call(self,folder,system,user,images=(),prefix=None,json_output=True,validator=None,response_schema=None):
        check_stop(self.args)
        folder=Path(folder)
        if (folder/'response.json').exists():
            text=read(folder/'response.json')['text']
            if validator:validator(parse_json(text,prefix or ''))
            return text
        content=[{'type':'text','text':user}]
        for label,path in images:content.extend([{'type':'text','text':label},image_part(path,max_side=2400)])
        messages=[{'role':'system','content':system},{'role':'user','content':content}]
        if prefix is not None:messages.extend([{'role':'assistant','content':prefix},
                                              {'role':'user','content':'Complete the above response: '}])
        spec={'model':self.args.model,'temperature':self.args.temperature,'max_tokens':self.args.max_tokens,
              'messages':[{'role':'system','content':system},{'role':'user','content':user}],
              'images':[(label,str(path)) for label,path in images],'assistant_prefix':prefix,
              'json_output':json_output,'max_attempts':self.args.max_model_attempts,'response_schema':response_schema}
        write(folder/'request.json',spec)
        kwargs={'response_format':{'type':'json_object'}} if json_output and prefix is None else {}
        if response_schema is not None:
            kwargs={'response_format':{'type':'json_schema','json_schema':{'name':'planner','strict':True,'schema':response_schema}}}
        for attempt in range(self.args.max_model_attempts):
            check_stop(self.args)
            print(f'MODEL {folder} attempt={attempt+1}',flush=True)
            result=self.client.chat.completions.create(model=self.args.model,messages=messages,
                temperature=self.args.temperature,max_tokens=self.args.max_tokens,**kwargs)
            write(folder/f'raw_response_{attempt}.json',result.model_dump())
            check_stop(self.args)
            try:
                if result.choices[0].finish_reason=='length':raise ValueError('Truncated model output')
                text=result.choices[0].message.content
                if json_output:
                    value=parse_json(text,prefix or '')
                    if validator:validator(value)
            except ValueError as exc:
                write(folder/f'invalid_{attempt}.json',{'error':str(exc)});continue
            write(folder/'response.json',{'text':text});return text
        raise ValueError(f'Invalid model output after {self.args.max_model_attempts} attempts: {folder}')


def substitute(template,**values):
    for key,value in values.items():template=template.replace('{'+key+'}',str(value))
    return template


def action_to_android(prediction,width,height):
    name=prediction['ACTION'].lower();args=prediction['ARGS']
    if name=='click_element':
        bbox=args['bbox']
        if len(bbox)!=4 or any(not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1000 for v in bbox):
            raise ValueError('Actor bbox must be four normalized 0..1000 numbers')
        if bbox[2]<bbox[0] or bbox[3]<bbox[1]:raise ValueError('Reversed actor bbox')
        return {'action_type':'click','x':min(width-1,round((bbox[0]+bbox[2])/2000*width)),
                'y':min(height-1,round((bbox[1]+bbox[3])/2000*height))}
    if name in ('scroll','swipe'):
        if args['direction'] not in ('up','down','left','right'):raise ValueError('Invalid scroll direction')
        return {'action_type':name,'direction':args['direction']}
    if name in ('input','answer'):
        if not isinstance(args['text'],str):raise ValueError('Input/answer text must be a string')
        return {'action_type':'input_text' if name=='input' else 'answer','text':args['text']}
    if name=='stop':
        if args['task_status'] not in ('success','failure'):raise ValueError('Invalid STOP status')
        return {'action_type':'status','goal_status':'complete' if args['task_status']=='success' else 'infeasible'}
    return {'action_type':{'press_enter':'keyboard_enter','press_home':'navigate_home','press_back':'navigate_back'}[name]}


def actor_schema(action_type):
    api={'CLICK_ELEMENT':'click_element','SCROLL':'scroll','SWIPE':'swipe','INPUT':'input','PRESS_ENTER':'press_enter',
         'PRESS_HOME':'press_home','PRESS_BACK':'press_back','STOP':'stop','ANSWER':'answer'}[action_type]
    properties={}
    if action_type=='CLICK_ELEMENT':
        properties={'bbox':{'type':'array','items':{'type':'number','minimum':0,'maximum':1000},'minItems':4,'maxItems':4}}
    elif action_type in ('SCROLL','SWIPE'):properties={'direction':{'type':'string','enum':['up','down','left','right']}}
    elif action_type in ('INPUT','ANSWER'):properties={'text':{'type':'string'}}
    elif action_type=='STOP':properties={'task_status':{'type':'string','enum':['success','failure']}}
    return {'type':'object','properties':{'ACTION':{'type':'string','enum':[api]},
            'ARGS':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}},
            'required':['ACTION','ARGS'],'additionalProperties':False}


def validate_plan(plan):
    actions=plan.get('Future Action Plan')
    if not isinstance(plan.get('Thought'),str) or not isinstance(actions,list) or not actions:
        raise ValueError('Planner must return Thought and a nonempty Future Action Plan')
    if any(not isinstance(a,dict) or a.get('type') not in ACTION_NAMES or not isinstance(a.get('description'),str) for a in actions):
        raise ValueError('Planner action type/description is invalid')


class Episode:
    def __init__(self,args,env,model,bank):
        self.args=args;self.env=env;self.model=model;self.bank=bank;self.base,self.actor=load_prompts()

    def retrieve_procedural(self,folder,goal,description,history,previous,before):
        return self.bank.select('procedural',goal)

    def run(self,folder,task,goal,limit):
        history=[];previous=None;transitions=[]
        for step in range(limit):
            check_stop(self.args)
            d=folder/f'step_{step:03d}';d.mkdir();before=d/'before.png';after=d/'after.png'
            width,height=self.env.screenshot(before)
            description=self.model.call(d/'observe',self.base['SCREEN_DESC']['SYSTEM'],
                self.base['SCREEN_DESC']['USER'].replace('{screenshot}','[CURRENT SCREEN]'),[('CURRENT SCREEN',before)],json_output=False)
            workflows,trace=self.retrieve_procedural(d,goal,description,history,previous,before)
            write(d/'procedural_retrieval.json',trace)
            key='THINK_DESC_WITH_MEMORY' if workflows else 'THINK_DESC'
            system=self.base[key]['SYSTEM'].replace('SCROLL, INPUT','SCROLL, SWIPE, INPUT')
            system+='\nAndroidWorld interface: SCROLL describes content navigation (down moves the finger upward). SWIPE describes finger motion from a screen edge (down moves from top to bottom).'
            # AndroidWorld IR tasks require an ANSWER action; the offline template lacks it.
            if task in self.args.ir_tasks:system=system.replace('PRESS_BACK, STOP]','PRESS_BACK, STOP, ANSWER]')
            user=substitute(self.base[key]['USER'],screenshot='[CURRENT SCREEN]',screen_desc=description,
                history_actions=(', '.join(history) if history else 'None')+'.',user_request=goal,
                relevant_workflows=json.dumps([{'title':n['title'],'actions':n['actions']} for n in workflows]))
            allowed=sorted(ACTION_NAMES if task in self.args.ir_tasks else ACTION_NAMES-{'ANSWER'})
            schema={'type':'object','properties':{'Thought':{'type':'string'},'Future Action Plan':{'type':'array','minItems':1,
                'items':{'type':'object','properties':{'type':{'type':'string','enum':allowed},'description':{'type':'string'}},
                         'required':['type','description'],'additionalProperties':False}}},
                'required':['Thought','Future Action Plan'],'additionalProperties':False}
            if workflows:
                schema['properties']['Reference to Workflows']={'type':'string'}
                schema['required'].append('Reference to Workflows')
            plan=parse_json(self.model.call(d/'planner',system,user,[('CURRENT SCREEN',before)],validator=validate_plan,response_schema=schema))
            next_action=plan['Future Action Plan'][0]
            if next_action['type'] not in ACTION_NAMES:raise ValueError('Unknown planner action type')
            stationary=[];query=next_action['description']
            if next_action['type']=='CLICK_ELEMENT':
                stationary,trace=self.bank.select('stationary',query);write(d/'stationary_retrieval.json',trace)
            action_space=self.base['ACTION_PREDICT']['ACTION_SPACE']['SCREEN_TAG']
            action_space+='\n\nSWIPE:\n  summary: native AndroidWorld edge-to-edge finger gesture\n  usage: swipe(direction=up|down|left|right)\n\nANSWER:\n  summary: answer an information retrieval task\n  usage: answer(text="the answer")'
            selected_space=next(x for x in action_space.split('\n\n') if x.split(':\n')[0]==next_action['type'])
            system=self.actor['SYSTEM']['SCREEN_TAG_WITH_MEMORY' if stationary else 'SCREEN_TAG']
            system=system.replace('{action_space}',selected_space)
            system+='\nAll bbox coordinates use integers in 0..1000 relative to the CURRENT screenshot.'
            template=self.actor['USER_WITH_MEMORY' if stationary else 'USER']
            template='\n'.join(l for l in template.splitlines() if (history or '{history_actions}' not in l)
                               and (previous or '{prev_action_result}' not in l))
            user=substitute(template,screenshot='[CURRENT SCREEN]',screen_desc=description,history_actions=', '.join(history)+'.',
                prev_action_result=previous,user_request=goal,revelant_ui='[HISTORICAL UI REFERENCE PATCHES]')
            prefix=substitute(self.actor['ASST'],action_thought=json.dumps(plan['Thought']),next_single_action=json.dumps(next_action))
            images=[('CURRENT SCREEN',before)]+[(f'HISTORICAL function: {n["function"]}',n['image']) for n in stationary]
            output=parse_json(self.model.call(d/'actor',system,user,images,prefix,response_schema=actor_schema(next_action['type'])),prefix)
            action=action_to_android(output,width,height);write(d/'action.json',action)
            write(d/'context.json',{'screen_description':description,'plan':plan,'history':history,'previous_result':previous})
            print(f'ACTION {task} {step+1}/{limit} {action}',flush=True)
            if action['action_type']=='status':break
            check_stop(self.args)
            self.env.api('POST','/execute_action',json=action);time.sleep(self.args.settle_seconds)
            self.env.screenshot(after)
            transition={'before':str(before.resolve()),'after':str(after.resolve()),'action':action,
                        'description':next_action['description'],'step':step,'screen_description':description}
            transitions.append(transition)
            previous=self.model.call(d/'reflect',self.base['ACTION_RESULT']['SYSTEM'],substitute(self.base['ACTION_RESULT']['USER'],
                before_screenshot='[BEFORE]',after_screenshot='[AFTER]',last_action=next_action['description'],user_request=goal),
                [('BEFORE',before),('AFTER',after)],json_output=False)
            transition['effect']=previous;history.append(next_action['description'])
            if action['action_type']=='answer':break
        self.env.screenshot(folder/'final.png')
        # Official state/answer score is primary; no added "Agent must emit STOP" gate.
        # AndroidWorld documents RetroPlayingQueue's missing playback table as a known
        # evaluator error. Record that task as unscored/unsuccessful so the run can continue.
        score_error=None
        try:score=self.env.api('GET','/task/score',params=self.env.params)['score']
        except requests.HTTPError as exc:
            if task!='RetroPlayingQueue' or exc.response.status_code!=500:raise
            score=None;score_error=str(exc)
        episode={'task_id':task,'goal':goal,'score':score,'success':score==1,'steps':step+1,
                 'budget':limit,'transitions':transitions,
                 'status':'score_error' if score_error else 'completed'}
        if score_error:episode['score_error']=score_error
        return episode


def ordered_model_calls(function,jobs,workers,args):
    """Bound in-flight requests and commit in source order; cancel queued work on failure."""
    jobs=iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=deque()
        succeeded=False
        try:
            for job in islice(jobs,workers):pending.append(pool.submit(function,job))
            while pending:
                check_stop(args)
                yield pending.popleft().result()
                for job in islice(jobs,1):pending.append(pool.submit(function,job))
            succeeded=True
        finally:
            if not succeeded:
                event=getattr(args,'stop_event',None)
                if event is not None:event.set()
                for future in pending:future.cancel()


def update_memory(args,bank,episodes,folder,model,parser,round_number):
    workers=getattr(args,'build_workers',1)
    successes=[e for e in episodes if e['success'] and e['transitions']]
    if successes:
        vectors=np.asarray(bank.encoder([e['goal'] for e in successes]));graph=nx.Graph();graph.add_nodes_from(range(len(successes)))
        for i in range(len(successes)):
            for j in range(i):
                if float(vectors[i]@vectors[j])>args.cluster_threshold:graph.add_edge(i,j)
        clusters=sorted(sorted(c) for c in nx.find_cliques(graph));write(folder/'clusters.json',clusters)
        def abstract(job):
            index,cluster=job
            trajectories=[{'goal':successes[i]['goal'],'actions':[{'action':s['action'],'description':s['description'],
                'observation':s['screen_description'],'effect':s['effect']} for s in successes[i]['transitions']]} for i in cluster]
            return parse_json(model.call(folder/f'workflow_{index:03d}',ABSTRACT,json.dumps(trajectories)))
        for index,value in enumerate(ordered_model_calls(abstract,enumerate(clusters),workers,args)):
            cluster=clusters[index]
            check_stop(args)
            source=[successes[i]['task_id'] for i in cluster]
            for workflow in value['workflows']:
                if not isinstance(workflow['title'],str) or not workflow['title'] or not workflow['actions']:
                    raise ValueError('Invalid abstract workflow')
                bank.add_workflow(workflow['title'],workflow['actions'],round_number,source)
    # Crop serially on the shared GPU; overlap queued VLM calls with later crops.
    def click_jobs():
        for episode in episodes:
            for transition in episode['transitions']:
                check_stop(args)
                action=transition['action']
                if action['action_type']!='click':continue
                d=folder/'stationary'/episode['task_id']/f'step_{transition["step"]:03d}';d.mkdir(parents=True,exist_ok=True)
                patch=d/'element.png';box,boxes=parser.crop_click(transition['before'],action['x'],action['y'],patch)
                write(d/'regions.json',{'proposals':boxes,'selected':box,'click':action})
                if box is None:
                    write(d/'decision.json',{'accepted':False,'reason':'No detected region contains the executed click'});continue
                yield episode['task_id'],transition,d,patch
    def annotate(job):
        task,transition,d,patch=job
        value=parse_json(model.call(d/'function',FUNCTION,json.dumps({'action':transition['action']}),
            [('BEFORE',transition['before']),('CLICKED ELEMENT',patch),('AFTER',transition['after'])]))
        if type(value['valid']) is not bool or not isinstance(value['function'],str):raise ValueError('Invalid function annotation')
        return task,d,patch,value
    for task,d,patch,value in ordered_model_calls(annotate,click_jobs(),workers,args):
        check_stop(args)
        if value['valid'] and value['function'].strip():
            identity,added=bank.add_visual(value['function'],patch,round_number,task)
            write(d/'decision.json',{'accepted':True,'function_id':identity,'new_visual':added,**value})
        else:write(d/'decision.json',{'accepted':False,**value})
    bank.data['completed_round']=round_number
    write(folder/'memory.json',bank.data)


def task_selection(args):
    metadata=read(ROOT/'Benchmark/AndroidWorld/android_world/task_metadata.json')
    tasks=[x['task_name'] for x in metadata if x['difficulty']=='easy']
    assert len(tasks)==61
    if args.tasks:
        selected=args.tasks.split(',')
        if len(selected)!=len(set(selected)) or not set(selected)<=set(tasks):raise ValueError('--tasks must be distinct Easy IDs')
        tasks=selected
    stats=read(ROOT/'Docs/实验方案/assets/20261026-gui-protocol/androidworld_registry_stats.json')
    complexities={x['task']:x['complexity'] for x in stats['items']}
    return tasks,{t:int(10*complexities[t]) for t in tasks}


def preflight(args):
    checks=[]
    for label,url in [('vllm',args.model_url+'/models'),('androidworld',args.android_url+'/health')]:
        try:
            r=requests.get(url,timeout=10);r.raise_for_status();body=r.json()
            ok=label!='vllm' or args.model in [x['id'] for x in body['data']]
            checks.append({'check':label,'ok':ok,'url':url})
        except requests.RequestException as exc:checks.append({'check':label,'ok':False,'url':url,'error':str(exc)})
    for name,path in [('embedding',args.embedding_model),('detector',args.detector_weights)]:
        checks.append({'check':name,'ok':Path(path).exists(),'path':str(path)})
    running=subprocess.run(['docker','inspect','-f','{{.State.Running}}',args.android_container],capture_output=True,text=True)
    checks.append({'check':'android_container','ok':running.returncode==0 and running.stdout.strip()=='true'})
    print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)
    return all(c['ok'] for c in checks)


def argument_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['plan','preflight','model-smoke','run','report'],default='run')
    p.add_argument('--run',default='easy61_empty_qwen3vl8b_v1')
    p.add_argument('--model-url',default='http://127.0.0.1:8170/v1');p.add_argument('--model',default='qwen3-vl-8b-instruct')
    p.add_argument('--android-url',default='http://127.0.0.1:15006');p.add_argument('--android-container',default='magnet_cold_aw')
    p.add_argument('--embedding-model',default='/data/pretrained_models/MiniCPM-Embedding')
    p.add_argument('--detector-weights',type=Path,default=ROOT/'Benchmark/OmniParser/weights/icon_detect/model.pt')
    p.add_argument('--device',default='cuda:0');p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--tasks',help='Optional comma-separated Easy IDs, for a separate smoke run')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--instance-policy',choices=['fixed','new-per-round'],default='fixed')
    p.add_argument('--temperature',type=float,default=1.);p.add_argument('--max-tokens',type=int,default=2000)
    p.add_argument('--settle-seconds',type=float,default=2.)
    p.add_argument('--max-model-attempts',type=int,default=5)
    p.add_argument('--cluster-threshold',type=float,default=.75)
    p.add_argument('--procedural-threshold',type=float,default=.4);p.add_argument('--stationary-threshold',type=float,default=.7)
    p.add_argument('--procedural-top-n',type=int,default=5);p.add_argument('--procedural-top-k',type=int,default=5)
    p.add_argument('--stationary-top-n',type=int,default=20);p.add_argument('--stationary-top-k',type=int,default=20)
    return p


def main():
    p=argument_parser()
    args=p.parse_args()
    if args.rounds<0 or not re.fullmatch('[A-Za-z0-9_.-]+',args.run):p.error('Invalid rounds or run name')
    for kind in ('procedural','stationary'):
        if not 1<=getattr(args,kind+'_top_k')<=getattr(args,kind+'_top_n'):p.error('Require 1 <= top_k <= top_n')
    args.task_index=0;tasks,budgets=task_selection(args)
    proto=(ROOT/'Benchmark/AndroidWorld/android_world/task_evals/information_retrieval/proto/tasks.textproto').read_text()
    args.ir_tasks=re.findall(r'^  name: "([^"]+)"',proto,re.M)
    root=ROOT/'Experiments/memory_baselines/artifacts/magnet_cold_continual'/args.run
    if args.stage=='plan':
        print(json.dumps({'root':str(root),'initialization':'empty dual memory','tasks':tasks,'budgets':budgets,
                          'passes':args.rounds+1,'episodes':len(tasks)*(args.rounds+1),'model_url':args.model_url},indent=2));return
    if args.stage=='report':report(root);return
    if args.stage=='preflight':raise SystemExit(0 if preflight(args) else 1)
    args.max_steps=max(budgets.values());args.max_tokens=int(args.max_tokens)
    root.mkdir(parents=True,exist_ok=True)
    options={k:v for k,v in vars(args).items() if k not in ('stage','run')};options['detector_weights']=str(options['detector_weights'])
    options.update(tasks=tasks,budgets=budgets,initialization='empty',parser='OmniParser-v2 detector + EasyOCR; VLM function semantics')
    sources=['Methods/MAGNET/src/gui_baseline/cold_continual.py','Methods/MAGNET/src/gui_baseline/cold_memory.py',
             'Methods/MAGNET/src/gui_baseline/cold_vision.py','Methods/MAGNET/configs/fragments/base_modern.yaml',
             'Methods/MAGNET/configs/fragments/modes/coat/coat_bbox.yaml','Benchmark/AndroidWorld/android_world/task_metadata.json',
             'Docs/实验方案/assets/20261026-gui-protocol/androidworld_registry_stats.json']
    options['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    options['detector_sha256']=hashlib.sha256(args.detector_weights.read_bytes()).hexdigest()
    if (root/'config.json').exists() and read(root/'config.json')!=options:raise ValueError('Run config changed; use a new --run')
    write(root/'config.json',options)
    model=Model(args)
    if args.stage=='model-smoke':
        sample=ROOT/'Methods/MAGNET/sample_data/android_in_the_zoo/test/general/GENERAL-11634661049784322652/GENERAL-11634661049784322652_1.png'
        prompts,_=load_prompts()
        value=model.call(root/'model_smoke',prompts['SCREEN_DESC']['SYSTEM'],
            prompts['SCREEN_DESC']['USER'].replace('{screenshot}','[CURRENT SCREEN]'),[('CURRENT SCREEN',sample)],json_output=False)
        print(value);return
    if not preflight(args):raise SystemExit('Preflight failed; no benchmark episode executed')
    locks=ROOT/'Experiments/memory_baselines/artifacts/locks';locks.mkdir(parents=True,exist_ok=True)
    with (locks/'androidworld_http.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        encoder=MiniCPMEncoder(args.embedding_model,args.device);parser=RegionParser(args.detector_weights,args.device)
        env=Android(args)
        try:
            bank=DualMemory(empty_memory(),encoder,vars(args));write(root/'initial_memory.json',bank.data)
            for number in range(args.rounds+1):
                folder=root/f'round_{number:02d}';folder.mkdir(exist_ok=True)
                if (folder/'summary.json').exists():
                    bank=DualMemory(read(folder/'memory.json'),encoder,vars(args));continue
                episodes=[]
                seed=args.seed+(number if args.instance_policy=='new-per-round' else 0)
                for task in tasks:
                    target=folder/'tasks'/task
                    if (target/'checkpoint.json').exists():
                        checkpoint=read(target/'checkpoint.json');episodes.append(checkpoint['episode'])
                        bank=DualMemory(checkpoint['memory'],encoder,vars(args));continue
                    if target.exists():
                        target.rename(target.with_name(task+'_interrupted_'+str(time.time_ns())))
                    target.mkdir(parents=True)
                    env.api('GET','/suite/reinitialize',params={'seed':seed,'n_task_combinations':1})
                    goal=env.setup(task)
                    write(target/'task.json',{'task_id':task,'goal':goal,'seed':seed,'budget':budgets[task]})
                    try:episode=Episode(args,env,model,bank).run(target,task,goal,budgets[task])
                    finally:env.cleanup()
                    episodes.append(episode)
                    # One atomic commit covers trajectory score and retrieval counters.
                    write(target/'checkpoint.json',{'episode':episode,'memory':bank.data})
                    write(folder/'progress.json',{'completed':len(episodes),'total':len(tasks),'successes':sum(e['success'] for e in episodes)})
                    print(f'PROGRESS round={number}/{args.rounds} tasks={len(episodes)}/{len(tasks)} success={sum(e["success"] for e in episodes)}',flush=True)
                if number<args.rounds:
                    before_update=copy.deepcopy(bank.data)
                    # Round updates resume from the immutable pre-update bank, not half-written additions.
                    write(folder/'memory_before_update.json',before_update)
                    update_memory(args,bank,episodes,folder/'build',model,parser,number)
                bank.data['completed_round']=number;write(folder/'memory.json',bank.data)
                write(folder/'summary.json',{'round':number,'episodes':episodes,'success_rate':sum(e['success'] for e in episodes)/len(tasks),
                      'procedural_units':len(bank.data['procedural']),'stationary_functions':len(bank.data['stationary']),
                      'stationary_images':sum(len(n['images']) for n in bank.data['stationary'])})
                report(root)
        finally:env.close()


def report(root):
    rows=[]
    for path in sorted(root.glob('round_*/summary.json')):
        s=read(path);rows.append({k:v for k,v in s.items() if k!='episodes'})
    write(root/'summary.json',rows)
    text=['# MAGNET AndroidWorld cold-start continual learning','',
          'Initialization: empty procedural + stationary memory. No Amex bootstrap.',
          '','| Round | Success rate | Procedural | Functions | Images |','|---|---:|---:|---:|---:|']
    for r in rows:text.append(f'| {r["round"]} | {r["success_rate"]:.4f} | {r["procedural_units"]} | {r["stationary_functions"]} | {r["stationary_images"]} |')
    (root/'report.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
