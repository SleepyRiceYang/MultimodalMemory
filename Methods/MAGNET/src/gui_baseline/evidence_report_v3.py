"""Normalize recorded MAGNET requests into separate condition/task HTML trajectories."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image

from .cold_continual import ROOT, load_prompts
from .common import read, write

ROLES=['observe','subtask','subtask_applicability','planner_evidence_selection',
       'planner','actor_evidence_selection','actor','reflect']


def image_spec(path,label=None):
    p=Path(path).resolve()
    with Image.open(p) as im:
        spec={'path':str(p),'width':im.width,'height':im.height}
    if label is not None:spec['label']=label
    return spec


def normalize(run):
    run=Path(run).resolve();templates={};trajectories=[]
    base,actor=load_prompts()
    for condition in ['B2','B3']:
        if not (run/condition/'config.json').exists():
            continue
        config=read(run/condition/'config.json')
        for task in config['tasks']:
            folder=run/condition/'tasks'/task
            task_file=folder/'task.json'
            task_data=read(task_file) if task_file.exists() else {'goal':'尚未启动：未记录任务指令'}
            cp=folder/'checkpoint.json'
            episode=read(cp)['episode'] if cp.exists() else None
            costs={'calls':0,'prompt_tokens':0,'completion_tokens':0,'request_images':0,
                   'selected_views':0,'loaded_views':0,'original_stationary_images':0}
            steps=[]
            for step in sorted(folder.glob('step_[0-9][0-9][0-9]')):
                context=read(step/'context.json') if (step/'context.json').exists() else {}
                action=read(step/'action.json') if (step/'action.json').exists() else {'action_type':'unrecorded'}
                normalized={'type':action['action_type'],'coordinate_space':'pixels',
                            'status':'not_executed_stop' if action['action_type']=='status' else
                                     ('issued' if (step/'after.png').exists() else 'execution_unconfirmed'),
                            **{k:v for k,v in action.items() if k!='action_type'}}
                plan=context.get('plan',{})
                summary=plan.get('Future Action Plan',[{'description':'No recorded plan'}])[0]['description']
                row={'id':step.name,'summary':summary,'action':normalized,'modules':[],'evidence':[],'references':[]}
                for side in ['before','after']:
                    if (step/(side+'.png')).exists():row[side]=image_spec(step/(side+'.png'))
                seen_images=set()
                for role in ['planner','actor']:
                    p=step/(role+'_evidence.json')
                    if p.exists():
                        evidence=read(p);row['evidence'].append({'role':role,**evidence})
                        costs['selected_views']+=len(evidence['selected_images'])
                        costs['loaded_views']+=len(evidence['loaded_images'])
                        for path in evidence['loaded_images']:
                            if path not in seen_images:
                                row['references'].append(image_spec(path,role+' selected reference'));seen_images.add(path)
                for role in ROLES:
                    request_path=step/role/'request.json'
                    response_path=step/role/'response.json'
                    if not request_path.exists():continue
                    request=read(request_path)
                    if role=='actor' and config.get('stationary_policy')=='original_MAGNET_unchanged':
                        costs['original_stationary_images']+=len(request['images'])-1
                        for label,path in request['images'][1:]:
                            if path not in seen_images:
                                row['references'].append(image_spec(path,label));seen_images.add(path)
                    costs['request_images']+=len(request['images'])
                    shared={'system':request['messages'][0]['content']}
                    inputs={'images':request['images']}
                    if role=='observe':
                        shared['user_template']=base['SCREEN_DESC']['USER']
                    elif role in ['subtask','subtask_applicability','planner_evidence_selection','actor_evidence_selection']:
                        inputs.update(json.loads(request['messages'][1]['content']))
                        if 'current_request' in inputs:
                            # Exact nested role user remains available through raw_request.
                            inputs['current_request']='See corresponding role inputs and raw request; full nested text omitted here.'
                    elif role in ['planner','actor']:
                        shared['user_templates']=({k:base[k]['USER'] for k in ['THINK_DESC','THINK_DESC_WITH_MEMORY']}
                                                   if role=='planner' else {k:actor[k] for k in ['USER','USER_WITH_MEMORY']})
                        shared['evidence_use_policy']=config['use_prompt']
                        inputs.update({'task':task_data['goal'],'screen_description':context.get('screen_description','Not recorded'),
                                       'history':context.get('history','Not recorded')})
                        if role=='actor':
                            inputs['previous_result']=context.get('previous_result')
                            inputs['assistant_prefix']=request['assistant_prefix']
                        text=request['messages'][1]['content']
                        if not context:
                            inputs['role_user_text']=text
                        evidence_path=step/'planner_evidence.json'
                        if role=='planner' and evidence_path.exists():
                            inputs.update(read(evidence_path))
                        if role=='planner':
                            start=text.find('<RELEVANT WORKFLOWS>:')
                            if start>=0:inputs['retrieved_workflows_text']=text[start:].split('<YOUR RESPONSE>')[0]
                    elif role=='reflect':
                        shared['user_template']=base['ACTION_RESULT']['USER']
                        inputs.update({'task':task_data['goal'],'last_action':summary})
                    tid='prompt-'+hashlib.sha256(json.dumps(shared,sort_keys=True).encode()).hexdigest()[:12]
                    templates[tid]=shared
                    output=read(response_path)['text'] if response_path.exists() else 'Response not recorded'
                    row['modules'].append({'role':role,'input':inputs,'output':output,'template_id':tid,
                                           'raw_request':str(request_path),'raw_response':str(response_path) if response_path.exists() else None})
                    for raw in (step/role).glob('raw_response_*.json'):
                        usage=read(raw)['usage'];costs['calls']+=1
                        for k in ['prompt_tokens','completion_tokens']:costs[k]+=usage[k]
                steps.append(row)
            trajectories.append({'condition':condition,'task':task,'goal':task_data['goal'],
                                 'score':episode['score'] if episode else None,
                                 'status':'completed' if episode else ('incomplete' if task_file.exists() else 'not_started'),
                                 'note':'仅对 procedural 历史图门控；Actor 保留原 stationary 参考图。' if config.get('stationary_policy')=='original_MAGNET_unchanged' else '旧协议：Planner 和 Actor 均参与历史图门控。',
                                 'costs':costs,'steps':steps})
    return {'title':run.name+' · independent condition trajectories','templates':templates,'trajectories':trajectories}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();data=normalize(args.run)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    write(out/'trajectories.json',data)
    renderer=ROOT/'.agents/skills/gui-agent-trajectory/scripts/render.py'
    spec=importlib.util.spec_from_file_location('trajectory_renderer',renderer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pages=module.render(data,out)
    write(out/'costs.json',{c:{k:sum(t['costs'][k] for t in data['trajectories'] if t['condition']==c)
                               for k in data['trajectories'][0]['costs']}
                          for c in sorted({t['condition'] for t in data['trajectories']})})
    print(json.dumps({'pages':len(pages),'output':str(out)},ensure_ascii=False))


if __name__=='__main__':main()
