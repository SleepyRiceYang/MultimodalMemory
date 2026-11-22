"""Resumable synchronous-round MAGNET procedural evolution on construction tasks only."""
from __future__ import annotations
import argparse,copy,fcntl,json,re,time
import requests
from openai import APIConnectionError,APITimeoutError,InternalServerError
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
from .common import read,write,JsonModel
from .runner import Android,Desktop,redact
from .evolution_store import EvolutionBank,empty_bank
from .evolution_build import episode_from_result,update_from_episodes
from .prompts_evolution import PLANNER,ACTOR_NOTICE,ABSTRACT,VERIFY
from .prompts import ANDROID_ACTOR

WORKSPACE=Path(__file__).resolve().parents[4]
VARIANTS=('text','visual')

PLANNER_SCHEMA={'type':'object','properties':{
    'subtasks':{'type':'array','items':{'type':'string','minLength':1},'minItems':1,'maxItems':8},
    'next_subtask':{'type':'string','minLength':1},'reason':{'type':'string'}},
    'required':['subtasks','next_subtask','reason'],'additionalProperties':False}

class PlannerOutputError(Exception):
    """Planner output remained invalid after its bounded repair attempt."""

def validate_plan(plan):
    if not isinstance(plan.get('next_subtask'),str) or not plan['next_subtask'].strip():raise ValueError('Planner omitted next_subtask')
    if not isinstance(plan.get('subtasks'),list) or not 1<=len(plan['subtasks'])<=8:raise ValueError('Invalid planner subtasks')
    if any(not isinstance(s,str) or not s.strip() for s in plan['subtasks']):raise ValueError('Invalid planner subtask text')



def freeze_date(value, date_label):
    if isinstance(value,str):
        return re.sub(r'The current date is [A-Za-z]+, [A-Za-z]+ \d{1,2}, \d{4}\.',
                      'The current date is '+date_label+'.',value)
    if isinstance(value,list):return [freeze_date(v,date_label) for v in value]
    if isinstance(value,dict):return {k:freeze_date(v,date_label) for k,v in value.items()}
    return value


class PlanningPolicy:
    def __init__(self, bank, model_url, visual, update, max_tokens):
        self.bank=bank;self.model=JsonModel(model_url);self.visual=visual;self.update=update;self.max_tokens=max_tokens

    def plan(self, folder, goal, observation, history):
        context=self.bank.select(goal,visual=self.visual,update=self.update)
        write(Path(folder)/'retrieval.json',context)
        payload={'current_goal':goal,'current_action_history':history[-10:],'relevant_workflows':context['workflows']}
        images=[str(observation)]+context['images']
        labels=['CURRENT OBSERVATION of THIS execution; authoritative state']+context['image_labels']
        try:
            plan=self.model.generate(folder,PLANNER,payload,images,self.max_tokens,image_labels=labels,repair_schema=PLANNER_SCHEMA,validator=validate_plan)
            validate_plan(plan)
        except ValueError as exc:
            if str(exc).startswith(('Invalid generation after two attempts:', 'Planner omitted next_subtask', 'Invalid planner subtasks', 'Invalid planner subtask text')):
                raise PlannerOutputError(str(exc)) from exc
            raise
        return plan


class PlannedAndroidModel:
    def __init__(self, policy, actor):self.policy=policy;self.actor=actor
    def generate(self, folder, system, payload, images=(), max_tokens=2048):
        plan=self.policy.plan(Path(folder)/'planner',payload['task'],images[0],payload['previous_actions'])
        return self.actor.generate(folder,system,{**payload,'magnet_planner_guidance':plan,'planner_notice':ACTOR_NOTICE},images,max_tokens)


class EvolutionAndroid(Android):
    def __init__(self,args,policy):
        super().__init__(args);self.model=PlannedAndroidModel(policy,self.model)
    def setup(self,task):
        # Reset the suite RNG before each task so resume and round order do not
        # change task parameters or initialization noise relative to the other arm.
        self.api('GET','/suite/reinitialize',params={'seed':self.args.seed,'n_task_combinations':1})
        return super().setup(task)


class EvolutionDesktop(Desktop):
    def __init__(self,args,policy):super().__init__(args);self.policy=policy
    def run(self,folder,goal,context):
        from mm_agents.qwen import QwenAgent
        agent=QwenAgent(model='qwen3-vl-8b-instruct',max_tokens=self.args.max_tokens,temperature=0,top_p=.9,
            action_space='pyautogui',observation_type='screenshot',coordinate_type='relative',history_n=3,image_max=3,
            fold_size=1,enable_thinking=False,base_url=self.args.model_url,api_key='EMPTY')
        agent.reset();native=agent.call_llm;counter=0;history=[];current_image=None
        def call(payload,model):
            nonlocal counter
            plan=self.policy.plan(folder/f'planner_{counter:03d}',goal,current_image,history)
            payload=copy.deepcopy(payload)
            payload['messages'].insert(1,{'role':'user','content':ACTOR_NOTICE+'\n'+json.dumps(plan,ensure_ascii=False)})
            payload['messages']=freeze_date(payload['messages'],self.args.date_label)
            write(folder/f'request_{counter:03d}.json',{'payload':redact(payload),'planner':plan,'experience_images_in_actor':False})
            response=native(payload,model);write(folder/f'response_{counter:03d}.json',{'response':response});counter+=1;return response
        agent.call_llm=call;done=False;steps=0
        for step in range(self.args.max_steps):
            current_image=folder/f'screen_{step:03d}.png';current_image.write_bytes(self.obs['screenshot'])
            _,actions=agent.predict(goal,self.obs)
            if not actions:raise ValueError('Native actor returned no executable action')
            for action in actions:
                if steps>=self.args.max_steps:break
                self.obs,reward,done,info=self.env.step(action,2)
                (folder/f'action_{steps:03d}_after.png').write_bytes(self.obs['screenshot'])
                write(folder/f'action_{steps:03d}.json',{'action':action,'done':done,'reward':reward,'info':info})
                history.append({'action':action});steps+=1
                print(f'ACTION {folder.name} {steps}: {action}',flush=True)
                if done:break
            if done or steps>=self.args.max_steps:break
        (folder/'final.png').write_bytes(self.obs['screenshot'])
        return {'score':float(self.env.evaluate()),'agent_done':bool(done),'steps':steps,'model_calls':counter,'planner_calls':counter}


def flatten(ids):return list(ids) if isinstance(ids,list) else [t for app in sorted(ids) for t in ids[app]]


def run_phase(root,args,task_ids,bank_data,phase,variant,round_number,update):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    initial=root/'input_bank.json'
    if initial.exists() and read(initial)!=bank_data:raise ValueError('Phase input bank changed')
    if not initial.exists():write(initial,bank_data)
    summary_name=getattr(args,'phase_summary_name','summary.json')
    bank=EvolutionBank(bank_data,args.embedding_url,args.options)
    policy=PlanningPolicy(bank,args.model_url,variant=='visual',update,args.options['planner_max_tokens'])
    adapter=None;rows=[];started=time.monotonic();pending=sum(not (root/t/'result.json').exists() or read(root/t/'result.json').get('status')!='completed' for t in task_ids);processed=0
    try:
        for index,task in enumerate(task_ids,1):
            folder=root/task;path=folder/'result.json'
            if path.exists() and read(path).get('status')=='completed':
                if read(folder/'input_task.json')!={'task_id':task,'phase':phase,'variant':variant,'round':round_number}:raise ValueError('Completed task provenance changed')
                bank.data=read(folder/'state_after.json');rows.append(read(path));continue
            if folder.exists():folder.rename(folder.with_name(task+f'_interrupted_{time.time_ns()}'))
            folder.mkdir();write(folder/'input_task.json',{'task_id':task,'phase':phase,'variant':variant,'round':round_number})
            if adapter is None:adapter=EvolutionAndroid(args,policy) if args.benchmark=='androidworld' else EvolutionDesktop(args,policy)
            print(f'START {args.benchmark}/{variant}/{phase}/round_{round_number:02d}/{task} {index}/{len(task_ids)}',flush=True)
            begin=time.monotonic();initialized=False;state_before=copy.deepcopy(bank.data);limit=None
            try:
                goal=adapter.setup(task);initialized=True
                canonical=args.root/'goals'/f'{task}.json'
                canonical.parent.mkdir(parents=True,exist_ok=True)
                with canonical.with_suffix('.lock').open('w') as goal_lock:
                    fcntl.flock(goal_lock,fcntl.LOCK_EX)
                    if canonical.exists() and read(canonical)['goal']!=goal:raise ValueError('Task goal changed across rounds or arms')
                    if not canonical.exists():write(canonical,{'task_id':task,'goal':goal})
                write(folder/'task.json',{'task_id':task,'goal':goal})
                limit=min(args.max_steps,args.complexities[task]) if args.benchmark=='androidworld' else args.max_steps
                if args.benchmark=='androidworld':out=adapter.run(folder,goal,{'text':'','images':[]},max_steps=limit)
                else:out=adapter.run(folder,goal,{})
                planner_files=list(folder.glob('step_*/planner/response_*.json')) if args.benchmark=='androidworld' else list(folder.glob('planner_*/response_*.json'))
                actor_files=list(folder.glob('step_*/response_*.json')) if args.benchmark=='androidworld' else []
                out['planner_calls']=len(planner_files)
                out['planner_usage']={key:sum((read(p).get('usage') or {}).get(key,0) or 0 for p in planner_files)
                                      for key in ('prompt_tokens','completion_tokens','total_tokens')}
                if actor_files:
                    out['actor_usage']={key:sum((read(p).get('usage') or {}).get(key,0) or 0 for p in actor_files)
                                        for key in ('prompt_tokens','completion_tokens','total_tokens')}
                out.update(status='completed',task_id=task,benchmark=args.benchmark,variant=variant,phase=phase,
                           round=round_number,step_limit=limit,seconds=time.monotonic()-begin)
            except (PlannerOutputError,APIConnectionError,APITimeoutError,InternalServerError) as exc:
                import traceback
                failure_type=('planner_invalid_output' if isinstance(exc,PlannerOutputError) else
                              'model_timeout' if isinstance(exc,APITimeoutError) else
                              'model_server_error' if isinstance(exc,InternalServerError) else 'model_connection_error')
                if failure_type!='planner_invalid_output':
                    # System-wide service failure must pause the phase rather than mark every task failed.
                    session=requests.Session();session.trust_env=False
                    try:
                        health=session.get(args.model_url.removesuffix('/v1')+'/health',timeout=10)
                        healthy=health.ok and health.json().get('ready') is True
                    except (requests.RequestException,ValueError):healthy=False
                    finally:session.close()
                    if not healthy:
                        write(path,{'status':'error','task_id':task,'failure_type':'model_service_unavailable',
                                    'error':str(exc),'traceback':traceback.format_exc()})
                        raise
                bank.data=state_before
                out={'status':'completed','outcome':'execution_error','failure_type':failure_type,
                     'error':str(exc),'traceback':traceback.format_exc(),'score':0.0,
                     'steps':len(list(folder.glob('step_*/action.json'))) if args.benchmark=='androidworld' else len(list(folder.glob('action_[0-9][0-9][0-9].json'))),
                     'task_id':task,'benchmark':args.benchmark,'variant':variant,'phase':phase,
                     'round':round_number,'step_limit':limit,'seconds':time.monotonic()-begin}
                print(f'TASK ERROR {task}: {failure_type}; memory state rolled back',flush=True)
            except Exception as exc:
                import traceback
                write(path,{'status':'error','task_id':task,'error':str(exc),'traceback':traceback.format_exc()})
                raise
            finally:
                if initialized:adapter.cleanup()
            # The result is committed last: completed always implies a durable
            # metadata checkpoint. Failed/interrupted tasks roll back on resume.
            if not update and bank.data!=bank_data:raise ValueError('Held-out evaluation mutated memory')
            write(folder/'state_after.json',bank.data);write(path,out);rows.append(out)
            processed+=1;elapsed=time.monotonic()-started;eta=elapsed/max(1,processed)*(pending-processed)
            finish=(datetime.now(timezone.utc)+timedelta(seconds=eta)).strftime('%Y-%m-%d %H:%M UTC')
            write(root/summary_name,rows)
            print(f'PROGRESS {args.benchmark}/{variant}/{phase} {index}/{len(task_ids)} score={out["score"]} elapsed={elapsed:.0f}s ETA={eta:.0f}s finish~{finish}',flush=True)
    finally:
        if adapter:adapter.close()
    if not update and bank.data!=bank_data:raise ValueError('Evaluation checkpoint changed bank')
    write(root/summary_name,rows)
    return bank,rows


def bootstrap(args):
    path=args.root/'bootstrap/bank.json'
    if path.exists():return read(path)
    source=WORKSPACE/'Experiments/memory_baselines/artifacts/gui_magnet_procedural'/args.benchmark/args.options['source_run']
    config=read(source/'config.json')
    if config.get('split_role')!='memory_construction' or read(source/'split.snapshot.json')!=args.manifest:
        raise ValueError('Initial No Memory source is not this frozen construction split')
    if set(config['tasks'])!=set(args.construction_ids):raise ValueError('Incomplete initial source set')
    episodes=[]
    for task in args.construction_ids:
        folder=source/task/'no_memory';result=read(folder/'result.json')
        if result['status']!='completed':raise ValueError('Initial source task is incomplete')
        episode=episode_from_result(folder,args.benchmark,task,read(source/task/'task.json')['goal'],result,'bootstrap/'+task)
        if episode:episodes.append(episode)
    bank=EvolutionBank(empty_bank(args.benchmark,args.construction_ids),args.embedding_url,args.options)
    update_from_episodes(bank,episodes,args.root/'bootstrap/build',args.model_url,args.options,0)
    bank.save(path)
    return bank.data


def bank_path(args,variant,round_number):
    return args.root/'bootstrap/bank.json' if round_number==0 else args.root/variant/f'round_{round_number:02d}/bank.json'


def train_round(args,variant,round_number):
    output=bank_path(args,variant,round_number)
    if output.exists():return
    parent=bank_path(args,variant,round_number-1)
    if not parent.exists():raise FileNotFoundError(f'Previous round is missing: {parent}')
    directory=output.parent
    bank,rows=run_phase(directory/'construction',args,args.train_ids,read(parent),'construction',variant,round_number,True)
    episodes=[]
    for row in rows:
        task=row['task_id'];folder=directory/'construction'/task
        episode=episode_from_result(folder,args.benchmark,task,read(folder/'task.json')['goal'],row,f'{variant}/round_{round_number:02d}/{task}')
        if episode:episodes.append(episode)
    update_from_episodes(bank,episodes,directory/'update',args.model_url,args.options,round_number)
    bank.save(output)


def evaluate_round(args,variant,round_number):
    source=bank_path(args,variant,round_number)
    if not source.exists():raise FileNotFoundError(source)
    data=read(source)
    if set(args.test_ids)&set(data['construction_task_ids']):raise ValueError('Construction/test leakage')
    before=source.read_bytes()
    run_phase(args.root/variant/'evaluation'/f'round_{round_number:02d}',args,args.test_ids,data,'heldout',variant,round_number,False)
    if source.read_bytes()!=before:raise ValueError('Evaluation modified the stored bank')


def frozen_control(args,variant,round_number):
    source=bank_path(args,variant,0)
    run_phase(args.root/variant/'control'/f'round_{round_number:02d}',args,args.train_ids,read(source),
              'frozen_control',variant,round_number,False)


def report(args, write_files=True):
    records=[]
    for variant in VARIANTS:
        for r in range(args.rounds+1):
            for phase in ('construction','heldout','frozen_control'):
                path=(args.root/variant/f'round_{r:02d}/construction/summary.json' if phase=='construction' else
                      args.root/variant/('evaluation' if phase=='heldout' else 'control')/f'round_{r:02d}/summary.json')
                if not path.exists():continue
                rows=read(path);bank=bank_path(args,variant,0 if phase=='frozen_control' else r)
                records.append({'variant':variant,'round':r,'phase':phase,'completed':len(rows),
                    'expected':len(args.test_ids if phase=='heldout' else args.train_ids),
                    'success':sum(row['score']==1 for row in rows),'mean_score':sum(row['score'] for row in rows)/max(1,len(rows)),
                    'mean_steps':sum(row['steps'] for row in rows)/max(1,len(rows)),
                    'mean_seconds':sum(row['seconds'] for row in rows)/max(1,len(rows)),
                    'bank_nodes':len(read(bank)['nodes']) if bank.exists() else None})
    for row in records:
        if row['phase']!='heldout':continue
        baseline=args.root/row['variant']/'evaluation/round_00/summary.json'
        current=args.root/row['variant']/'evaluation'/f'round_{row["round"]:02d}/summary.json'
        if baseline.exists():
            a={r['task_id']:r['score']==1 for r in read(baseline)};b={r['task_id']:r['score']==1 for r in read(current)}
            if set(a)==set(b)==set(args.test_ids):
                row['gained_vs_m0']=sum(not a[t] and b[t] for t in a)
                row['lost_vs_m0']=sum(a[t] and not b[t] for t in a)
    initial=args.root/'bootstrap/bank.json'
    heading=f'Shared M0 workflows: {len(read(initial)["nodes"])}' if initial.exists() else 'Shared M0 not built yet.'
    lines=['# MAGNET procedural evolution progress','',heading,'','| Variant | Round | Phase | Completed | Success | Nodes | Gained / Lost vs M0 |',
           '|---|---:|---|---:|---:|---:|---:|']
    for r in records:lines.append(f'| {r["variant"]} | {r["round"]} | {r["phase"]} | {r["completed"]}/{r["expected"]} | {r["success"]} | {r["bank_nodes"]} | {r.get("gained_vs_m0","—")} / {r.get("lost_vs_m0","—")} |')
    if write_files:
        write(args.root/'progress.json',records)
        (args.root/'report.md').write_text('\n'.join(lines)+'\n')
    else:print('\n'.join(lines))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=('androidworld','osworld'),required=True)
    p.add_argument('--stage',choices=('bootstrap','train','evaluate','control','probe','all','report'),default='all')
    p.add_argument('--run',default='magnet_evolution_v1')
    p.add_argument('--profile',choices=('formal','smoke'),default='formal')
    p.add_argument('--config',type=Path,default=WORKSPACE/'Experiments/memory_baselines/config/gui_magnet_evolution_v1.json')
    p.add_argument('--rounds',type=int)
    p.add_argument('--round',type=int)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--probe-task-ids',nargs='+',help='Held-out diagnostic only; does not change the fixed evaluation set')
    p.add_argument('--model-url');p.add_argument('--embedding-url',default='http://127.0.0.1:8087')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv);args.options=read(args.config)
    args.rounds=args.rounds if args.rounds is not None else (2 if args.profile=='smoke' else args.options['rounds'])
    if args.rounds<1 or (args.round is not None and not 0<=args.round<=args.rounds):raise ValueError('Invalid round')
    if args.stage=='train' and args.round==0:raise ValueError('Use bootstrap for round zero')
    if not 1<=args.options['top_k']<=args.options['top_n'] or args.options['max_images']<2:
        raise ValueError('Invalid retrieval/image budget')
    if not 0<=args.options['cluster_threshold']<=1 or not 0<=args.options['recall_threshold']<=1:
        raise ValueError('Invalid similarity threshold')
    args.model_url=args.model_url or ('http://127.0.0.1:8104/v1' if args.benchmark=='androidworld' else 'http://127.0.0.1:8105/v1')
    args.manifest=read(WORKSPACE/'Experiments/memory_baselines/splits/gui_50_50_seed20261026'/f'{args.benchmark}.json')
    args.construction_ids=flatten(args.manifest['memory_construction']);all_test=flatten(args.manifest['test'])
    if len(args.construction_ids)!=50 or len(all_test)!=50 or set(args.construction_ids)&set(all_test):raise ValueError('Invalid frozen split')
    if args.stage=='probe':
        if not args.probe_task_ids or not set(args.probe_task_ids)<=set(all_test):raise ValueError('Probe IDs must belong to held-out split')
    elif args.probe_task_ids:raise ValueError('--probe-task-ids is only valid with --stage probe')
    args.train_ids=args.construction_ids;args.test_ids=all_test
    if args.profile=='smoke':
        args.train_ids=args.options['smoke'][args.benchmark]['construction'];args.test_ids=args.options['smoke'][args.benchmark]['test']
        if not set(args.train_ids)<=set(args.construction_ids) or not set(args.test_ids)<=set(all_test):raise ValueError('Smoke task outside assigned split')
    args.root=WORKSPACE/'Experiments/memory_baselines/artifacts/gui_magnet_evolution'/args.benchmark/(args.run+'_'+args.profile)
    args.max_steps=args.options['android_max_steps'] if args.benchmark=='androidworld' else args.options['osworld_max_steps']
    if args.profile=='smoke':args.max_steps=min(args.max_steps,args.options['smoke_max_steps'])
    args.max_tokens=args.options['actor_max_tokens'];args.seed=args.manifest.get('suite_seed',20261026);args.task_index=args.manifest.get('task_index',0)
    args.vm_path=args.options['vm_path'];args.workspace=str(WORKSPACE);args.date_label=args.options['date_label']
    args.task_domains={t:app for app,ids in args.manifest['memory_construction'].items() for t in ids} if args.benchmark=='osworld' else {}
    if args.benchmark=='osworld':args.task_domains.update({t:app for app,ids in args.manifest['test'].items() for t in ids})
    args.complexities={r['task']:int(10*r['complexity']) for r in read(WORKSPACE/'Docs/实验方案/assets/20261026-gui-protocol/androidworld_registry_stats.json')['items']}
    snapshot={'benchmark':args.benchmark,'options':args.options,'rounds':args.rounds,'profile':args.profile,'model_url':args.model_url,
              'embedding_url':args.embedding_url,'manifest':args.manifest,'train_ids':args.train_ids,'test_ids':args.test_ids,
              'prompts':{'planner':PLANNER,'abstract':ABSTRACT,'verify':VERIFY,'actor_notice':ACTOR_NOTICE,'android_actor':ANDROID_ACTOR}}
    if args.dry_run:
        print(json.dumps({'root':str(args.root),'rounds':args.rounds,'construction_tasks':len(args.train_ids),'test_tasks':len(args.test_ids),
              'variants':args.variants,'model_url':args.model_url,'embedding_url':args.embedding_url,'stage':args.stage},indent=2));return
    args.root.mkdir(parents=True,exist_ok=True)
    config=args.root/'config.json'
    if config.exists() and read(config)!=snapshot:raise ValueError('Evolution configuration changed; choose a new --run')
    if not config.exists():write(config,snapshot)
    if args.stage=='report':report(args,False);return
    print(f'EVOLUTION RUN root={args.root} profile={args.profile} rounds={args.rounds} stage={args.stage}',flush=True)
    import requests
    session=requests.Session();session.trust_env=False
    embedding=session.get(args.embedding_url.rstrip('/')+'/health',timeout=20);embedding.raise_for_status()
    models=session.get(args.model_url.rstrip('/')+'/models',timeout=20);models.raise_for_status()
    if not embedding.json().get('normalized') or not any(m['id']=='qwen3-vl-8b-instruct' for m in models.json()['data']):
        raise ValueError('Unexpected model or embedding service')
    write(args.root/'services.json',{'embedding':embedding.json(),'models':models.json(),'checked_at':datetime.now(timezone.utc).isoformat()})
    lockroot=WORKSPACE/'Experiments/memory_baselines/artifacts/locks';lockroot.mkdir(parents=True,exist_ok=True)
    with (lockroot/('androidworld_http.lock' if args.benchmark=='androidworld' else 'osworld_vm.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage in ('all','bootstrap'):bootstrap(args)
        if args.stage=='all':
            for variant in args.variants:evaluate_round(args,variant,0)
        if args.stage in ('all','train'):
            for r in ([args.round] if args.round is not None else range(1,args.rounds+1)):
                # Alternate arm order across rounds; keep identical task order.
                for variant in (args.variants if r%2 else list(reversed(args.variants))):train_round(args,variant,r)
                if args.stage=='all':
                    for variant in args.variants:evaluate_round(args,variant,r)
                report(args)
        if args.stage=='evaluate':
            for r in ([args.round] if args.round is not None else range(args.rounds+1)):
                for variant in args.variants:evaluate_round(args,variant,r)
        if args.stage=='control':
            for r in ([args.round] if args.round is not None else range(1,args.rounds+1)):
                for variant in args.variants:frozen_control(args,variant,r)
        if args.stage=='probe':
            number=args.round if args.round is not None else 0
            for variant in args.variants:
                source=bank_path(args,variant,number);before=source.read_bytes()
                run_phase(args.root/variant/'probes'/f'round_{number:02d}',args,args.probe_task_ids,read(source),
                          'probe',variant,number,False)
                if source.read_bytes()!=before:raise ValueError('Probe modified memory')
        report(args)


if __name__=='__main__':main()
