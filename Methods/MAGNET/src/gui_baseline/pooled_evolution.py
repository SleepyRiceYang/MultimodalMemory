"""Resume existing MAGNET checkpoints using shared inference and isolated GUI workers.

Construction is serial within each variant; variants/benchmarks can overlap.
Held-out tasks may be sharded because content and access metadata are frozen.
"""
from __future__ import annotations
import argparse,copy,fcntl,json,os,signal,subprocess,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace
from .common import read,write
from .evolution import WORKSPACE,run_phase,train_round,bank_path,report,PLANNER,ABSTRACT,VERIFY,ACTOR_NOTICE,ANDROID_ACTOR,PLANNER_SCHEMA
from .parallel_eval import shards

BASE=WORKSPACE/'Experiments/memory_baselines/artifacts/gui_magnet_evolution'
STOP=threading.Event()
PYTHONS={'androidworld':'/root/anaconda3/envs/vision2code/bin/python','osworld':'/root/anaconda3/envs/osworld/bin/python'}


def load_args(benchmark,run,model_url,profile='formal'):
    root=BASE/benchmark/(run+'_'+profile);snapshot=read(root/'config.json')
    prompts={'planner':PLANNER,'abstract':ABSTRACT,'verify':VERIFY,'actor_notice':ACTOR_NOTICE,'android_actor':ANDROID_ACTOR}
    if snapshot['benchmark']!=benchmark or snapshot['prompts']!=prompts:
        raise ValueError('Checkpoint benchmark/prompts differ; refusing to mix experiments')
    manifest=snapshot['manifest'];options=snapshot['options']
    args=SimpleNamespace(benchmark=benchmark,root=root,options=options,manifest=manifest,
        model_url=model_url,embedding_url=snapshot['embedding_url'],rounds=snapshot['rounds'],
        train_ids=snapshot['train_ids'],test_ids=snapshot['test_ids'],
        construction_ids=list(manifest['memory_construction']) if benchmark=='androidworld' else
            [t for ids in manifest['memory_construction'].values() for t in ids],
        seed=manifest.get('suite_seed',20261026),task_index=manifest.get('task_index',0),
        max_steps=options['android_max_steps' if benchmark=='androidworld' else 'osworld_max_steps'],
        max_tokens=options['actor_max_tokens'],workspace=str(WORKSPACE),vm_path=options['vm_path'],date_label=options['date_label'])
    if profile=='smoke':args.max_steps=min(args.max_steps,options['smoke_max_steps'])
    args.task_domains={t:d for split in ['memory_construction','test'] for d,ids in manifest[split].items() for t in ids} if benchmark=='osworld' else {}
    args.complexities={r['task']:int(10*r['complexity']) for r in read(WORKSPACE/'Docs/实验方案/assets/20261026-gui-protocol/androidworld_registry_stats.json')['items']}
    return args


def serialize(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


def completed_count(folder,tasks):
    return sum((Path(folder)/t/'result.json').exists() and read(Path(folder)/t/'result.json').get('status')=='completed' for t in tasks)


def run_job(path):
    spec=read(path);args=SimpleNamespace(**spec['args']);args.root=Path(args.root)
    def interrupted(*_):raise KeyboardInterrupt('Worker interrupted; unfinished task will be rolled back')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    lockroot=WORKSPACE/'Experiments/memory_baselines/artifacts/locks';lockroot.mkdir(parents=True,exist_ok=True)
    envname=args.android_container if args.benchmark=='androidworld' else spec['job_id']
    with (lockroot/(envname+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if spec['stage']=='train':train_round(args,spec['variant'],spec['round'])
        else:
            source=bank_path(args,spec['variant'],spec['round']);before=source.read_bytes()
            bank,rows=run_phase(Path(spec['phase_root']),args,spec['tasks'],read(source),
                'heldout',spec['variant'],spec['round'],False)
            if source.read_bytes()!=before or bank.data!=json.loads(before):raise ValueError('Test modified memory')
        write(Path(path).with_suffix('.done.json'),{'status':'completed','job':spec['job_id']})


def launch_job(args,stage,variant,round_number,slot,tasks=None,shard=0):
    if STOP.is_set():raise RuntimeError('Scheduler stopping')
    child=copy.deepcopy(args)
    child.android_url=f'http://127.0.0.1:{15003+slot}'
    child.android_container=f'gui_async_aw_{slot}'
    progress_tasks=tasks if stage=='evaluate' else args.train_ids
    progress_root=(args.root/variant/'evaluation'/f'round_{round_number:02d}' if stage=='evaluate' else args.root/variant/f'round_{round_number:02d}'/'construction')
    resumed=completed_count(progress_root,progress_tasks)
    job_id=f'{args.benchmark}_{stage}_{variant}_r{round_number:02d}_s{shard}'
    directory=args.root/'pool_runtime';directory.mkdir(parents=True,exist_ok=True)
    spec=dict(args=serialize(child),stage=stage,variant=variant,round=round_number,job_id=job_id)
    if stage=='evaluate':
        phase=args.root/variant/'evaluation'/f'round_{round_number:02d}'
        source=bank_path(args,variant,round_number)
        if not source.exists():raise FileNotFoundError(source)
        spec.update(phase_root=str(phase),tasks=tasks)
        spec['args']['phase_summary_name']=f'summary.pool_shard_{shard}.json'
    path=directory/(job_id+'.json');write(path,spec)
    env={**os.environ,'PYTHONPATH':str(WORKSPACE/'Methods/MAGNET/src')+':'+str(WORKSPACE/'Benchmark/OSWorld'),
         'NO_PROXY':'127.0.0.1,localhost','no_proxy':'127.0.0.1,localhost','GUI_POOL_CLIENT_TAG':args.benchmark,
         'OSWORLD_MAX_RETRY_TIMES':'2','OSWORLD_OPENAI_TIMEOUT':'180','CUDA_VISIBLE_DEVICES':''}
    cmd=[PYTHONS[args.benchmark],'-u','-m','gui_baseline.pooled_evolution','--job',str(path)]
    print(f'POOL JOB START {job_id} env_slot={slot}',flush=True)
    with (directory/(job_id+'.log')).open('a') as log:
        proc=subprocess.Popen(cmd,cwd=WORKSPACE/'Benchmark/OSWorld',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(directory/(job_id+'.pid.json'),{'pid':proc.pid,'started':time.time(),'command':cmd})
        interrupted=False;deadline=None;started=time.monotonic();last_update=started
        while proc.poll() is None:
            now=time.monotonic()
            if now-last_update>=30:
                n=completed_count(progress_root,progress_tasks);new=n-resumed;remaining=len(progress_tasks)-n
                eta=f'{(now-started)/new*remaining:.0f}s' if new>0 else ('0s' if remaining==0 else 'pending first completion')
                print(f'POOL PROGRESS {job_id} {n}/{len(progress_tasks)} resumed={resumed} elapsed={now-started:.0f}s ETA={eta}',flush=True);last_update=now
            if STOP.is_set() and not interrupted:
                proc.send_signal(signal.SIGINT);interrupted=True;deadline=time.monotonic()+40
            if deadline and time.monotonic()>deadline:
                os.killpg(proc.pid,signal.SIGKILL);break
            time.sleep(1)
        code=proc.wait()
    if code:raise RuntimeError(f'{job_id} exited {code}; inspect {directory/(job_id+".log")}')
    print(f'POOL JOB DONE {job_id}',flush=True)


def evaluate_parallel(args,round_number):
    # Two workers per benchmark. Each frozen variant is split into two disjoint shards.
    for variant in ['text','visual']:
        phase=args.root/variant/'evaluation'/f'round_{round_number:02d}'
        if completed_count(phase,args.test_ids)==len(args.test_ids):
            print(f'SKIP complete {args.benchmark}/{variant}/test/r{round_number}',flush=True);continue
        phase.mkdir(parents=True,exist_ok=True)
        data=read(bank_path(args,variant,round_number))
        initial=phase/'input_bank.json'
        if initial.exists() and read(initial)!=data:raise ValueError('Evaluation input bank changed')
        if not initial.exists():write(initial,data)
        groups=shards(args.test_ids,2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(launch_job,args,'evaluate',variant,round_number,i,tasks,i) for i,tasks in enumerate(groups)]
            for f in as_completed(futures):f.result()
        rows=[read(phase/t/'result.json') for t in args.test_ids]
        if any(r['status']!='completed' for r in rows):raise ValueError('Incomplete evaluation')
        if any(read(phase/t/'state_after.json')!=data for t in args.test_ids):raise ValueError('Evaluation checkpoint mutated memory')
        write(phase/'summary.json',rows)


def evolve(args):
    runtime=args.root/'pool_runtime';runtime.mkdir(parents=True,exist_ok=True)
    write(runtime/'transport.json',{'model_url':args.model_url,'backends':[8111,8113,8114],
        'gpus':[1,3,4],'workers_per_benchmark':2,'within_variant_construction':'ordered',
        'evaluation':'two frozen shards','local_worker_cuda_visible_devices':'',
        'local_evaluator_device':'CPU; VLM and embedding remain remote','planner_format':'original JSON-object first; constrained repair only after invalid output',
        'planner_repair_schema':PLANNER_SCHEMA,
        'original_config_unchanged':True,'activated':time.time()})
    evaluate_parallel(args,0)
    for r in range(1,args.rounds+1):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(launch_job,args,'train',variant,r,slot) for slot,variant in enumerate(['text','visual'] if r%2 else ['visual','text'])]
            for f in as_completed(futures):f.result()
        evaluate_parallel(args,r);report(args)
    report(args)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path)
    p.add_argument('--run',default='magnet_evolution_v1');p.add_argument('--profile',choices=['formal','smoke'],default='formal')
    p.add_argument('--benchmarks',nargs='+',choices=['androidworld','osworld'],default=['androidworld','osworld'])
    p.add_argument('--model-url',default='http://127.0.0.1:8110/v1');p.add_argument('--status',action='store_true')
    args=p.parse_args()
    if args.job:return run_job(args.job)
    specs=[load_args(b,args.run,args.model_url,args.profile) for b in args.benchmarks]
    if not args.status:print(f'POOL RUN run={args.run} profile={args.profile} benchmarks={args.benchmarks} model_url={args.model_url}',flush=True)
    if args.status:
        for spec in specs:
            print(f'\n{spec.benchmark}: {spec.root}')
            print('variant | round | phase | committed | success | bank ready')
            for variant in ['text','visual']:
                for r in range(spec.rounds+1):
                    for phase,folder,tasks in [('construction',spec.root/variant/f'round_{r:02d}'/'construction',spec.train_ids),('heldout',spec.root/variant/'evaluation'/f'round_{r:02d}',spec.test_ids)]:
                        if not folder.exists():continue
                        rows=[read(folder/t/'result.json') for t in tasks if (folder/t/'result.json').exists() and read(folder/t/'result.json').get('status')=='completed']
                        print(f'{variant} | {r} | {phase} | {len(rows)}/{len(tasks)} | {sum(x["score"]==1 for x in rows)} | {bank_path(spec,variant,r).exists()}')
        return
    import requests
    session=requests.Session();session.trust_env=False
    health=session.get(args.model_url.removesuffix('/v1')+'/health',timeout=30);health.raise_for_status()
    if not health.json()['ready']:raise RuntimeError('Shared pool is not ready')
    for spec in specs:
        embedding=session.get(spec.embedding_url.rstrip('/')+'/health',timeout=20);embedding.raise_for_status()
        if not embedding.json().get('normalized'):raise ValueError('Unexpected embedding service')
    if 'androidworld' in args.benchmarks:
        for port in [15003,15004]:
            ready=session.get(f'http://127.0.0.1:{port}/docs',timeout=10);ready.raise_for_status()
    # Holding old locks prevents accidentally starting the legacy serial scripts concurrently.
    locks=[]
    try:
        for spec in specs:
            for name in [spec.benchmark+'_pool_scheduler', 'androidworld_http' if spec.benchmark=='androidworld' else 'osworld_vm']:
                path=WORKSPACE/'Experiments/memory_baselines/artifacts/locks'/f'{name}.lock';path.parent.mkdir(parents=True,exist_ok=True)
                lock=path.open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        def stop(*_):STOP.set()
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures={pool.submit(evolve,spec):spec.benchmark for spec in specs}
            errors=[]
            for f in as_completed(futures):
                try:f.result()
                except Exception as exc:
                    errors.append({'benchmark':futures[f],'error':str(exc),'traceback':traceback.format_exc()})
                    print(json.dumps(errors[-1],ensure_ascii=False),flush=True)
            if errors:raise RuntimeError('Some benchmarks stopped; completed checkpoints are retained')
    finally:
        for lock in locks:lock.close()

if __name__=='__main__':main()
