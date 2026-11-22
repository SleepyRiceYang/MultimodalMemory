"""Read-only MAGNET evaluation with one isolated environment per process.

Construction deliberately remains ordered. Existing formal artifacts are inputs only.
"""
from __future__ import annotations
import argparse,copy,fcntl,json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace
from .common import read,write
from .evolution import WORKSPACE,flatten,run_phase


def shards(tasks, workers):
    if workers<1 or not tasks or len(set(tasks))!=len(tasks):
        raise ValueError('Expected unique tasks and positive workers')
    return [tasks[i::workers] for i in range(min(workers,len(tasks)))]


def worker(job):
    args=SimpleNamespace(**job['args']);args.root=Path(args.root)
    lockroot=WORKSPACE/'Experiments/memory_baselines/artifacts/locks';lockroot.mkdir(parents=True,exist_ok=True)
    # Android services are persistent; prevent two workers from resetting one emulator.
    name=(args.android_container if args.benchmark=='androidworld' else job['environment_id'])
    if name=='android_world_smoke':name='androidworld_http'
    with (lockroot/(name+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bank,rows=run_phase(Path(job['output']),args,job['tasks'],job['bank'],
                            'evaluation',job['variant'],job['round'],False)
    if bank.data!=job['bank']:raise ValueError('Read-only memory changed')
    return {'worker':job['environment_id'],'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=['androidworld','osworld'],required=True)
    p.add_argument('--workers',type=int,default=2)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--tasks',nargs='+')
    group.add_argument('--all-test-tasks',action='store_true')
    p.add_argument('--variant',choices=['text','visual'],default='text')
    p.add_argument('--bank',type=Path,help='Frozen bank; defaults to existing formal M0')
    p.add_argument('--round',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-url',default='http://127.0.0.1:8103/v1')
    p.add_argument('--embedding-url',default='http://127.0.0.1:8087')
    p.add_argument('--android-urls',nargs='+',default=['http://127.0.0.1:15003','http://127.0.0.1:15004'])
    p.add_argument('--android-containers',nargs='+',default=['gui_async_aw_0','gui_async_aw_1'])
    p.add_argument('--max-steps',type=int,help='Explicit smoke cap; omitted uses formal budget')
    args=p.parse_args()
    if args.max_steps is not None and args.max_steps<1:raise ValueError('max-steps must be positive')
    if args.round<0 or (args.round>0 and args.bank is None):raise ValueError('For nonzero rounds supply the matching --bank')
    options=read(WORKSPACE/'Experiments/memory_baselines/config/gui_magnet_evolution_v1.json')
    manifest=read(WORKSPACE/'Experiments/memory_baselines/splits/gui_50_50_seed20261026'/f'{args.benchmark}.json')
    if args.all_test_tasks:args.tasks=flatten(manifest['test'])
    groups=shards(args.tasks,args.workers)
    if not set(args.tasks)<=set(flatten(manifest['test'])):raise ValueError('Only held-out tasks are allowed')
    if args.benchmark=='androidworld':
        if len(args.android_urls)<len(groups) or len(args.android_containers)<len(groups):raise ValueError('Each worker needs a separate Android environment')
        if len(set(args.android_urls[:len(groups)]))!=len(groups) or len(set(args.android_containers[:len(groups)]))!=len(groups):raise ValueError('Duplicate Android environment')
    bank_path=(args.bank or WORKSPACE/'Experiments/memory_baselines/artifacts/gui_magnet_evolution'/args.benchmark/'magnet_evolution_v1_formal/bootstrap/bank.json').resolve()
    original=bank_path.read_bytes();bank=json.loads(original)
    if bank['benchmark']!=args.benchmark:raise ValueError('Bank benchmark mismatch')
    if set(bank['construction_task_ids'])!=set(flatten(manifest['memory_construction'])):raise ValueError('Bank construction split mismatch')
    complexities={r['task']:int(10*r['complexity']) for r in read(WORKSPACE/'Docs/实验方案/assets/20261026-gui-protocol/androidworld_registry_stats.json')['items']}
    domains={t:d for split in ['memory_construction','test'] for d,ids in manifest[split].items() for t in ids} if args.benchmark=='osworld' else {}
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    spec={**vars(args),'bank':str(bank_path),'output':str(output),'frozen_bank':bank,'options':options,'groups':groups}
    spec=json.loads(json.dumps(spec,default=str))
    with (output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'config.json').exists() and read(output/'config.json')!=spec:raise ValueError('Output configuration changed')
        write(output/'config.json',spec)
        if (output/'summary.json').exists():
            print('Already completed:',output);return
        jobs=[]
        for i,group in enumerate(groups):
            root=output/f'worker_{i:02d}'
            params=dict(benchmark=args.benchmark,model_url=args.model_url,embedding_url=args.embedding_url,
                root=str(root),workspace=str(WORKSPACE),options=options,vm_path=options['vm_path'],
                seed=manifest.get('suite_seed',20261026),task_index=manifest.get('task_index',0),
                max_steps=args.max_steps or options['android_max_steps' if args.benchmark=='androidworld' else 'osworld_max_steps'],
                max_tokens=options['actor_max_tokens'],date_label=options['date_label'],complexities=complexities,
                task_domains=domains,android_url=args.android_urls[i] if args.benchmark=='androidworld' else '',
                android_container=args.android_containers[i] if args.benchmark=='androidworld' else '')
            jobs.append(dict(args=params,tasks=group,bank=copy.deepcopy(bank),variant=args.variant,round=args.round,
                             output=str(root/'tasks'),environment_id=f'{output.name}_{i}'))
        resumed=sum((Path(j['output'])/task/'result.json').exists() and read(Path(j['output'])/task/'result.json').get('status')=='completed' for j in jobs for task in j['tasks'])
        start=time.monotonic();rows=[];failures=[]
        with ProcessPoolExecutor(max_workers=len(groups)) as pool:
            futures={pool.submit(worker,j):j for j in jobs}
            for future in as_completed(futures):
                try:rows.extend(future.result()['rows'])
                except Exception as exc:
                    failures.append({'worker':futures[future]['environment_id'],'error':str(exc),'traceback':traceback.format_exc()})
                    write(output/'errors.json',failures)
        if bank_path.read_bytes()!=original:raise ValueError('Source bank changed during evaluation')
        if failures:raise RuntimeError('Workers failed; inspect errors.json; rerun resumes completed tasks')
        if len(rows)!=len(args.tasks) or {r['task_id'] for r in rows}!=set(args.tasks):raise ValueError('Task coverage mismatch')
        rows.sort(key=lambda r:args.tasks.index(r['task_id']))
        summary=dict(benchmark=args.benchmark,variant=args.variant,workers=len(groups),tasks=len(rows),
                     wall_seconds=time.monotonic()-start,task_seconds=sum(r['seconds'] for r in rows),
                     resumed_tasks=resumed,wall_seconds_scope='this invocation including worker startup',
                     successes=sum(r['score']==1 for r in rows),frozen_bank_unchanged=True,rows=rows)
        write(output/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
