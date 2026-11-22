"""Isolated AndroidWorld workers with deterministic round barriers.

Every task gets the same round-start memory and its own access counters.
At the barrier, replay actual selected IDs in task/step order, then build memory.
This protocol is explicit: it is not serial inter-task access-history equivalence.
"""
from __future__ import annotations
import copy
import fcntl
import hashlib
import json
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace

import requests

from .cold_continual import (ROOT,Episode,Model,argument_parser,check_stop,
                             report,task_selection,update_memory)
from .cold_memory import DualMemory,empty_memory
from .cold_vision import MiniCPMEncoder,RegionParser
from .common import read,write
from .runner import Android

PROTOCOL='round_snapshot_per_task__canonical_access_replay_v2'


def memory_digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def assignments(args,tasks):
    if args.workers<1 or args.build_workers<1:raise ValueError('Worker counts must be positive')
    count=min(args.workers,len(tasks))
    if len(args.android_urls)<count or len(args.android_containers)<count:
        raise ValueError('Every active worker requires an Android URL and a separate container')
    urls=args.android_urls[:count];containers=args.android_containers[:count]
    if len(set(urls))!=count or len(set(containers))!=count:raise ValueError('Android environments must be distinct')
    return [{'worker':i,'android_url':urls[i],'android_container':containers[i],'tasks':tasks[i::count]} for i in range(count)]


def merge_task_accesses(bank,folder,tasks):
    """Only completed task traces contribute, always in the frozen task order."""
    for task in tasks:
        target=folder/'tasks'/task
        read(target/'checkpoint.json')  # Missing task means the round cannot commit.
        for step in sorted(target.glob('step_*')):
            for kind in ('procedural','stationary'):
                path=step/(kind+'_retrieval.json')
                if path.exists():bank.replay_accesses(read(path))


def run_round(args,folder,tasks,budgets,snapshot,encoder,*,env_factory=Android,model_factory=Model,episode_factory=Episode):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    digest=memory_digest(snapshot)
    if (folder/'input_memory.json').exists() and memory_digest(read(folder/'input_memory.json'))!=digest:
        raise ValueError('Round input memory changed')
    write(folder/'input_memory.json',snapshot)
    progress_lock=threading.Lock();completed={}
    for task in tasks:
        checkpoint=folder/'tasks'/task/'checkpoint.json'
        if checkpoint.exists():
            row=read(checkpoint)
            if row['protocol']!=PROTOCOL or row['input_memory_sha256']!=digest:raise ValueError('Task checkpoint uses another memory protocol/snapshot')
            completed[task]=row['episode']
    started=time.monotonic();resumed=len(completed)
    def commit_progress():
        write(folder/'progress.json',{'completed':len(completed),'total':len(tasks),
            'successes':sum(e['success'] for e in completed.values()),'resumed':resumed,
            'workers':min(args.workers,len(tasks)),'wall_seconds':time.monotonic()-started})
    commit_progress()
    def worker(job):
        local=SimpleNamespace(**vars(args));local.android_url=job['android_url'];local.android_container=job['android_container']
        lockdir=ROOT/'Experiments/memory_baselines/artifacts/locks';lockdir.mkdir(parents=True,exist_ok=True)
        env=None
        with (lockdir/(job['android_container']+'.lock')).open('w') as envlock:
            fcntl.flock(envlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            model=model_factory(local)
            try:
                for task in job['tasks']:
                    check_stop(args)
                    if task in completed:continue
                    target=folder/'tasks'/task
                    if target.exists():target.rename(target.with_name(task+'_interrupted_'+str(time.time_ns())))
                    target.mkdir(parents=True)
                    if env is None:env=env_factory(local)
                    env.api('GET','/suite/reinitialize',params={'seed':args.round_seed,'n_task_combinations':1})
                    goal=env.setup(task)
                    write(target/'task.json',{'task_id':task,'goal':goal,'seed':args.round_seed,'budget':budgets[task],
                                              'worker':job['worker'],'android_container':local.android_container})
                    # No mutable bank is shared across tasks, even within one shard.
                    bank=DualMemory(snapshot,encoder,vars(args))
                    try:episode=episode_factory(local,env,model,bank).run(target,task,goal,budgets[task])
                    finally:env.cleanup()
                    check_stop(args)
                    write(target/'checkpoint.json',{'protocol':PROTOCOL,'input_memory_sha256':digest,
                          'episode':episode,'memory':bank.data})
                    with progress_lock:
                        completed[task]=episode;commit_progress()
                        print(f'PARALLEL PROGRESS round={args.round_number} worker={job["worker"]} '
                              f'{len(completed)}/{len(tasks)} successes={sum(e["success"] for e in completed.values())}',flush=True)
            finally:
                if env is not None:env.close()
    with ThreadPoolExecutor(max_workers=min(args.workers,len(tasks))) as pool:
        futures=[pool.submit(worker,j) for j in assignments(args,tasks)]
        succeeded=False
        try:
            for future in as_completed(futures):future.result()
            succeeded=True
        finally:
            if not succeeded:
                # Preserve the original exception while stopping other workers.
                args.stop_event.set()
                for future in futures:future.cancel()
    check_stop(args)
    episodes=[read(folder/'tasks'/task/'checkpoint.json')['episode'] for task in tasks]
    merged=DualMemory(snapshot,encoder,vars(args))
    merge_task_accesses(merged,folder,tasks)
    write(folder/'memory_after_access_replay.json',merged.data)
    return episodes,merged


def preflight(args,jobs):
    checks=[]
    targets=[('vllm',args.model_url+'/models')]+[(j['android_container'],j['android_url']+'/health') for j in jobs]
    for name,url in targets:
        try:
            response=requests.get(url,timeout=10);response.raise_for_status();body=response.json()
            checks.append({'check':name,'ok':name!='vllm' or args.model in [m['id'] for m in body['data']],'url':url})
        except requests.RequestException as exc:checks.append({'check':name,'ok':False,'url':url,'error':str(exc)})
    for name,path in [('embedding',args.embedding_model),('detector',args.detector_weights)]:
        checks.append({'check':name,'ok':Path(path).exists(),'path':str(path)})
    for job in jobs:
        status=subprocess.run(['docker','inspect','-f','{{.State.Running}}',job['android_container']],capture_output=True,text=True)
        checks.append({'check':job['android_container']+' process','ok':status.returncode==0 and status.stdout.strip()=='true'})
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    return all(c['ok'] for c in checks)


def main():
    p=argument_parser();p.description=__doc__
    p.set_defaults(run='easy61_empty_qwen3vl8b_parallel_v2')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--build-workers',type=int,default=8)
    p.add_argument('--android-urls',nargs='+',default=[f'http://127.0.0.1:{15006+i}' for i in range(4)])
    p.add_argument('--android-containers',nargs='+',default=['magnet_cold_aw']+[f'magnet_cold_aw_{i}' for i in range(1,4)])
    args=p.parse_args()
    if args.rounds<0 or not args.run or '/' in args.run or args.run in ('.','..'):p.error('Invalid rounds or run name')
    for kind in ('procedural','stationary'):
        if not 1<=getattr(args,kind+'_top_k')<=getattr(args,kind+'_top_n'):p.error('Require 1 <= top_k <= top_n')
    tasks,budgets=task_selection(args);jobs=assignments(args,tasks)
    root=ROOT/'Experiments/memory_baselines/artifacts/magnet_cold_continual'/args.run
    if args.stage=='plan':
        print(json.dumps({'root':str(root),'protocol':PROTOCOL,'tasks':len(tasks),'episodes':len(tasks)*(args.rounds+1),
              'workers':jobs,'build_workers':args.build_workers,'automatic_start':False},ensure_ascii=False,indent=2));return
    if args.stage=='report':report(root);return
    if args.stage=='preflight':raise SystemExit(0 if preflight(args,jobs) else 1)
    if args.stage!='run':p.error('Concurrent entry supports plan/preflight/run/report; use serial entry for model-smoke')
    if not preflight(args,jobs):raise SystemExit('Preflight failed; no benchmark episode executed')
    args.task_index=0;args.max_steps=max(budgets.values())
    proto=(ROOT/'Benchmark/AndroidWorld/android_world/task_evals/information_retrieval/proto/tasks.textproto').read_text()
    import re
    args.ir_tasks=re.findall(r'^  name: "([^"]+)"',proto,re.M)
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('stage','run')}
    options.update(protocol=PROTOCOL,tasks=tasks,budgets=budgets,worker_assignments=jobs,initialization='empty')
    sources=['cold_parallel.py','cold_continual.py','cold_memory.py','cold_vision.py','runner.py','common.py']
    options['source_sha256']={name:hashlib.sha256((ROOT/'Methods/MAGNET/src/gui_baseline'/name).read_bytes()).hexdigest() for name in sources}
    templates=['Methods/MAGNET/configs/fragments/base_modern.yaml','Methods/MAGNET/configs/fragments/modes/coat/coat_bbox.yaml',
               'Benchmark/AndroidWorld/android_world/task_metadata.json','Docs/实验方案/assets/20261026-gui-protocol/androidworld_registry_stats.json']
    options['input_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in templates}
    options['detector_sha256']=hashlib.sha256(args.detector_weights.read_bytes()).hexdigest()
    root.mkdir(parents=True,exist_ok=True)
    locks=ROOT/'Experiments/memory_baselines/artifacts/locks';locks.mkdir(parents=True,exist_ok=True)
    stop=threading.Event();args.stop_event=stop
    def interrupted(*_):
        stop.set();raise KeyboardInterrupt('Stop requested; completed task checkpoints are retained')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    with (root/'run.lock').open('w') as runlock,(locks/'androidworld_http.lock').open('w') as envlock:
        fcntl.flock(runlock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(envlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'config.json').exists() and read(root/'config.json')!=options:raise ValueError('Concurrent config changed; use a new run')
        write(root/'config.json',options)
        encoder=MiniCPMEncoder(args.embedding_model,args.device);parser=RegionParser(args.detector_weights,args.device)
        model=Model(args);bank=DualMemory(empty_memory(),encoder,vars(args));write(root/'initial_memory.json',bank.data)
        for number in range(args.rounds+1):
            check_stop(args);folder=root/f'round_{number:02d}';folder.mkdir(exist_ok=True)
            if (folder/'summary.json').exists():
                bank=DualMemory(read(folder/'memory.json'),encoder,vars(args));continue
            args.round_number=number;args.round_seed=args.seed+(number if args.instance_policy=='new-per-round' else 0)
            start=time.monotonic()
            episodes,bank=run_round(args,folder,tasks,budgets,bank.data,encoder)
            if number<args.rounds:update_memory(args,bank,episodes,folder/'build',model,parser,number)
            check_stop(args);bank.data['completed_round']=number
            write(folder/'memory.json',bank.data)
            write(folder/'summary.json',{'round':number,'protocol':PROTOCOL,'episodes':episodes,
                'success_rate':sum(e['success'] for e in episodes)/len(tasks),'workers':len(jobs),
                'build_workers':args.build_workers,'wall_seconds':time.monotonic()-start,
                'procedural_units':len(bank.data['procedural']),'stationary_functions':len(bank.data['stationary']),
                'stationary_images':sum(len(n['images']) for n in bank.data['stationary'])})
            report(root)


if __name__=='__main__':main()
