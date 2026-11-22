"""Matched five-task GUI comparison: No Memory, MAGNET procedural text/visual."""
from __future__ import annotations
import argparse,fcntl,time,traceback
from datetime import datetime,timezone,timedelta
from pathlib import Path
from .common import read,write
from .memory import ProceduralBank
from .runner import AW,OS,Android,Desktop
from .prompts import ANDROID_ACTOR

CONDITIONS=('no_memory','magnet_procedural_text','magnet_procedural_visual')
EMPTY={'text':'','images':[],'selected':[],'candidates':[],'ranking':'none','stationary_memory':False}


def score_rows(root,tasks):
    rows=[]
    for task in tasks:
        for condition in CONDITIONS:
            path=root/task/condition/'result.json'
            if path.exists():rows.append(read(path))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=('androidworld','osworld'),required=True)
    p.add_argument('--condition',choices=CONDITIONS+('all',),required=True)
    p.add_argument('--workspace',default='/root/VLM-Memory')
    p.add_argument('--run',default='comparison_5_v1')
    p.add_argument('--seed',type=int,default=20261028)
    p.add_argument('--task-index',type=int,default=0)
    p.add_argument('--split-file',help='Frozen GUI 50/50 manifest')
    p.add_argument('--split-role',choices=('memory_construction','test'))
    p.add_argument('--bank-dir',help='Explicit bank path for split-based memory evaluation')
    p.add_argument('--source-run',help='Required provenance for a formal memory bank')
    p.add_argument('--android-step-policy',choices=('fixed','complexity10'),default='fixed')
    p.add_argument('--max-steps',type=int,default=30)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--embedding-url',default='http://127.0.0.1:8087')
    p.add_argument('--model-url')
    p.add_argument('--task-ids',nargs='+',help='Run only these tasks from the frozen split; keep the full run configuration')
    p.add_argument('--vm-path',default='/data/yangshengjie/workspace/VLM-Memory/Benchmark/OSWorld/docker_vm_data/Ubuntu.qcow2')
    p.add_argument('--dry-run',action='store_true',help='Print planned tasks and paths without starting environments or models')
    args=p.parse_args(argv)
    args.model_url=args.model_url or ('http://127.0.0.1:8104/v1' if args.benchmark=='androidworld' else 'http://127.0.0.1:8105/v1')
    manifest=read(args.split_file) if args.split_file else None
    if manifest:
        if not args.split_role or manifest['benchmark']!=args.benchmark:raise ValueError('Split role/benchmark mismatch')
        if args.benchmark=='androidworld':
            tasks=manifest[args.split_role];args.seed=manifest['suite_seed'];args.task_index=manifest['task_index'];args.task_domains={}
        else:
            selected_ids=manifest[args.split_role]
            tasks=[task for domain in sorted(selected_ids) for task in selected_ids[domain]]
            args.task_domains={task:domain for domain,ids in selected_ids.items() for task in ids}
        if len(tasks)!=50 or len(set(tasks))!=50:raise ValueError('Expected 50 unique tasks in split role')
        construction=manifest['memory_construction'];heldout=manifest['test']
        ids=lambda x:set(x) if args.benchmark=='androidworld' else {task for values in x.values() for task in values}
        if len(ids(construction))!=50 or len(ids(heldout))!=50 or ids(construction)&ids(heldout):raise ValueError('Invalid split isolation')
    else:
        if args.split_role:raise ValueError('--split-role requires --split-file')
        tasks=AW if args.benchmark=='androidworld' else OS;args.task_domains={}
    selected_tasks=set(args.task_ids or tasks)
    if not selected_tasks or selected_tasks-set(tasks):
        raise ValueError('--task-ids must belong to the selected split')
    complexities={}
    if args.android_step_policy=='complexity10':
        if args.benchmark!='androidworld':raise ValueError('complexity10 only applies to AndroidWorld')
        stats=read(Path(args.workspace)/'Docs/实验方案/assets/20261026-gui-protocol/androidworld_registry_stats.json')
        complexities={x['task']:int(10*x['complexity']) for x in stats['items']}
        if any(task not in complexities for task in tasks):raise ValueError('Missing Android task complexity')
    selected=list(CONDITIONS) if args.condition=='all' else [args.condition]
    if manifest and args.split_role=='memory_construction' and selected!=['no_memory']:
        raise ValueError('Construction source trajectories must use No Memory')
    base=Path(args.workspace)/'Experiments/memory_baselines/artifacts/gui_magnet_procedural'/args.benchmark
    root=base/args.run
    if args.dry_run:
        planned=[task for task in tasks if task in selected_tasks]
        print({'benchmark':args.benchmark,'conditions':selected,'task_count':len(planned),'first_task':planned[0],'last_task':planned[-1],'run':str(root),
               'model_url':args.model_url,'embedding_url':args.embedding_url,'bank':str(Path(args.bank_dir) if args.bank_dir else base/'bank'), 'role':args.split_role})
        return
    lockpath=Path(args.workspace)/'Experiments/memory_baselines/artifacts/locks';lockpath.mkdir(parents=True,exist_ok=True)
    with (lockpath/(args.benchmark+'_http.lock' if args.benchmark=='androidworld' else 'osworld_vm.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir(parents=True,exist_ok=True)
        config={**{k:v for k,v in vars(args).items() if k not in ('condition','dry_run','task_ids')},
                'tasks':tasks,'conditions':list(CONDITIONS),'actor_prompt':ANDROID_ACTOR if args.benchmark=='androidworld' else 'native mm_agents.qwen',
                'scoring':'Android explicit termination + official state score; OS official env.evaluate',
                'retrieval':'frozen goal-query cosine; text and visual share retrieved nodes',
                'history':'Android current screenshot + last 10 actions; OS native last 3 screenshots',
                'stationary_memory':False,'omniparser_used':False,'visual_evidence':'workflow BEFORE/AFTER; max 2 images'}
        snapshot=root/'config.json'
        if snapshot.exists() and read(snapshot)!=config:raise ValueError('Run configuration changed; select a new --run')
        write(snapshot,config)
        if manifest:
            copy=root/'split.snapshot.json'
            if copy.exists() and read(copy)!=manifest:raise ValueError('Split manifest changed within run')
            write(copy,manifest)
        bank=ProceduralBank(Path(args.bank_dir) if args.bank_dir else base/'bank',args.embedding_url) if any(c!='no_memory' for c in selected) else None
        if bank and manifest:
            source_ids=set(bank.data['source_task_ids'])
            expected=ids(manifest['memory_construction'])
            if source_ids!=expected:raise ValueError('Memory bank construction IDs do not match frozen split')
            if not args.source_run or bank.data.get('source_run')!=args.source_run:
                raise ValueError('Memory bank source run does not match the required construction run')
        adapter=Android(args) if args.benchmark=='androidworld' else Desktop(args)
        pending=sum(1 for c in selected for task in selected_tasks if not (root/task/c/'result.json').exists()
                    or read(root/task/c/'result.json').get('status')!='completed')
        start=time.monotonic();processed=0
        try:
            for condition in selected:
                for index,task in enumerate(tasks,1):
                    if task not in selected_tasks:continue
                    folder=root/task/condition;folder.mkdir(parents=True,exist_ok=True)
                    result_path=folder/'result.json'
                    if result_path.exists() and read(result_path).get('status')=='completed':
                        print(f'SKIP {args.benchmark}/{condition}/{task}: completed',flush=True)
                        continue
                    if any(folder.iterdir()):
                        archived=folder.with_name(f'{condition}_interrupted_{time.time_ns()}')
                        folder.rename(archived);folder.mkdir()
                    result={'benchmark':args.benchmark,'task_id':task,'condition':condition}
                    episode_start=time.monotonic();initialized=False
                    print(f'START {args.benchmark}/{condition}/{task} {index}/{len(tasks)}',flush=True)
                    try:
                        goal=adapter.setup(task);initialized=True
                        task_record=root/task/'task.json'
                        if task_record.exists() and read(task_record)['goal']!=goal:
                            raise ValueError('Goal changed between comparison conditions')
                        write(task_record,{'task_id':task,'goal':goal,'seed':args.seed if args.benchmark=='androidworld' else None})
                        if condition=='no_memory':
                            context=dict(EMPTY)
                        else:
                            retrieval=root/task/'retrieval.json'
                            if retrieval.exists():
                                shared=read(retrieval)
                                if shared['goal']!=goal:raise ValueError('Retrieval goal changed')
                            else:
                                shared={**bank.select(args.benchmark,task,goal),'goal':goal}
                                write(retrieval,shared)
                            context=bank.view(shared,condition=='magnet_procedural_visual')
                        write(folder/'memory.json',context)
                        step_limit=min(args.max_steps,complexities[task]) if args.benchmark=='androidworld' and args.android_step_policy=='complexity10' else args.max_steps
                        result.update(adapter.run(folder,goal,context,max_steps=step_limit) if args.benchmark=='androidworld' else adapter.run(folder,goal,context))
                        result['step_limit']=step_limit
                        result.update(status='completed',selected_ids=[n['id'] for n in context['selected']],
                                      historical_images=len(context['images']))
                    except Exception as exc:
                        result.update(status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                        print(result['traceback'],flush=True)
                    finally:
                        if initialized:
                            try:adapter.cleanup()
                            except Exception as exc:write(folder/'cleanup_error.json',{'error':str(exc)})
                    result['seconds']=time.monotonic()-episode_start;write(result_path,result)
                    if bank:bank.verify()
                    processed+=1;rows=score_rows(root,tasks);write(root/'summary.json',rows)
                    elapsed=time.monotonic()-start;remaining=(pending-processed)*elapsed/max(processed,1)
                    finish=(datetime.now(timezone.utc)+timedelta(seconds=remaining)).strftime('%Y-%m-%d %H:%M UTC')
                    print(f'PROGRESS {args.benchmark}/{condition} {index}/{len(tasks)} score={result.get("score")} elapsed={elapsed:.0f}s ETA={remaining:.0f}s finish~{finish}',flush=True)
        finally:adapter.close()
        checks=[]
        for task in tasks:
            a=root/task/'magnet_procedural_text/memory.json';b=root/task/'magnet_procedural_visual/memory.json';n=root/task/'no_memory/memory.json'
            if a.exists() and b.exists():
                x,y=read(a),read(b)
                checks.append({'task_id':task,'text_visual_same_text':x['text']==y['text'],
                               'text_visual_same_nodes':x['selected']==y['selected'],
                               'text_historical_images':len(x['images']),'visual_historical_images':len(y['images']),
                               'no_memory_images':len(read(n)['images']) if n.exists() else None})
        previous=read(root/'pairing_checks.json') if (root/'pairing_checks.json').exists() else {}
        write(root/'pairing_checks.json',{'frozen_bank':bank.verify() if bank else previous.get('frozen_bank'),'checks':checks})
        rows=score_rows(root,tasks);write(root/'summary.json',rows)
        errors=[r for r in rows if r['condition'] in selected and r['task_id'] in selected_tasks and r['status']=='error']
        if errors:raise RuntimeError(f'{len(errors)} selected condition(s) failed; inspect result.json and rerun the same script')

if __name__=='__main__':main()
