#!/root/magnet_cold_env/bin/python
"""Repeat each observed success-to-failure case with its original round input memory."""
import fcntl, json, os, re, sys, threading
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/root/VLM-Memory')
sys.path.insert(0,str(ROOT/'Methods/MAGNET/src'))
from gui_baseline.cold_continual import argument_parser, task_selection
from gui_baseline.cold_parallel import assignments, preflight, run_round
from gui_baseline.cold_vision import MiniCPMEncoder
from gui_baseline.common import read, write

source=ROOT/'Experiments/memory_baselines/artifacts/magnet_cold_continual/easy61_empty_qwen3vl8b_parallel_v2'
out=ROOT/'Experiments/memory_baselines/artifacts/magnet_regression_repeats_v1'
args=argument_parser().parse_args([])
args.workers=4; args.build_workers=8
args.android_urls=[f'http://127.0.0.1:{15006+i}' for i in range(4)]
args.android_containers=['magnet_cold_aw']+[f'magnet_cold_aw_{i}' for i in range(1,4)]
args.task_index=0; args.round_seed=args.seed; args.stop_event=threading.Event()
all_tasks,budgets=task_selection(args); args.max_steps=max(budgets.values())
proto=(ROOT/'Benchmark/AndroidWorld/android_world/task_evals/information_retrieval/proto/tasks.textproto').read_text()
args.ir_tasks=re.findall(r'^  name: "([^"]+)"',proto,re.M)
original=[{e['task_id']:e for e in read(source/f'round_{i:02d}'/'summary.json')['episodes']} for i in range(4)]
cases={i:[t for t in all_tasks if original[i-1][t]['success'] and not original[i][t]['success']] for i in range(1,4)}
out.mkdir(parents=True,exist_ok=True)
write(out/'design.json',{'source':str(source),'cases':cases,'repeats':3,'seed':args.seed,'memory':'original round input; independent copy per task','temperature':args.temperature})
if not preflight(args,assignments(args,all_tasks)):raise SystemExit('preflight failed')
locks=ROOT/'Experiments/memory_baselines/artifacts/locks'; locks.mkdir(parents=True,exist_ok=True)
with (out/'run.lock').open('w') as lock,(locks/'androidworld_http.lock').open('w') as envlock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(envlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 encoder=MiniCPMEncoder(args.embedding_model,args.device)
 for i,tasks in cases.items():
  snapshot=read(source/f'round_{i:02d}'/'input_memory.json'); args.round_number=i
  for repeat in range(1,4):
   folder=out/f'round_{i:02d}'/f'repeat_{repeat:02d}'
   episodes,_=run_round(args,folder,tasks,budgets,snapshot,encoder)
   write(folder/'summary.json',{'round':i,'repeat':repeat,'episodes':episodes})
   rows=[]
   for j,ts in cases.items():
    for task in ts:
     scores=[]
     for k in range(1,4):
      p=out/f'round_{j:02d}'/f'repeat_{k:02d}'/'tasks'/task/'checkpoint.json'
      if p.exists():scores.append(read(p)['episode']['success'])
     rows.append({'round':j,'task':task,'successes':sum(scores),'completed':len(scores),'results':scores})
   write(out/'results.json',rows)
   print('REPEAT COMPLETE',i,repeat,flush=True)
