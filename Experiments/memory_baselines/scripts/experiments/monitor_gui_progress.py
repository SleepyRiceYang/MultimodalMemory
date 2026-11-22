#!/usr/bin/env python3
"""Record GUI checkpoints, changes, worker logs and model health every 15 minutes."""
import argparse,datetime,fcntl,json,re,shutil,subprocess,time,urllib.request
from collections import Counter
from pathlib import Path
root=Path('/root/VLM-Memory');artifacts=root/'Experiments/memory_baselines/artifacts'
statefile=artifacts/'logs/gui_progress_monitor_state.json'
ERROR=re.compile(r'traceback|\bexception\b|\berror\b|failed|timeout',re.I)


def action_error_category(error):
 if error.startswith('Invalid generation after two attempts:'):return 'invalid_generation_after_retries'
 if error.startswith('Invalid planner subtasks'):return 'invalid_planner_subtasks'
 if error.startswith('Invalid action fields'):return 'invalid_action_fields'
 if error.startswith('Invalid action:'):return 'invalid_action_schema'
 if '500 Server Error' in error:return 'http_500_action_or_environment'
 if 'Connection aborted' in error or 'RemoteDisconnected' in error:return 'environment_connection_aborted'
 return error[:160]

def status_counts(output):
 benchmark=None;result={}
 for line in output.splitlines():
  if line.startswith(('androidworld:','osworld:')):benchmark=line.split(':',1)[0]
  elif benchmark and line.startswith(('text |','visual |')):
   parts=[x.strip() for x in line.split('|')]
   if len(parts)==6:
    result[f'{benchmark}/{parts[0]}/round_{parts[1]}/{parts[2]}']={'completed':int(parts[3].split('/')[0]),'total':int(parts[3].split('/')[1]),'success':int(parts[4])}
 return result


def latest_eta(path, phase):
 try:lines=path.read_text(errors='replace').splitlines()
 except OSError:return None
 for line in reversed(lines):
  if 'PROGRESS ' not in line or phase not in line:continue
  match=re.search(r'\bETA=(\d+)s',line)
  if match:return {'seconds':int(match.group(1)),'line':line.strip()[:400]}
 return None


def latest_scheduler_eta(variant):
 path=artifacts/'logs'/'gui_evolution_pool.log';needle=f'POOL PROGRESS osworld_train_{variant}_r03_s0'
 try:lines=path.read_text(errors='replace').splitlines()
 except OSError:return None
 for line in reversed(lines):
  if needle not in line:continue
  match=re.search(r'\bETA=(\d+)s',line)
  if match:return {'seconds':int(match.group(1)),'line':line.strip()[:400]}
 return None


def remaining_estimate(counts):
 runtime=artifacts/'gui_magnet_evolution'/'osworld'/'magnet_evolution_v1_formal'/'pool_runtime'
 pending_build=[v for v in ('text','visual') if counts.get(f'osworld/{v}/round_3/construction',{}).get('completed',0)<50]
 pending_test=[v for v in ('text','visual') if counts.get(f'osworld/{v}/round_3/heldout',{}).get('completed',0)<50]
 build=[]
 for v in pending_build:
  eta=latest_scheduler_eta(v) or latest_eta(runtime/f'osworld_train_{v}_r03_s0.log',f'osworld/{v}/construction')
  if eta:build.append(eta['seconds'])
 historical=[]
 for v in pending_test:
  for p in runtime.glob(f'osworld_evaluate_{v}_r02_s*.log'):
   try:lines=p.read_text(errors='replace').splitlines()
   except OSError:continue
   for line in reversed(lines):
    match=re.search(rf'PROGRESS osworld/{v}/heldout 25/25 .*elapsed=(\d+)s',line)
    if match:historical.append(int(match.group(1)));break
 active_test=[]
 for v in pending_test:
  for p in runtime.glob(f'osworld_evaluate_{v}_r03_s*.log'):
   eta=latest_eta(p,f'osworld/{v}/heldout')
   if eta:active_test.append(eta['seconds'])
 test_eta=max(active_test) if active_test else (max(historical) if historical else None)
 build_eta=max(build) if build else (0 if not pending_build else None)
 total=build_eta+test_eta if build_eta is not None and test_eta is not None else None
 return {
  'construction_remaining_seconds':build_eta,
  'construction_workers_with_estimate':len(build),
  'heldout_remaining_seconds':test_eta,
  'heldout_estimate_basis':'active round-3 shard ETAs' if active_test else ('longest completed round-2 heldout shard; assumes round-3 shards run concurrently' if historical else 'unavailable'),
  'overall_remaining_seconds_excluding_final_validation':total,
  'pending_construction_conditions':pending_build,
  'pending_heldout_conditions':pending_test,
 }


def inspect():
 now=datetime.datetime.now(datetime.timezone.utc)
 out=subprocess.run(['bash',str(root/'Experiments/memory_baselines/scripts/experiments/resume_gui_evolution_pool.sh'),'--status'],capture_output=True,text=True,timeout=60)
 proc=subprocess.run(['pgrep','-af','[g]ui_baseline.pooled_evolution'],capture_output=True,text=True,timeout=10)
 try:
  with urllib.request.urlopen('http://127.0.0.1:8110/health',timeout=15) as r:health=json.load(r)
 except Exception as exc:health={'ready':False,'error':str(exc)}
 dependencies={}
 for name,url in [('embedding','http://127.0.0.1:8087/health'),('android_0','http://127.0.0.1:15003/docs'),('android_1','http://127.0.0.1:15004/docs')]:
  try:
   with urllib.request.urlopen(url,timeout=10) as response:
    dependencies[name]={'http_status':response.status,'ready':response.status==200}
    if name=='embedding':
     body=json.load(response);dependencies[name]['ready']=body.get('normalized') is True
  except Exception as exc:dependencies[name]={'ready':False,'error':str(exc)}
 disk=shutil.disk_usage(root);disk_info={'free_gib':round(disk.free/1024**3,2),'total_gib':round(disk.total/1024**3,2)}
 errors=[];logs=[];action_errors=Counter()
 for bench in ('androidworld','osworld'):
  base=artifacts/'gui_magnet_evolution'/bench/'magnet_evolution_v1_formal'
  for p in base.glob('*/**/result.json'):
   if any('_interrupted_' in part for part in p.parts):continue
   try:d=json.loads(p.read_text())
   except (OSError,ValueError):continue
   if d.get('status')=='error' or d.get('outcome')=='execution_error':
    errors.append({'benchmark':bench,'file':str(p.relative_to(root)),'status':d.get('status'),'failure_type':d.get('failure_type'),'error':d.get('error')})
  for p in base.glob('*/**/action_error.json'):
   if any('_interrupted_' in part for part in p.parts):continue
   try:
    error=json.loads(p.read_text()).get('error','unknown');phase='/'.join(p.relative_to(base).parts[:3])
    action_errors[bench+'/'+phase+': '+action_error_category(error)]+=1
   except (OSError,ValueError):continue
  recent=sorted((base/'pool_runtime').glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:2]
  for p in recent:
   with p.open('rb') as f:
    f.seek(0,2);size=f.tell();f.seek(max(0,size-131072))
    tail=f.read().decode('utf-8',errors='replace').replace('\r','\n').splitlines()[-250:]
   matches=[line.strip()[:1000] for line in tail if ERROR.search(line) and "'error': ''" not in line and '"error": ""' not in line][-8:]
   progress=[line.strip()[:400] for line in tail if re.search(r'\bPROGRESS\s+',line,re.I)][-2:]
   logs.append({'file':str(p.relative_to(root)),'age_seconds':round(time.time()-p.stat().st_mtime),'recent_error_lines':matches,'recent_phase_progress':progress})
 counts=status_counts(out.stdout)
 estimate=remaining_estimate(counts)
 try:previous=json.loads(statefile.read_text())
 except (OSError,ValueError):previous={}
 delta={k:v['completed']-previous.get(k,{}).get('completed',0) for k,v in counts.items() if v['completed']!=previous.get(k,{}).get('completed',0)} if previous else {}
 statefile.parent.mkdir(parents=True,exist_ok=True)
 tmp=statefile.with_suffix('.tmp');tmp.write_text(json.dumps(counts,ensure_ascii=False,indent=2));tmp.replace(statefile)
 path=root/'Docs/实验分析'/f'{now:%Y%m%d}-GUI实验进度监控.md'
 with path.open('a') as f:
  f.write(f'\n## {now:%Y-%m-%d %H:%M:%S} UTC\n\n')
  f.write('本次新增已提交任务：\n\n```json\n'+json.dumps(delta,ensure_ascii=False,indent=2)+'\n```\n\n')
  f.write('调度器及worker进程：\n\n```text\n'+(proc.stdout or '未检测到进程\n')+'```\n\n')
  f.write('模型池：\n\n```json\n'+json.dumps(health,ensure_ascii=False,indent=2)+'\n```\n\n')
  f.write('工作区磁盘：\n\n```json\n'+json.dumps(disk_info,ensure_ascii=False,indent=2)+'\n```\n\n')
  f.write('Embedding和Android服务：\n\n```json\n'+json.dumps(dependencies,ensure_ascii=False,indent=2)+'\n```\n\n')
  f.write('已提交任务进度：\n\n```text\n'+out.stdout+out.stderr+'```\n\n')
  f.write('构建与留出阶段剩余时间估算（秒）：\n\n```json\n'+json.dumps(estimate,ensure_ascii=False,indent=2)+'\n```\n\n')
  f.write('未归档任务异常：\n\n```json\n'+json.dumps(errors,ensure_ascii=False,indent=2)+'\n```\n\n')
  f.write('累计动作校验错误（不等同于任务级异常）：\n\n```json\n'+json.dumps(action_errors,ensure_ascii=False,indent=2)+'\n```\n\n')
  f.write('近期worker日志异常、阶段吞吐与ETA及更新时间：\n\n```json\n'+json.dumps(logs,ensure_ascii=False,indent=2)+'\n```\n')
 print(f'{now.isoformat()} wrote {path}; phase changes={delta}; task errors={len(errors)}',flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
 lock=(artifacts/'locks/gui_progress_monitor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  try:inspect()
  except Exception as exc:print(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} monitor error: {exc}',flush=True)
  if args.once:return
  time.sleep(900)

if __name__=='__main__':main()
