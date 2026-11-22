"""Idempotent local pool launcher; only owns explicitly recorded process IDs."""
import argparse,json,os,subprocess,time
from pathlib import Path
import psutil,requests
from .common import write,read
ROOT=Path(__file__).resolve().parents[4]
STATE=ROOT/'Experiments/memory_baselines/artifacts/gui_pool';STATE.mkdir(parents=True,exist_ok=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','status'],nargs='?',default='start');args=p.parse_args()
    jobs=[(f'gpu{gpu}',port,{'GPU_INDEX':str(gpu),'PORT':str(port),'MAX_NUM_SEQS':'4'},
           ['bash',str(ROOT/'Experiments/memory_baselines/scripts/serve/start_gui_parallel.sh')]) for gpu,port in [(3,8113),(4,8114),(6,8116)]]
    jobs.append(('router',8110,{'GUI_POOL_BACKENDS':'http://127.0.0.1:8113,http://127.0.0.1:8114,http://127.0.0.1:8116'},['/root/anaconda3/envs/vllm_env/bin/python','-m','uvicorn','gui_baseline.model_pool:app','--host','127.0.0.1','--port','8110']))
    s=requests.Session();s.trust_env=False
    for name,port,env,cmd in jobs:
        pidfile=STATE/(name+'.pid.json');owned=False
        if pidfile.exists():
            old=read(pidfile)
            try:
                proc=psutil.Process(old['pid']);owned=proc.create_time()==old['create_time'] and proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
            except psutil.Error:pass
        if args.action=='start' and not owned:
            import socket
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'Port {port} occupied by an unrecorded service')
            log=(STATE/(name+'.log')).open('a')
            proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL,start_new_session=True)
            write(pidfile,{'pid':proc.pid,'create_time':psutil.Process(proc.pid).create_time(),'command':cmd,'environment':env})
            print(f'START {name} pid={proc.pid} port={port}',flush=True)
    if args.action=='start':
        start=time.monotonic()
        while time.monotonic()-start<300:
            try:
                r=s.get('http://127.0.0.1:8110/health',timeout=10)
                if r.ok and r.json()['ready']:print(json.dumps(r.json(),indent=2));return
            except requests.RequestException:pass
            print('WAIT shared pool backend readiness',flush=True);time.sleep(5)
        raise TimeoutError('Pool startup timeout; inspect artifacts/gui_pool/*.log')
    r=s.get('http://127.0.0.1:8110/health',timeout=20);r.raise_for_status();print(json.dumps(r.json(),indent=2))

if __name__=='__main__':main()
