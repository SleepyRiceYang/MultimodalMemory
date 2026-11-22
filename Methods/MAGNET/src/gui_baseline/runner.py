"""Paired online procedural-only MAGNET adaptation; never updates memory in evaluation."""
from __future__ import annotations
import argparse,copy,fcntl,io,json,os,subprocess,time,traceback
from pathlib import Path
import requests
from PIL import Image
from .common import read,write,image_part,JsonModel
from .build import build_bank
from .memory import ProceduralBank
from .prompts import ANDROID_ACTOR

AW=['SystemBrightnessMin','SystemBluetoothTurnOff','ClockStopWatchRunning','MarkorCreateFolder','ContactsNewContactDraft']
OS=['bedcedc4-4d72-425e-ad62-21960b11fe0d','a4d98375-215b-4a4d-aee9-3d4370fccc41','ec4e3f68-9ea4-4c18-a5c9-69f89d1178b3','13584542-872b-42d8-b299-866967b5c3ef','5ced85fc-fa1a-4217-95fd-0fb530545ce2']

def memory_messages(messages,context):
    """Identical added text across arms; image blocks are the sole representation switch."""
    result=copy.deepcopy(messages)
    if not context['text'] and not context['images']:
        return result
    blocks=[{'type':'text','text':context['text']+'\nHistorical evidence order: workflow BEFORE, then AFTER.'}]
    blocks.extend(image_part(p) for p in context['images'])
    # A separate user turn preserves native tool/system prompts and current-screen ordering.
    result.insert(1,{'role':'user','content':blocks})
    result.insert(2,{'role':'assistant','content':'I will check applicability against the current task and screen.'})
    return result

def redact(value):
    if isinstance(value,dict):
        if value.get('type')=='image_url':return {'type':'image_url','image_url':{'url':'<image data omitted; see saved observations/evidence>'}}
        return {k:redact(v) for k,v in value.items()}
    if isinstance(value,list):return [redact(v) for v in value]
    return value

def normalize_android_action(value):
    if not isinstance(value,dict) or not isinstance(value.get('action_type'),str):
        raise ValueError('Invalid action: action_type must be a string; use the allowed action fields')
    action=copy.deepcopy(value);action.pop('reason',None)
    # Canonicalize only an unambiguous completion-field alias; do not repair actions.
    if action.get('action_type')=='status' and 'status' in action:
        if 'goal_status' in action and action['goal_status']!=action['status']:
            raise ValueError('Conflicting termination fields')
        action['goal_status']=action.pop('status')
    return action

class Android:
    def __init__(self,args):
        self.args=args;self.http=requests.Session();self.http.trust_env=False;self.model=JsonModel(args.model_url)
        self.api('GET','/suite/reinitialize',params={'seed':args.seed,'n_task_combinations':1})
    def api(self,method,path,**kw):
        r=self.http.request(method,getattr(self.args,'android_url','http://127.0.0.1:15000').rstrip('/')+path,timeout=180,**kw);r.raise_for_status();return r.json()
    def screenshot(self,path):
        r=subprocess.run(['docker','exec',getattr(self.args,'android_container','android_world_smoke'),'adb','exec-out','screencap','-p'],capture_output=True,check=True,timeout=30)
        im=Image.open(io.BytesIO(r.stdout)).convert('RGB');im.save(path);return im.size
    def ensure_network(self):
        """Restore the emulator's a11y gRPC link after Wi-Fi tasks turn it off."""
        adb=['docker','exec',getattr(self.args,'android_container','android_world_smoke'),'adb','shell']
        def run(*cmd):
            return subprocess.run(adb+list(cmd),capture_output=True,text=True,check=True,timeout=15).stdout
        if 'Wifi is connected to' in run('cmd','wifi','status'):
            return
        print('RECOVER AndroidWorld emulator Wi-Fi before task reset',flush=True)
        run('svc','wifi','enable')
        run('cmd','wifi','connect-network','AndroidWifi','open')
        for _ in range(30):
            if 'Wifi is connected to' in run('cmd','wifi','status'):
                return
            time.sleep(1)
        raise RuntimeError('AndroidWorld emulator Wi-Fi did not reconnect; a11y gRPC is unavailable')
    def setup(self,task):
        self.ensure_network()
        self.params={'task_type':task,'task_idx':getattr(self.args,'task_index',0)};self.api('POST','/reset',params={'go_home':True})
        self.api('POST','/task/initialize',params=self.params)
        return self.api('GET','/task/goal',params=self.params)['goal']
    def run(self,folder,goal,context,max_steps=None):
        limit=max_steps or self.args.max_steps
        history=[];done=False;predicted=False
        for step in range(limit):
            d=folder/f'step_{step:03d}';d.mkdir();w,h=self.screenshot(d/'screen.png')
            payload={'task':goal,'previous_actions':history[-10:],'step':step+1,'steps_remaining':limit-step}
            if context['text']:
                payload.update(procedural_memory=context['text'],image_order=context.get('image_order','CURRENT screenshot; historical BEFORE; historical AFTER'))
            images=[str(d/'screen.png')]+context['images']
            try:
                value=self.model.generate(d,ANDROID_ACTOR,payload,images,self.args.max_tokens,
                    **({'image_labels':context['image_labels']} if 'image_labels' in context else {}),
                    **({'response_schema':context['action_schema']} if 'action_schema' in context else {}))
                action=normalize_android_action(value);kind=action['action_type']
                fields={'click':{'x','y'},'long_press':{'x','y'},'double_tap':{'x','y'},'input_text':{'text'},'scroll':{'direction'},
                        'keyboard_enter':set(),'navigate_home':set(),'navigate_back':set(),'wait':set(),'open_app':{'app_name'},'answer':{'text'},'status':{'goal_status'}}
                if kind not in fields or set(action)!={'action_type',*fields[kind]}:raise ValueError('Invalid action fields')
                if kind in ('click','long_press','double_tap'):
                    for k,size in [('x',w),('y',h)]:
                        if type(action[k]) not in (int,float) or not 0<=action[k]<=1000:raise ValueError('Invalid coordinate')
                        action[k]=min(size-1,round(action[k]*size/1000))
                if kind=='scroll' and action['direction'] not in ('up','down','left','right'):raise ValueError('Invalid scroll')
                if kind=='status' and action['goal_status'] not in ('complete','infeasible'):raise ValueError('Invalid termination')
                write(d/'action.json',action)
                print(f'ACTION {folder.parent.name}/{folder.name} {step+1}: {action}',flush=True)
                if kind=='status':done=True;predicted=action['goal_status']=='complete';history.append({'action':value});break
                self.api('POST','/execute_action',json=action);history.append({'action':value});time.sleep(2)
            except (ValueError,requests.RequestException) as exc:
                write(d/'action_error.json',{'error':str(exc)});history.append({'error':str(exc)})
        self.screenshot(folder/'final.png')
        state=self.api('GET','/task/score',params=self.params)['score']
        return {'score':state if done else 0.,'task_state_score':state,'agent_done':done,'predicted_done':predicted,'steps':len(history)}
    def cleanup(self):self.api('POST','/task/tear_down',params=self.params)
    def close(self):self.http.close()

class Desktop:
    def __init__(self,args):
        self.args=args
        from desktop_env.desktop_env import DesktopEnv
        self.env=DesktopEnv(provider_name='docker',path_to_vm=args.vm_path,headless=True,action_space='pyautogui',
                require_a11y_tree=False,screen_size=(1920,1080),enable_proxy=False)
    def setup(self,task):
        examples=Path(self.args.workspace)/'Benchmark/OSWorld/evaluation_examples/examples'
        domain=getattr(self.args,'task_domains',{}).get(task)
        matches=[examples/domain/(task+'.json')] if domain else list(examples.glob('*/'+task+'.json'))
        if len(matches)!=1 or not matches[0].is_file():raise ValueError('OSWorld task config not uniquely found: '+task)
        config=read(matches[0])
        self.obs=self.env.reset(task_config=config);return config['instruction']
    def run(self,folder,goal,context):
        from mm_agents.qwen import QwenAgent
        agent=QwenAgent(model='qwen3-vl-8b-instruct',max_tokens=self.args.max_tokens,temperature=0,top_p=.9,
              action_space='pyautogui',observation_type='screenshot',coordinate_type='relative',history_n=3,image_max=3,
              fold_size=1,enable_thinking=False,base_url=self.args.model_url,api_key='EMPTY')
        agent.reset();native=agent.call_llm;counter=[0]
        def call(payload,model):
            payload=copy.deepcopy(payload);payload['messages']=memory_messages(payload['messages'],context)
            write(folder/f'request_{counter[0]:03d}.json',{'payload':redact(payload),'historical_images':context['images']})
            response=native(payload,model);write(folder/f'response_{counter[0]:03d}.json',{'response':response});counter[0]+=1;return response
        agent.call_llm=call
        done=False;steps=0
        for step in range(self.args.max_steps):
            (folder/f'screen_{step:03d}.png').write_bytes(self.obs['screenshot'])
            response,actions=agent.predict(goal,self.obs)
            if not actions:raise ValueError('Native agent returned no executable action')
            for action in actions:
                self.obs,reward,done,info=self.env.step(action,2)
                (folder/f'action_{steps:03d}_after.png').write_bytes(self.obs['screenshot'])
                write(folder/f'action_{steps:03d}.json',{'action':action,'done':done,'reward':reward,'info':info});steps+=1
                print(f'ACTION {folder.parent.name}/{folder.name} {steps}: {action}',flush=True)
                if done:break
            if done:break
        (folder/'final.png').write_bytes(self.obs['screenshot'])
        return {'score':float(self.env.evaluate()),'agent_done':bool(done),'steps':steps,'model_calls':counter[0]}
    def cleanup(self):pass
    def close(self):self.env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',choices=['androidworld','osworld'],required=True)
    p.add_argument('--stage',choices=['build','evaluate','all'],default='all');p.add_argument('--workspace',default='/root/VLM-Memory')
    p.add_argument('--run',default=None);p.add_argument('--seed',type=int,default=20261028)
    p.add_argument('--max-steps',type=int,default=30);p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--embedding-url',default='http://127.0.0.1:8087');p.add_argument('--model-url')
    p.add_argument('--vm-path',default='/data/yangshengjie/workspace/VLM-Memory/Benchmark/OSWorld/docker_vm_data/Ubuntu.qcow2')
    args=p.parse_args();args.run=args.run or ('paired_5_v2' if args.benchmark=='androidworld' else 'paired_5_v1');args.model_url=args.model_url or ('http://127.0.0.1:8104/v1' if args.benchmark=='androidworld' else 'http://127.0.0.1:8105/v1')
    base=Path(args.workspace)/'Experiments/memory_baselines/artifacts/gui_magnet_procedural'/args.benchmark;base.mkdir(parents=True,exist_ok=True)
    lock=(base/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage in ('build','all'):build_bank(args.workspace,args.benchmark,base/'bank',args.model_url,args.embedding_url)
    if args.stage=='build':return
    root=base/args.run;root.mkdir(exist_ok=True)
    config={**{k:v for k,v in vars(args).items() if k!='stage'},'tasks':AW if args.benchmark=='androidworld' else OS,'stationary_memory':False,'omniparser':False,
            'conditions':['text','visual'],'historical_image_budget':2,'retrieval':'frozen public goal cosine; identical selected IDs/text','image_type':'workflow BEFORE/AFTER keyframes',
            'android_system_prompt':ANDROID_ACTOR,'osworld_system_prompt':'native mm_agents.qwen internal tool prompt; memory injected as historical user turn'}
    if (root/'config.json').exists() and read(root/'config.json')!=config:raise ValueError('Run config changed; choose new --run')
    write(root/'config.json',config);bank=ProceduralBank(base/'bank',args.embedding_url)
    envlockpath=Path(args.workspace)/'Experiments/memory_baselines/artifacts/locks';envlockpath.mkdir(exist_ok=True)
    envlock=(envlockpath/(args.benchmark+'_http.lock' if args.benchmark=='androidworld' else 'osworld_vm.lock')).open('w');fcntl.flock(envlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    adapter=Android(args) if args.benchmark=='androidworld' else Desktop(args)
    results=[];started=time.monotonic();count=0
    try:
        for task in config['tasks']:
            for condition in ['text','visual']:
                folder=root/task/condition;folder.mkdir(parents=True,exist_ok=True)
                if (folder/'result.json').exists() and read(folder/'result.json')['status']=='completed':
                    results.append(read(folder/'result.json'));continue
                # Never mix a partially executed trajectory into resumed fresh environment state.
                if list(folder.iterdir()):
                    archive=folder.with_name(condition+'_interrupted_'+str(time.time_ns()));folder.rename(archive);folder.mkdir()
                start=time.monotonic();result={'benchmark':args.benchmark,'task_id':task,'condition':condition}
                try:
                    print(f'START {args.benchmark} {task} {condition}',flush=True)
                    goal=adapter.setup(task);ctxpath=root/task/'retrieval.json'
                    if ctxpath.exists():
                        context=read(ctxpath)
                        if context['goal']!=goal:raise ValueError('Paired task goal changed')
                    else:
                        context={**bank.select(args.benchmark,task,goal),'goal':goal};write(ctxpath,context)
                    view=bank.view(context,condition=='visual');write(folder/'memory.json',view)
                    if not view['selected']:raise ValueError('No procedural memory selected: invalid memory validation')
                    result.update(adapter.run(folder,goal,view));result['status']='completed';result['selected_ids']=[x['id'] for x in view['selected']]
                    result['historical_images']=len(view['images'])
                except Exception as exc:
                    result.update(status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());print(result['traceback'],flush=True)
                finally:
                    try:adapter.cleanup()
                    except Exception as exc:write(folder/'cleanup_error.json',{'error':str(exc)})
                result['seconds']=time.monotonic()-start;write(folder/'result.json',result);results.append(result);bank.verify();count+=1
                write(root/'summary.json',results);elapsed=time.monotonic()-started
                print(f'PROGRESS {args.benchmark} {len(results)}/10 score={result.get("score")} elapsed={elapsed:.0f}s ETA={elapsed/count*(10-len(results)):.0f}s',flush=True)
    finally:adapter.close()
    checks=[]
    for task in config['tasks']:
        if not (root/task/'text/memory.json').exists() or not (root/task/'visual/memory.json').exists():
            checks.append({'task':task,'available':False});continue
        a=read(root/task/'text/memory.json');b=read(root/task/'visual/memory.json')
        checks.append({'task':task,'same_text':a['text']==b['text'],'same_selection':a['selected']==b['selected'],
                       'text_images':len(a['images']),'visual_images':len(b['images'])})
    write(root/'pairing_checks.json',{'frozen_memory':bank.verify(),'stationary_memory':False,'checks':checks})
    write(root/'summary.json',results)

if __name__=='__main__':main()
