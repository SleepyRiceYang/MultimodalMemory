"""Construct bounded procedural units from recorded GUI state transitions."""
from __future__ import annotations
import json,shutil,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from .common import read,write,JsonModel,encode
from .memory import validate_node,parameterize,payload_text
from .prompts import BUILD,VERIFY


def source_episodes(workspace, benchmark, source_run=None, expected_task_ids=None):
    base=Path(workspace)/'Experiments/memory_baselines/artifacts'/benchmark
    episodes=[]
    if source_run:
        run=Path(workspace)/'Experiments/memory_baselines/artifacts/gui_magnet_procedural'/benchmark/source_run
        if not (run/'config.json').is_file():raise FileNotFoundError('Missing source run config: '+str(run))
        config=read(run/'config.json')
        if config.get('benchmark')!=benchmark or config.get('split_role')!='memory_construction':
            raise ValueError('Source run is not the frozen construction split')
        tasks=list(config['tasks'])
        if len(tasks)!=50 or len(set(tasks))!=50:raise ValueError('Expected 50 source tasks')
        if expected_task_ids is not None and set(tasks)!=set(expected_task_ids):raise ValueError('Construction manifest/source run mismatch')
        for task in tasks:
            folder=run/task/'no_memory';result_path=folder/'result.json'
            if not result_path.is_file() or read(result_path).get('status')!='completed':
                raise ValueError('No Memory source task incomplete: '+task)
            result=read(result_path);goal=read(run/task/'task.json')['goal'];steps=[]
            if benchmark=='androidworld':
                dirs=sorted(folder.glob('step_*'))
                for i,d in enumerate(dirs):
                    action_path=d/'action.json'
                    if not action_path.exists() or (d/'action_error.json').exists():continue
                    action=read(action_path)
                    if action['action_type']=='status':continue
                    after=dirs[i+1]/'screen.png' if i+1<len(dirs) else folder/'final.png'
                    if (d/'screen.png').exists() and after.exists():
                        steps.append({'step':i,'action':action,'before':str(d/'screen.png'),'after':str(after)})
            else:
                for action_path in sorted(folder.glob('action_???.json')):
                    i=int(action_path.stem.split('_')[1]);action=read(action_path)['action']
                    if action in ('DONE','FAIL','WAIT'):continue
                    before=folder/'screen_000.png' if i==0 else folder/f'action_{i-1:03d}_after.png'
                    after=folder/f'action_{i:03d}_after.png'
                    if not before.exists() or not after.exists():raise ValueError('Missing per-action OS observation: '+task)
                    steps.append({'step':i,'action':action,'before':str(before),'after':str(after)})
            episodes.append({'task_id':task,'goal':goal,'source_state_score':result.get('task_state_score',result['score']),
                             'source_agent_score':result['score'],'result_path':str(result_path),'steps':steps})
        return episodes
    if benchmark=='androidworld':
        root=base/'qwen3_vl_8b_androidworld_nomemory_5_20261026_png_v1'
        for p in sorted(root.glob('*/result.json')):
            result=read(p)
            if result['status']!='completed':continue
            steps=[]
            dirs=sorted(p.parent.glob('step_*'))
            for i,d in enumerate(dirs):
                if not (d/'action.json').exists() or (d/'action_error.json').exists():continue
                action=read(d/'action.json')
                if action['action_type']=='status':continue
                after=dirs[i+1]/'screen.png' if i+1<len(dirs) else p.parent/'final.png'
                if after.exists():steps.append({'step':i,'action':action,'before':str(d/'screen.png'),'after':str(after)})
            episodes.append({'task_id':p.parent.name,'goal':result['goal'],'source_state_score':result['task_state_score'],
                             'source_agent_score':result['score'],'result_path':str(p),'steps':steps})
    else:
        root=base/'qwen3_vl_8b_osworld_nomemory_5_20261026'
        tasks=read(root/'tasks.json')
        for domain,ids in tasks.items():
            for task_id in ids:
                folder=root/'pyautogui/screenshot/qwen3-vl-8b-instruct'/domain/task_id
                records=[json.loads(x) for x in (folder/'traj.jsonl').read_text().splitlines()]
                steps=[]
                # Native trajectories save post-action frames only; never invent an initial frame.
                for i in range(1,len(records)):
                    action=records[i].get('action','')
                    if action in ('DONE','FAIL','WAIT'):continue
                    before=folder/records[i-1]['screenshot_file'];after=folder/records[i]['screenshot_file']
                    if before.exists() and after.exists():steps.append({'step':i,'action':action,'before':str(before),'after':str(after)})
                goal=read(Path(workspace)/'Benchmark/OSWorld/evaluation_examples/examples'/domain/(task_id+'.json'))['instruction']
                score=float((folder/'result.txt').read_text())
                episodes.append({'task_id':task_id,'goal':goal,'source_state_score':score,'source_agent_score':score,
                                 'result_path':str(folder/'result.txt'),'steps':steps})
    return episodes


def build_bank(workspace, benchmark, root, model_url, embedding_url, source_run=None, expected_task_ids=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'bank.json').exists():
        old=read(root/'bank.json')
        if source_run is not None and old.get('source_run')!=source_run:raise ValueError('Existing bank has a different source run')
        if expected_task_ids is not None and set(old['source_task_ids'])!=set(expected_task_ids):
            raise ValueError('Existing bank source IDs differ from split')
        return old
    model=JsonModel(model_url);episodes=source_episodes(workspace,benchmark,source_run,expected_task_ids)
    write(root/'source_episodes.json',episodes)
    nodes=[];audit=[];started=time.monotonic()
    for number,episode in enumerate(episodes,1):
        elapsed=time.monotonic()-started
        eta=(elapsed/(number-1))*(len(episodes)-number+1) if number>1 else None
        finish=(datetime.now(timezone.utc)+timedelta(seconds=eta)).strftime('%Y-%m-%d %H:%M UTC') if eta is not None else 'pending'
        print(f'BUILD {benchmark} {number}/{len(episodes)} source={episode["task_id"]} elapsed={elapsed:.0f}s ETA={eta:.0f}s finish~{finish}' if eta is not None else
              f'BUILD {benchmark} {number}/{len(episodes)} source={episode["task_id"]} elapsed={elapsed:.0f}s ETA=pending',flush=True)
        # Bound noisy duplicate tails before asking the model; retain links to actual source IDs.
        steps=[];last=None;repeats=0
        for step in episode['steps']:
            key=json.dumps(step['action'],sort_keys=True)
            repeats=repeats+1 if key==last else 1;last=key
            if repeats>2:break
            steps.append(step)
            if len(steps)>=14:break
        if len(steps)<2:continue
        source={**episode,'steps':steps}
        snapshots=[('BEFORE action '+str(s['step']),s['before']) for s in steps]
        snapshots.append(('AFTER action '+str(steps[-1]['step']),steps[-1]['after']))
        payload={'source_goal':source['goal'],'source_terminal_state_score':source['source_state_score'],
                 'source_agent_score':source['source_agent_score'],'actions':[{'step':s['step'],'action':s['action']} for s in steps],
                 'screenshot_order':[key for key,_ in snapshots]}
        name=episode['task_id']
        try:result=model.generate(root/'calls'/('extract_v2_'+name),BUILD,payload,[p for _,p in snapshots],3072)
        except Exception as exc:
            audit.append({'source':name,'error':str(exc)});continue
        for j,candidate in enumerate(result.get('workflows',[])[:2]):
            try:
                ids,evidence=validate_node(candidate,source)
                by_id={s['step']:s for s in steps}
                selected=[by_id[i] for i in ids]
                verdict=model.generate(root/'calls'/f'verify_v2_{name}_{j}',VERIFY,
                    {'candidate':candidate,'source_goal':source['goal'],'source_terminal_state_score':source['source_state_score'],
                     'actual_actions':[{'step':s['step'],'action':s['action']} for s in selected],
                     'screenshot_order':['BEFORE action '+str(s['step']) for s in selected]+['AFTER action '+str(selected[-1]['step'])]},
                    [s['before'] for s in selected]+[evidence[1]],1536)
                if verdict.get('supported') is not True or not verdict.get('before_evidence') or not verdict.get('after_evidence'):
                    raise ValueError('Visual transition verification rejected: '+str(verdict))
                node=parameterize(candidate,source);node['id']=f'{benchmark}/magnet_procedural/{len(nodes)}'
                node['source_task_id']=name;node['evidence']=[]
                for k,path in enumerate(evidence):
                    target=Path('evidence')/f'workflow_{len(nodes)}_{k}.png';(root/target).parent.mkdir(exist_ok=True)
                    shutil.copy2(path,root/target);node['evidence'].append(str(target))
                node['verification']=verdict
                node['source_terminal_state_score']=source['source_state_score'];node['source_agent_score']=source['source_agent_score']
                node['verification_scope']='visual_subtask_transition; not necessarily full source task success'
                nodes.append(node);audit.append({'source':name,'candidate':j,'accepted':node['id']})
            except Exception as exc:audit.append({'source':name,'candidate':j,'rejected':str(exc)})
    write(root/'construction_audit.json',audit)
    if not nodes:raise ValueError('No verified procedural units; cannot claim memory baseline validation')
    texts=['Topic: '+n['title']+'\nScope: '+n['app']+'; '+n['precondition']+'\nQuestion Anchor: '+n['goal'] for n in nodes]
    vectors=np.asarray(encode(embedding_url,texts),dtype=np.float32)
    np.save(root/'vectors.npy',vectors)
    write(root/'index.json',{'texts':texts,'embedding_url':embedding_url,'encoder':'BGE-large-en-v1.5','dimension':vectors.shape[1]})
    bank={'benchmark':benchmark,'method':'MAGNET-GUI-procedural-adapted','stationary_memory':False,
          'visual_evidence':'workflow_before_after_keyframes','omniparser_used':False,
          'source_run':source_run,'source_task_ids':[e['task_id'] for e in episodes],'nodes':nodes}
    write(root/'bank.json',bank)
    return bank
