"""Frozen-source CapCoMem motivation pilot. Historical artifacts are read-only inputs.

Run with --stage prepare, assets, select, evaluate, or report. See the companion README.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import random
import re
import shutil
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ..config import ROOT, WORKSPACE, write_json
from ..models.local import LocalModel, image_hash, parse_object
from .pilot import execute, read
from .run_100_100_nomemory import heartbeat
from ..memory.retrieval import post

SEED = 20261104
ART = ROOT / 'artifacts'
MP = ART / 'mpdocvqa/qwen3_vl_8b_mpdocvqa_8192_32768_v1_nomemory_four_memory_v2_embedding_v1'
GUI = ART / 'gui_magnet_procedural/androidworld'
GB = GUI / 'banks/formal_50_bank_v1'
FORMS = dict(M0='No historical memory', M1='Fixed experience core text',
             M2='Core + full source frame + short caption', M3='Core + source ROI + short caption',
             M4='Core + before/after of one historical action + short caption')
NOTICE = 'HISTORICAL EXPERIENCE: these images and values belong to a previous task, not the current task. Check applicability.\n'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def freeze(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Frozen specification changed: {path}; use a new output directory')
    write_json(path, value)


def category(before, after):
    return ('unchanged_correct' if before else 'unchanged_wrong') if before == after else ('improved' if after else 'harmed')


def sample_rows(rows, count, seed=SEED):
    """Disjoint strata; shortages are filled without relabelling or replacement."""
    if len({r['id'] for r in rows}) != len(rows) or count > len(rows):
        raise ValueError('Sampling requires unique IDs and enough candidates')
    rng = random.Random(seed)
    groups = {k: sorted([dict(r) for r in rows if r['category'] == k], key=lambda x:x['id'])
              for k in ('improved', 'harmed', 'unchanged_correct', 'unchanged_wrong')}
    for g in groups.values(): rng.shuffle(g)
    chosen = groups['improved'][:20] + groups['harmed'][:20]
    unchanged = groups['unchanged_correct'][:10] + groups['unchanged_wrong'][:10]
    extra = [r for k in ('unchanged_correct','unchanged_wrong') for r in groups[k] if r not in unchanged]
    rng.shuffle(extra)
    chosen += (unchanged + extra)[:20]
    chosen = chosen[:count]
    selected = {r['id'] for r in chosen}
    rest = [dict(r) for r in rows if r['id'] not in selected]; rng.shuffle(rest)
    for r in chosen: r['supplement'] = False
    for r in rest[:count-len(chosen)]: r['supplement'] = True; chosen.append(r)
    return chosen


def prepare(root):
    mp = read(MP/'dataset_manifest.json')
    gui = read(ROOT/'splits/gui_50_50_seed20261026/androidworld.json')
    # Enforce document isolation rather than silently claiming it.
    overlap = {t['metadata']['doc_id'] for t in mp['memory']} & {t['metadata']['doc_id'] for t in mp['test']}
    if overlap: raise ValueError(f'MP source/test document overlap: {sorted(overlap)}')
    scores = list(csv.DictReader((WORKSPACE/'Docs/实验分析/20261026-01-五组Baseline三Benchmark逐题结果.csv').open()))
    by = {(r['method'],r['task_id']): r for r in scores if r['benchmark']=='mpdocvqa'}
    rows=[]
    for task in mp['test']:
        a,b=[by[m,task['id']] for m in ('no_memory','ksi_visual')]
        rows.append(dict(id=task['id'],category=category(a['success']=='True',b['success']=='True'),
                         old_no_memory=float(a['score']),old_memory=float(b['score']),reference='ksi_visual',
                         reference_run=str(MP),original_split='test'))
    gr=[]
    for r in csv.DictReader((WORKSPACE/'Docs/实验分析/20261028-01-AndroidWorld-50-50三条件配对统计分析.csv').open(encoding='utf-8-sig')):
        gr.append(dict(id=r['task_id'],category=category(float(r['no_memory_score'])==1,float(r['text_score'])==1),
                       old_no_memory=float(r['no_memory_score']),old_memory=float(r['text_score']),
                       reference='magnet_procedural_text',reference_run=str(GUI/'formal_50_test_v1'),
                       original_split='test',supplement=False))
    if {r['id'] for r in gr} != set(gui['test']): raise ValueError('GUI historical coverage mismatch')
    rng=random.Random(SEED)
    selection=dict(seed=SEED,exploratory=True,allow_self_memory=True,mp_document_overlap=sorted(overlap),
        mpdocvqa=dict(construction=rng.sample(sorted(t['id'] for t in mp['memory']),20),test=sample_rows(rows,60)),
        androidworld=dict(construction=rng.sample(sorted(gui['memory_construction']),20),test=gr))
    freeze(root/'selection.json',selection)
    nodes=[]
    for n in read(MP/'bank.json')['methods']['ksi_visual']:
        if not n['images']: continue
        im=n['images'][0]
        task=next(t for t in mp['memory'] if t['id']==im['source_id'])
        nodes.append(dict(id=n['id'],benchmark='mpdocvqa',source_ids=n['source_ids'],core=n['text'],
            frame=task['images'][im['source_image_index']],roi=str(MP/im['path']),box=im['box'],
            index_text=n['index_text'],source_frame_index=im['source_image_index'],observation=im.get('observation','')))
    episodes={e['task_id']:e for e in read(GB/'source_episodes.json')}
    from gui_baseline.memory import payload_text
    for n in read(GB/'bank.json')['nodes']:
        # Existing workflow endpoints may span several actions. Use one cited action instead.
        step=next(s for s in episodes[n['source_task_id']]['steps'] if s['step']==n['steps'][0]['source_step'])
        nodes.append(dict(id=n['id'],benchmark='androidworld',source_ids=[n['source_task_id']],core=payload_text(n),
            frame=step['before'],after=step['after'],source_action=step['action'],source_step=step['step'],
            index_text=n['title'],observation=n['steps'][0]['instruction']))
    allowed={'mpdocvqa':{t['id'] for t in mp['memory']},'androidworld':set(gui['memory_construction'])}
    if any(not set(n['source_ids'])<=allowed[n['benchmark']] for n in nodes):raise ValueError('Non-construction memory source')
    source_hashes={str(p):image_hash(p) for p in (MP/'bank.json',MP/'dataset_manifest.json',GB/'bank.json',GB/'source_episodes.json')}
    freeze(root/'sources.json',dict(nodes=nodes,source_hashes=source_hashes))
    print('PREPARED', {b:Counter(r['category'] for r in selection[b]['test']) for b in ('mpdocvqa','androidworld')},flush=True)


def model_config(url):
    c=read(MP/'config.snapshot.json')['model']
    return {**c,'base_url':url,'timeout':180,'check_context_budget':True,'max_input_tokens':56000}


def retrieve(root, benchmark, task_id, query, embedding):
    path=root/'retrieval'/benchmark/f'{task_id}.json'
    spec=dict(benchmark=benchmark,task_id=task_id,query=query,top_k=1,source_hash=image_hash(root/'sources.json'))
    if path.exists():
        row=read(path)
        if row['spec']!=spec:raise ValueError('Frozen retrieval query changed')
        return row
    nodes=[n for n in read(root/'sources.json')['nodes'] if n['benchmark']==benchmark]
    index=root/'indices'/f'{benchmark}.json'
    if index.exists():
        saved=read(index)
        if saved['ids']!=[n['id'] for n in nodes]:raise ValueError('Index IDs changed')
    else:
        texts=post(embedding+'/tokens',dict(texts=[n['index_text'] for n in nodes],limit=480))
        data=post(embedding+'/encode',dict(texts=[t['text'] for t in texts]))
        saved=dict(ids=[n['id'] for n in nodes],vectors=data['embeddings'],encoder=post_health(embedding),texts=texts)
        freeze(index,saved)
    q=post(embedding+'/encode',dict(texts=[query],query=True))['embeddings'][0]
    scores=np.asarray(saved['vectors'])@np.asarray(q)
    rank=sorted(range(len(nodes)),key=lambda i:(-float(scores[i]),nodes[i]['id']))
    node=nodes[rank[0]]
    row=dict(spec=spec,selected=[node['id']],cosine=float(scores[rank[0]]),self_hit=task_id in node['source_ids'])
    freeze(path,row)
    return row


def post_health(url):
    import urllib.request
    with urllib.request.urlopen(url.rstrip('/')+'/health',timeout=15) as r:return json.load(r)


def asset(root, node, url):
    folder=root/'assets'/digest(node['id'])[:16]
    snapshot=folder/'asset.json'
    if snapshot.exists():
        if read(snapshot).get('protocol')=='source_separated_v3':return read(snapshot)
        snapshot.rename(folder/('asset_previous_'+str(time.time_ns())+'.json'))
    folder.mkdir(parents=True,exist_ok=True)
    model=LocalModel(model_config(url))
    frame=Path(node['frame'])
    images=[('Historical source BEFORE or document page',frame)]
    # Single-frame captions never see the later frame or action's literal output.
    schema={'type':'object','properties':{'box':{'type':'array','items':{'type':'integer','minimum':0,'maximum':1000},'minItems':4,'maxItems':4},
        'full_caption':{'type':'string','maxLength':320},'roi_caption':{'type':'string','maxLength':320},
        'pair_caption':{'type':'string','maxLength':320}},'required':['box','full_caption','roi_caption','pair_caption'],'additionalProperties':False}
    # MP uses the established crop; GUI obtains a box from source evidence only.
    prompt=json.dumps(dict(core=node['core'],source_observation=node['observation'],existing_box=node.get('box'),
        instructions='Return integer coordinates 0-1000 [left,top,right,bottom] ROI on FIRST image, focused on the cited evidence/action target. Each caption: at most TWO concise sentences. Full caption describes first frame and location; ROI caption ONLY the cropped region; pair caption visible change after this one action, or empty for documents. Do not invent successful transitions. No current task is supplied.'),ensure_ascii=False)
    if (folder/'caption_call.json').exists():
        (folder/'caption_call.json').rename(folder/('caption_call_previous_'+str(time.time_ns())+'.json'))
    response=model.generate('Describe only the supplied historical visual evidence.',prompt,images,max_tokens=1024,json_schema=schema,progress_path=folder/'caption.partial.json')
    write_json(folder/'caption_call.json',response)
    if response['finish_reason']=='length':raise ValueError('Truncated evidence description')
    data=parse_object(response['text'])
    box=node['box'] if 'box' in node else [v/1000 for v in data['box']]
    if len(box)!=4 or not (0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1):raise ValueError(f'Invalid source ROI: {box}')
    if node.get('roi'):
        shutil.copyfile(node['roi'],folder/'roi.png')
    else:
        with Image.open(frame) as im:
            w,h=im.size;pixels=(int(box[0]*w),int(box[1]*h),int(box[2]*w),int(box[3]*h))
            if pixels[2]<=pixels[0] or pixels[3]<=pixels[1]:raise ValueError('Empty ROI')
            im.crop(pixels).save(folder/'roi.png')
    caption_schema={'type':'object','properties':{'caption':{'type':'string','maxLength':320}},'required':['caption'],'additionalProperties':False}
    roi_call=model.generate('Describe only the supplied historical crop, in at most two short sentences.',
        'Describe the visible layout and relevant field types in this crop. Do not transcribe numeric amounts, identifiers, or answer strings. Do not infer earlier or later states.',[('Historical ROI',folder/'roi.png')],max_tokens=256,json_schema=caption_schema)
    write_json(folder/'roi_caption_call.json',roi_call)
    if roi_call['finish_reason']=='length':raise ValueError('Truncated ROI caption')
    data['roi_caption']=parse_object(roi_call['text'])['caption']
    data['pair_caption']=''
    if node.get('after'):
        pair_call=model.generate('Describe only the visible change between these historical frames, in at most two short sentences.',
            'Image 1 is BEFORE; image 2 is AFTER one action. Do not infer overall task success.',
            [('HISTORICAL BEFORE',frame),('HISTORICAL AFTER',Path(node['after']))],max_tokens=256,json_schema=caption_schema)
        write_json(folder/'pair_caption_call.json',pair_call)
        if pair_call['finish_reason']=='length':raise ValueError('Truncated transition caption')
        data['pair_caption']=parse_object(pair_call['text'])['caption']
    result=dict(protocol='source_separated_v3',node_id=node['id'],core_sha256=digest(node['core']),frame=str(frame),roi=str(folder/'roi.png'),
                after=node.get('after'),box=box,captions={k:' '.join(re.split(r'(?<=[.!?])\s+(?=[A-Z])',data[k])[:2]) for k in ('full_caption','roi_caption','pair_caption')})
    result['asset_hashes']={str(p):image_hash(Path(p)) for p in [result['frame'],result['roi'],result['after']] if p}
    freeze(snapshot,result)
    return result


def context(root, benchmark, task_id, form):
    if form not in FORMS or (benchmark=='mpdocvqa' and form=='M4'):raise ValueError('Unavailable memory form')
    if form=='M0':return dict(text='',images=[],selected=[],truncated=False)
    r=read(root/'retrieval'/benchmark/f'{task_id}.json')
    n=next(n for n in read(root/'sources.json')['nodes'] if n['id']==r['selected'][0])
    a=read(root/'assets'/digest(n['id'])[:16]/'asset.json')
    text=NOTICE+n['core'];images=[]
    if form=='M2':text+='\nHistorical full-frame evidence: '+a['captions']['full_caption'];images=[('HISTORICAL full frame',Path(a['frame']))]
    if form=='M3':text+='\nHistorical ROI evidence: '+a['captions']['roi_caption'];images=[('HISTORICAL cropped ROI',Path(a['roi']))]
    if form=='M4':
        if not a['after']:raise ValueError('Temporal memory unavailable')
        text+='\nHistorical single-action transition: '+a['captions']['pair_caption']
        images=[('HISTORICAL BEFORE',Path(a['frame'])),('HISTORICAL AFTER',Path(a['after']))]
    return dict(text=text,images=images,selected=[dict(id=n['id'],source_ids=n['source_ids'])],truncated=False,
                core_sha256=a['core_sha256'],self_hit=r['self_hit'],form=form)


def tasks(root,b,limit):
    selection=read(root/'selection.json')[b]
    mp=read(MP/'dataset_manifest.json') if b=='mpdocvqa' else None
    output=[]
    for role,ids in [('construction',selection['construction']),('test',[r['id'] for r in selection['test']])]:
        # Fixed diagnostic subset, favour fewer pages but never use outcomes to choose smoke tasks.
        if mp:ids=sorted(ids,key=lambda key:(len(next(t for split in ('memory','test') for t in mp[split] if t['id']==key)['images']),key))
        if not mp and role=='construction':
            covered={sid for n in read(root/'sources.json')['nodes'] if n['benchmark']==b for sid in n['source_ids']}
            ids=sorted(ids,key=lambda key:(key not in covered,key))
        for key in ids[:limit] if limit else ids:
            if mp:task=next(t for split in ('memory','test') for t in mp[split] if t['id']==key)
            else:task=dict(id=key,prompt=read(GUI/('formal_50_construction_v1' if role=='construction' else 'formal_50_test_v1')/key/'task.json')['goal'])
            output.append((role,task))
    return output


def choose(root,b,role,task,url):
    path=root/'choices'/b/role/f'{task["id"]}.json'
    if path.exists():return
    if any((root/'runs'/b/role/task['id']).glob('M*/result.json')):
        raise ValueError('Cannot submit a new selector after task outcomes exist')
    forms={k:v for k,v in FORMS.items() if b=='androidworld' or k!='M4'}
    model=LocalModel(model_config(url))
    cap_schema={'type':'object','properties':{'capabilities':{'type':'array','items':{'type':'string','enum':(['visual_localization','planning','verification'] if b=='androidworld' else ['page_localization','extraction','association'])},'minItems':1,'maxItems':3}},'required':['capabilities'],'additionalProperties':False}
    cap=model.generate('Identify required capabilities using only this task description.',task['prompt'],[],max_tokens=1024,json_schema=cap_schema)
    write_json(path.parent/(path.stem+'_capability_call.json'),cap)
    if cap['finish_reason']=='length':raise ValueError('Truncated capability selection')
    caps=parse_object(cap['text'])
    caps['capabilities']=list(dict.fromkeys(caps['capabilities']))
    schema={'type':'object','properties':{'form':{'type':'string','enum':list(forms)}},'required':['form'],'additionalProperties':False}
    result=dict(task_id=task['id'],capabilities=caps,capability_call=cap,choices={})
    for strategy in ('task','capability'):
        payload=dict(task=task['prompt'],forms=forms)
        if strategy=='capability':payload.update(caps)
        call=model.generate('Choose one memory form before task execution. No outcomes are available.',json.dumps(payload),[],max_tokens=1024,json_schema=schema)
        write_json(path.parent/(path.stem+'_'+strategy+'_call.json'),call)
        if call['finish_reason']=='length':raise ValueError('Truncated form selection')
        result['choices'][strategy]=dict(form=parse_object(call['text'])['form'],call=call)
    result['random_choices']={str(seed):random.Random(f'{seed}/{b}/{task["id"]}').choice(list(forms)) for seed in range(10)}
    result['selector_budget']=dict(model=model.config['name'],max_tokens=1024,capability_generation_counted=True)
    result['submitted_at']=time.time()
    freeze(path,result)


def run_mp(root,role,task,form,url):
    folder=root/'runs/mpdocvqa'/role/task['id']/form
    ctx=context(root,'mpdocvqa',task['id'],form)
    if (folder/'result.json').exists() and read(folder/'result.json')['status']=='completed':
        expected={**ctx,'images':[(label,str(path)) for label,path in ctx['images']]}
        if read(folder/'input.json')!=task or digest(read(folder/'memory_trace.json'))!=digest(expected):raise ValueError('Completed MP input/context changed')
        return
    config=read(MP/'config.snapshot.json');config['model']=model_config(url)
    # Scoring references are copied separately, never given to retrieval or selectors.
    target=root/'evaluation_targets'/f'{task["id"]}.json'
    source=MP/'evaluation_targets'/target.name
    if not source.exists():source=Path(str(MP).replace('_four_memory_v2_embedding_v1',''))/'evaluation_targets'/target.name
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    result=execute(task,ctx,folder,config,root)
    result.update(role=role,form=form,self_hit=ctx.get('self_hit',False),cost=measure_cost(folder,ctx));write_json(folder/'result.json',result)


def android_action_schema():
    number={'type':'number','minimum':0,'maximum':1000}
    string={'type':'string'}
    actions={'click':dict(x=number,y=number),'long_press':dict(x=number,y=number),
             'double_tap':dict(x=number,y=number),'input_text':dict(text=string),
             'scroll':dict(direction={'type':'string','enum':['up','down','left','right']}),
             'keyboard_enter':{},'navigate_home':{},'navigate_back':{},'wait':{},
             'open_app':dict(app_name=string),'answer':dict(text=string),
             'status':dict(goal_status={'type':'string','enum':['complete','infeasible']})}
    options=[]
    for action,fields in actions.items():
        properties=dict(reason=string,action_type={'type':'string','enum':[action]},**fields)
        options.append(dict(type='object',properties=properties,required=list(properties),additionalProperties=False))
    return {'anyOf':options}


def run_android(root,items,url,env_url,container,max_steps):
    from gui_baseline.runner import Android
    args=SimpleNamespace(model_url=url,android_url=env_url,android_container=container,seed=20261026,task_index=0,max_steps=max_steps,max_tokens=2048)
    lockroot=ART/'locks';lockroot.mkdir(exist_ok=True)
    with (lockroot/(container+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        adapter=Android(args)
        try:
            for role,task in items:
                for form in FORMS:
                    folder=root/'runs/androidworld'/role/task['id']/form
                    ctx=context(root,'androidworld',task['id'],form)
                    ctx['action_schema']=android_action_schema()
                    ctx['image_labels']=['CURRENT screenshot']+[label for label,_ in ctx['images']]
                    ctx['image_order']='; '.join(ctx['image_labels'])
                    ctx['images']=[str(p) for _,p in ctx['images']]
                    if (folder/'result.json').exists() and read(folder/'result.json').get('status')=='completed':
                        if read(folder/'memory_trace.json')!=ctx:raise ValueError('Completed GUI context changed')
                        continue
                    if folder.exists():folder.rename(folder.with_name(form+'_interrupted_'+str(time.time_ns())))
                    folder.mkdir(parents=True)
                    write_json(folder/'memory_trace.json',ctx)
                    started=time.monotonic();initialized=False
                    try:
                        # The AndroidWorld service advances task-goal variants on repeated
                        # initialization even with task_idx=0. Re-seed before each form so
                        # every condition receives the same frozen historical goal.
                        adapter.api('GET','/suite/reinitialize',params={'seed':args.seed,'n_task_combinations':1})
                        goal=adapter.setup(task['id']);initialized=True
                        if goal!=task['prompt']:raise ValueError('Current GUI goal differs from historical frozen goal')
                        result=adapter.run(folder,goal,ctx,max_steps=max_steps)
                        result.update(status='completed',task_id=task['id'],role=role,form=form,seconds=time.monotonic()-started,
                                      self_hit=ctx.get('self_hit',False),step_limit=max_steps)
                        result['cost']=measure_cost(folder,ctx)
                        write_json(folder/'result.json',result)
                    finally:
                        if initialized:adapter.cleanup()
                    print('DONE androidworld',role,task['id'],form,result['score'],flush=True)
        finally:adapter.close()


def measure_cost(folder,ctx):
    calls=list((folder/'calls').glob('*.json'))+list(folder.glob('step_*/response_*.json'))
    calls=[p for p in calls if not p.name.endswith('.partial.json')]
    usage=Counter();seconds=0.;image_inputs=[]
    for p in calls:
        c=read(p)
        for k,v in (c.get('usage') or {}).items():
            if isinstance(v,(int,float)):usage[k]+=v
        seconds+=c.get('seconds',0)
        image_inputs.extend(c.get('images',[]))
    for request_path in folder.glob('step_*/request.json'):
        request=read(request_path)
        labels=request.get('image_labels',[])
        for i,name in enumerate(request['images']):
            with Image.open(name) as im:
                original=list(im.size);im.thumbnail((1280,1280));sent=list(im.size)
            image_inputs.append(dict(path=name,original_size=original,sent_size=sent,label=labels[i] if labels else None))
    historical=[]
    for item in ctx['images']:
        path=Path(item[1]) if isinstance(item,tuple) else Path(item)
        with Image.open(path) as im: historical.append(dict(path=str(path),size=list(im.size)))
    return dict(usage=dict(usage),recorded_model_seconds=seconds,model_calls=len(calls),
                historical_images=historical,memory_chars=len(ctx['text']),model_image_inputs=image_inputs)


def audit(root):
    sources=read(root/'sources.json')
    for p,h in sources['source_hashes'].items():
        if image_hash(Path(p))!=h:raise ValueError('Historical source changed: '+p)
    for p in (root/'assets').glob('*/asset.json'):
        a=read(p)
        n=next(n for n in sources['nodes'] if n['id']==a['node_id'])
        if a['core_sha256']!=digest(n['core']):raise ValueError('Core text changed')
        for name,h in a['asset_hashes'].items():
            if image_hash(Path(name))!=h:raise ValueError('Visual asset changed: '+name)
    return True


def freeze_fixed(root,b,limit):
    forms=list(FORMS) if b=='androidworld' else list(FORMS)[:4]
    ids=[t['id'] for role,t in tasks(root,b,limit) if role=='construction']
    scored={}
    for f in forms:
        results=[read(root/'runs'/b/'construction'/key/f/'result.json') for key in ids]
        if any(r.get('status')!='completed' for r in results):raise ValueError('Incomplete fixed-form calibration')
        scored[f]=dict(mean=sum(r['score'] for r in results)/len(results),
                       tokens=sum(r.get('cost',{}).get('usage',{}).get('total_tokens',0) for r in results))
    best=max(forms,key=lambda f:(scored[f]['mean'],-scored[f]['tokens'],-forms.index(f)))
    path=root/'fixed_forms'/f'{b}.json'
    if path.exists():
        old=read(path)
        if old['scores']!=scored or old['form']!=best or old['construction_ids']!=ids:raise ValueError('Frozen best fixed form changed')
        return old
    spec=dict(form=best,scores=scored,construction_ids=ids,submitted_at=time.time(),
              retrospective_pilot_diagnostic=any((root/'runs'/b/'test').glob('*/M*/result.json')))
    freeze(path,spec)
    return spec


def report(root):
    rows=[]
    for p in (root/'runs').glob('*/*/*/M*/result.json'):
        r=read(p)
        if r.get('status')!='completed':continue
        b,role,key,form=p.relative_to(root/'runs').parts[:4]
        if form not in FORMS:continue
        rows.append(dict(benchmark=b,role=role,id=key,form=form,score=r['score'],self_hit=r.get('self_hit'),seconds=r['seconds'],cost=r.get('cost',{}),metrics={k:r.get('evaluation',{}).get(k) for k in ('anls','exact_match','answer_page_accuracy')} if b=='mpdocvqa' else dict(task_state_score=r.get('task_state_score'),agent_done=r.get('agent_done'),steps=r.get('steps'),step_limit=r.get('step_limit'))))
    groups={}
    protocol=[]
    for b,role,key in sorted({(r['benchmark'],r['role'],r['id']) for r in rows}):
        expected_forms=list(FORMS) if b=='androidworld' else list(FORMS)[:4]
        traces=[]
        for form in expected_forms:
            path=root/'runs'/b/role/key/form/'memory_trace.json'
            if path.exists():traces.append((form,read(path)))
        memory=[c for f,c in traces if f!='M0']
        core_matches=True
        source_nodes=read(root/'sources.json')['nodes']
        for c in memory:
            node=next(n for n in source_nodes if n['id']==c['selected'][0]['id'])
            core_matches &= c['text'].startswith(NOTICE+node['core']) and c['core_sha256']==digest(node['core'])
        paired=len(memory)==len(expected_forms)-1 and len({digest(c['selected']) for c in memory})==1 and len({c['core_sha256'] for c in memory})==1
        m0=next((c for f,c in traces if f=='M0'),None)
        protocol.append(dict(benchmark=b,role=role,id=key,paired_core_and_ids=paired,actual_core_matches_source=core_matches,no_memory_empty=bool(m0 is not None and not m0['text'] and not m0['images'] and not m0['selected'])))
    for r in rows:groups.setdefault('/'.join([r['benchmark'],r['role'],r['form']]),[]).append(r['score'])
    strategies=[]
    for b in ('mpdocvqa','androidworld'):
        construction=[r for r in rows if r['benchmark']==b and r['role']=='construction']
        if not construction:continue
        forms=[f for f in FORMS if b=='androidworld' or f!='M4']
        # Only complete paired tasks enter selection/reporting.
        paired={r['id'] for r in construction if {x['form'] for x in construction if x['id']==r['id']}==set(forms)}
        if not paired:continue
        best=max(forms,key=lambda f:(sum(r['score'] for r in construction if r['form']==f and r['id'] in paired)/len(paired),
            -sum(r.get('cost',{}).get('usage',{}).get('total_tokens',0) for r in construction if r['form']==f and r['id'] in paired),-forms.index(f)))
        for key in sorted({r['id'] for r in rows if r['benchmark']==b and r['role']=='test'}):
            values={r['form']:r['score'] for r in rows if r['benchmark']==b and r['role']=='test' and r['id']==key}
            if set(values)!=set(forms):continue
            choices=read(root/'choices'/b/'test'/f'{key}.json')
            strategies.append(dict(benchmark=b,id=key,best_fixed_form=best,fixed=values[best],oracle=max(values.values()),
                task=values[choices['choices']['task']['form']],capability=values[choices['choices']['capability']['form']],
                random={seed:values[f] for seed,f in choices['random_choices'].items()},
                selector_cost={strategy:dict(total_tokens=choices['choices'][strategy]['call']['usage'].get('total_tokens',0)+(choices['capability_call']['usage'].get('total_tokens',0) if strategy=='capability' else 0),
                    seconds=choices['choices'][strategy]['call']['seconds']+(choices['capability_call']['seconds'] if strategy=='capability' else 0)) for strategy in ('task','capability')},
                fixed_basis='construction replay; partial pilot only' if len(paired)<20 else '20 construction replay tasks',
                fixed_submission=read(root/'fixed_forms'/f'{b}.json') if (root/'fixed_forms'/f'{b}.json').exists() else None))
    strata={}
    for r in rows:
        if r['role']!='test':continue
        old=next(x for x in read(root/'selection.json')[r['benchmark']]['test'] if x['id']==r['id'])
        group='/'.join([r['benchmark'],old['category'],r['form']])
        strata.setdefault(group,[]).append(r['score'])
    cfg=read(root/'config.json')
    expected_keys={(b,role,t['id'],form) for b in cfg['benchmarks'] for role,t in tasks(root,b,cfg['limit_per_role']) for form in (list(FORMS) if b=='androidworld' else list(FORMS)[:4])}
    actual_keys={(r['benchmark'],r['role'],r['id'],r['form']) for r in rows}
    expected=len(expected_keys)
    if actual_keys==expected_keys and any(not all(p[k] for k in ('paired_core_and_ids','actual_core_matches_source','no_memory_empty')) for p in protocol):raise ValueError('Completed evaluation violates paired protocol')
    result=dict(completed_results=len(rows),expected_results=expected,complete=actual_keys==expected_keys,missing_results=sorted(expected_keys-actual_keys),unexpected_results=sorted(actual_keys-expected_keys),protocol_checks=protocol,
                frozen_sources_and_assets=audit(root),rows=rows,groups={k:dict(n=len(v),mean=sum(v)/len(v)) for k,v in groups.items()},
                strategies=strategies,strata={k:dict(n=len(v),mean=sum(v)/len(v)) for k,v in strata.items()},limitation='Exploratory smoke results; capability step outcomes require separate annotation; not full experiment or multi-round evolution.')
    write_json(root/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ART/'capcomem_motivation/pilot_v2')
    p.add_argument('--stage',choices=['prepare','assets','select','evaluate','report','all'],default='all')
    p.add_argument('--benchmarks',nargs='+',choices=['mpdocvqa','androidworld'],default=['mpdocvqa','androidworld'])
    p.add_argument('--limit-per-role',type=int,default=1,help='0 runs complete fixed lists; default one construction + one test per benchmark')
    p.add_argument('--model-urls',nargs='+',default=['http://127.0.0.1:8113/v1','http://127.0.0.1:8114/v1'])
    p.add_argument('--embedding-url',default='http://127.0.0.1:8087')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-steps',type=int,default=30)
    args=p.parse_args()
    if args.limit_per_role<0 or args.workers<1 or args.max_steps<1:p.error('Invalid limit/workers/step budget')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze(root/'config.json',{'protocol':'strict_gui_actions_v2',**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='stage'}})
        if args.stage in ('prepare','all'):prepare(root)
        if args.stage in ('assets','select','evaluate','all'):
            import urllib.request
            for url in args.model_urls:
                with urllib.request.urlopen(url+'/models',timeout=15) as r:
                    if 'qwen3-vl-8b-instruct' not in [m['id'] for m in json.load(r)['data']]:raise ValueError('Wrong solver model')
            items={b:tasks(root,b,args.limit_per_role) for b in args.benchmarks}
        if args.stage in ('assets','all'):
            for b in args.benchmarks:
                for role,t in items[b]:
                    r=retrieve(root,b,t['id'],t['prompt'],args.embedding_url)
                    n=next(n for n in read(root/'sources.json')['nodes'] if n['id']==r['selected'][0])
                    asset(root,n,args.model_urls[0])
        if args.stage in ('select','all'):
            for b in args.benchmarks:
                for role,t in items[b]:choose(root,b,role,t,args.model_urls[0])
        if args.stage in ('evaluate','all'):
            audit(root)
            # All selector submissions must predate execution, including when using separate stages.
            for b in args.benchmarks:
                for role,t in items[b]:
                    if not (root/'choices'/b/role/f'{t["id"]}.json').exists():raise ValueError('Submit selectors before executing tasks')
            for phase in ('construction','test'):
                jobs=[]
                with ThreadPoolExecutor(max_workers=args.workers) as pool, heartbeat('CapCoMem paired pilot '+phase):
                    if 'mpdocvqa' in items:
                        for i,(role,t) in enumerate(items['mpdocvqa']):
                            if role!=phase:continue
                            for j,form in enumerate(list(FORMS)[:4]):
                                jobs.append(pool.submit(run_mp,root,role,t,form,args.model_urls[(i+j)%len(args.model_urls)]))
                    if 'androidworld' in items:
                        from gui_baseline.parallel_eval import shards
                        phase_items=[(r,t) for r,t in items['androidworld'] if r==phase]
                        groups=shards([t['id'] for _,t in phase_items],min(2,args.workers))
                        for i,group in enumerate(groups):
                            jobs.append(pool.submit(run_android,root,[(r,t) for r,t in phase_items if t['id'] in group],
                                args.model_urls[i%len(args.model_urls)],f'http://127.0.0.1:{15003+i}',f'gui_async_aw_{i}',args.max_steps))
                    errors=[]
                    for future in as_completed(jobs):
                        try:
                            future.result()
                            write_json(root/'progress.json',dict(phase=phase,completed=sum(f.done() and f.exception() is None for f in jobs),expected=len(jobs),updated_at=time.time()))
                        except Exception as exc:
                            errors.append(dict(error=str(exc),traceback=traceback.format_exc()));write_json(root/'errors.json',errors)
                            print(errors[-1]['traceback'],flush=True)
                    if errors:raise RuntimeError('Pilot failed; inspect errors.json; completed conditions are resumable')
                if phase=='construction':
                    for b in args.benchmarks:freeze_fixed(root,b,args.limit_per_role)
        if args.stage in ('report','evaluate','all'):
            for b in args.benchmarks:
                forms=list(FORMS) if b=='androidworld' else list(FORMS)[:4]
                construction=[t['id'] for role,t in tasks(root,b,args.limit_per_role) if role=='construction']
                if all((root/'runs'/b/'construction'/key/form/'result.json').exists() for key in construction for form in forms):
                    freeze_fixed(root,b,args.limit_per_role)
            report(root)


if __name__=='__main__':main()
