"""Successful-trajectory clustering and verified workflow abstraction for evolution."""
from __future__ import annotations
import copy,json,re
from pathlib import Path
import numpy as np
from .common import read,write,encode,JsonModel
from .prompts_evolution import ABSTRACT,VERIFY


def episode_from_result(folder, benchmark, task, goal, result, episode_id):
    folder=Path(folder);steps=[]
    if result['score']!=1:return None
    if benchmark=='androidworld':
        dirs=sorted(folder.glob('step_*'))
        for i,d in enumerate(dirs):
            if not (d/'action.json').exists() or (d/'action_error.json').exists():continue
            action=read(d/'action.json')
            if action['action_type']=='status':continue
            after=dirs[i+1]/'screen.png' if i+1<len(dirs) else folder/'final.png'
            if (d/'screen.png').exists() and after.exists():
                steps.append({'step':i,'action':action,'before':str((d/'screen.png').resolve()),'after':str(after.resolve())})
    else:
        for p in sorted(folder.glob('action_???.json')):
            number=int(p.stem.split('_')[1]);action=read(p)['action']
            if action in ('DONE','FAIL','WAIT'):continue
            before=folder/'screen_000.png' if number==0 else folder/f'action_{number-1:03d}_after.png'
            after=folder/f'action_{number:03d}_after.png'
            if not before.exists() or not after.exists():raise ValueError('Missing successful trajectory observations')
            steps.append({'step':number,'action':action,'before':str(before.resolve()),'after':str(after.resolve())})
    if not steps:return None
    return {'task_id':task,'episode_id':episode_id,'goal':goal,'steps':steps,'score':1,
            'result_path':str((folder/'result.json').resolve())}


def instruction_clusters(episodes, url, threshold, max_cliques=256):
    if not episodes:return []
    vectors=np.asarray(encode(url,[e['goal'] for e in episodes]),dtype=np.float32)
    neighbors={i:set() for i in range(len(episodes))}
    for i in range(len(episodes)):
        for j in range(i):
            if float(vectors[i]@vectors[j])>threshold:neighbors[i].add(j);neighbors[j].add(i)
    cliques=[]
    # Pivoted Bron-Kerbosch enumerates maximal cliques, not connected components.
    def visit(chosen, pending, excluded):
        if not pending and not excluded:
            cliques.append(sorted(chosen))
            if len(cliques)>max_cliques:raise ValueError('Too many maximal cliques; explicitly revise construction configuration')
            return
        union=pending|excluded
        pivot=max(sorted(union),key=lambda n:len(pending&neighbors[n])) if union else None
        for node in sorted(pending-(neighbors[pivot] if pivot is not None else set())):
            visit(chosen|{node},pending&neighbors[node],excluded&neighbors[node])
            pending.remove(node);excluded.add(node)
    visit(set(),set(neighbors),set())
    return sorted(cliques,key=lambda c:tuple(c))


def normalize_candidate(candidate, sources):
    if not isinstance(candidate,dict):raise ValueError('Expected workflow object')
    candidate=copy.deepcopy(candidate)
    for key in ('title','app','precondition','goal','postcondition'):
        if not isinstance(candidate.get(key),str) or not candidate[key].strip():raise ValueError('Invalid workflow field: '+key)
    plan=candidate.get('plan')
    if not isinstance(plan,list) or not 1<=len(plan)<=12 or any(not isinstance(p,str) or not p.strip() for p in plan):
        raise ValueError('Invalid workflow plan')
    ids=candidate.get('source_task_ids')
    if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(sources):
        raise ValueError('Workflow has invalid supporting sources')
    if candidate.get('representative_task_id') not in ids:raise ValueError('Representative evidence is not a supporting source')
    corpus=' '.join(sources[t]['goal']+' '+json.dumps(sources[t]['steps'],ensure_ascii=False) for t in ids)
    parameters=[]
    proposed=candidate.get('parameters',[])
    if not isinstance(proposed,list) or any(not isinstance(p,dict) for p in proposed):raise ValueError('Invalid parameter declarations')
    for p in proposed:
        name,value=p.get('name'),p.get('value')
        if not isinstance(name,str) or not re.fullmatch('[a-z][a-z0-9_]{1,40}',name) or not isinstance(value,str) or len(value)<3 or value not in corpus:
            raise ValueError('Parameter literal is not supported by source actions')
        if value.lower() in ('on','off','true','false','create','delete'):raise ValueError('Operation semantics cannot become slots')
        for key in ('title','precondition','goal','postcondition'):candidate[key]=candidate[key].replace(value,'<'+name+'>')
        candidate['plan']=[step.replace(value,'<'+name+'>') for step in candidate['plan']]
        parameters.append({'name':name,'value':value})
    candidate['parameters']=parameters
    return candidate


def update_from_episodes(bank, episodes, root, model_url, options, round_number):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    episodes=[e for e in episodes if e and e['episode_id'] not in bank.data['processed_episodes']]
    if any(e['task_id'] not in bank.data['construction_task_ids'] or e['score']!=1 for e in episodes):
        raise ValueError('Only successful construction trajectories may enter memory')
    write(root/'eligible_episodes.json',episodes)
    cliques=instruction_clusters(episodes,bank.url,options['cluster_threshold'])
    # Bound each prompt without truncating the underlying trajectories. Record all groups.
    def public_source(e):
        # Filesystem paths remain in provenance, never consume model context.
        return {'task_id':e['task_id'],'goal':e['goal'],'score':e['score'],
                'steps':[{'step':s['step'],'action':s['action']} for s in e['steps']]}
    groups=[];oversized=[]
    def pack(ids):
        size=len(json.dumps([public_source(episodes[i]) for i in ids],ensure_ascii=False))
        if size<=options.get('max_build_chars',12000):groups.append(ids)
        elif len(ids)>1:
            middle=len(ids)//2;pack(ids[:middle]);pack(ids[middle:])
        else:oversized.append(episodes[ids[0]]['episode_id'])
    for clique in cliques:
        for i in range(0,len(clique),options['cluster_batch_size']):pack(clique[i:i+options['cluster_batch_size']])
    write(root/'clusters.json',{'maximal_cliques':cliques,'prompt_groups':groups,'threshold':options['cluster_threshold'],
                               'max_build_chars':options.get('max_build_chars',12000),'oversized_episodes':oversized})
    model=JsonModel(model_url);audit=[];new=0
    for index,ids in enumerate(groups):
        group=[episodes[i] for i in ids];sources={e['task_id']:e for e in group}
        images=[];labels=[]
        for e in group:
            for phase,path in [('BEFORE',e['steps'][0]['before']),('AFTER',e['steps'][-1]['after'])]:
                images.append(path);labels.append(f'SOURCE {e["task_id"]} {phase}; NOT current evaluation state')
        payload={'successful_trajectories':[public_source(e) for e in group]}
        print(f'EVOLVE ABSTRACT group={index+1}/{len(groups)} sources={list(sources)}',flush=True)
        response=model.generate(root/'calls'/f'abstract_{index:03d}',ABSTRACT,payload,images,options['build_max_tokens'],image_labels=labels)
        rows=response.get('workflows')
        if not isinstance(rows,list):raise ValueError('Abstraction omitted workflows list')
        for j,row in enumerate(rows[:2]):
            try:node=normalize_candidate(row,sources)
            except (ValueError,TypeError,KeyError) as exc:
                audit.append({'group':index,'candidate':j,'rejected':str(exc)});continue
            support={t:sources[t] for t in node['source_task_ids']}
            verdict=model.generate(root/'calls'/f'verify_{index:03d}_{j}',VERIFY,
                {'candidate':node,'supporting_trajectories':[public_source(e) for e in support.values()]},images,1536,image_labels=labels)
            if verdict.get('supported') is not True or not verdict.get('before_evidence') or not verdict.get('after_evidence'):
                audit.append({'group':index,'candidate':j,'rejected':verdict});continue
            representative=support[node.pop('representative_task_id')]
            node['evidence']=[representative['steps'][0]['before'],representative['steps'][-1]['after']]
            node['verification']=verdict
            identity,added=bank.add(node,[e['episode_id'] for e in support.values()],round_number)
            new+=int(added);audit.append({'group':index,'candidate':j,'accepted':identity,'new':added})
    bank.data['processed_episodes']=sorted(set(bank.data['processed_episodes'])|{e['episode_id'] for e in episodes})
    summary={'eligible_successes':len(episodes),'clusters':len(cliques),'prompt_groups':len(groups),'new_nodes':new,'total_nodes':len(bank.data['nodes']),
             'oversized_episodes':oversized,'audit':audit}
    write(root/'construction_audit.json',summary)
    return summary
