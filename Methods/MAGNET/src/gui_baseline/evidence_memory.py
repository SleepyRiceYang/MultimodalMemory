"""Build frozen, source-grounded state/view metadata for B2/B3."""
import argparse,copy,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from .cold_continual import ROOT,Model,parse_json
from .common import read,write

ANNOTATE='''Describe reusable visual evidence supported ONLY by the supplied historical material.
This is memory construction, not evaluation. Do not solve a new task. Do not preserve literal
names, numbers, dates, answers or coordinates as reusable facts. Do not invent unsupported
preconditions or negative examples. Unknown conditions must be "unknown".
Return JSON {"when_to_use":"...","when_not_to_use":"...","visible_cues":["..."],
"verification_cue":"...","evidence_purpose":"locate_control|recognize_state|verify_result|unknown"}.
The evidence is historical; BEFORE/AFTER are not the current environment. A successful local
transition does not prove the whole task succeeded.'''


def build(args):
    source=Path(args.memory);original=read(source);bank=copy.deepcopy(original)
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    provenance=source.parent.parent
    jobs=[]
    for kind in ['procedural','stationary']:
        for node in bank[kind]:
            views=[]
            if kind=='stationary':
                # Keep distinct instances; a crop has no fabricated full-frame semantics.
                for image in node['images']:
                    views.append({'view_id':f'patch_{image["id"]}','view_type':'focus_crop',
                                  'path':image['path'],'source':image['source']})
            else:
                for task in node['source']:
                    for number in range(3):
                        p=provenance/f'round_{number:02d}/tasks'/task/'checkpoint.json'
                        if not p.exists():continue
                        episode=read(p)['episode']
                        if not episode['success']:continue
                        transitions=[x for x in episode['transitions'] if x['action']['action_type'] not in ['swipe','scroll']]
                        if not transitions:continue
                        # At most three observed milestones per source workflow, not assumed state labels.
                        for idx in sorted({0,len(transitions)//2,len(transitions)-1}):
                            t=transitions[idx]
                            for side in ['before','after']:
                                views.append({'view_id':f'r{number}_{task}_s{t["step"]}_{side}',
                                    'view_type':side,'path':t[side],'source':{'task':task,'round':number,'step':t['step']},
                                    'observed_action':t['description']})
                        break
            assert views,('No historical evidence',kind,node['id'])
            assert all(Path(v['path']).exists() for v in views)
            # Annotation sees one real local instance; all other instances remain separately attributed.
            selected=views[:2] if kind=='procedural' else views[:1]
            jobs.append((kind,node,views,selected))
    def annotate(job):
        kind,node,views,selected=job
        model=Model(SimpleNamespace(model_url=args.model_url,model=args.model,temperature=0.,
                                   max_tokens=1400,max_model_attempts=3))
        text=model.call(folder/'annotations'/kind/str(node['id']),ANNOTATE,
            json.dumps({'knowledge':{k:node[k] for k in (['title','actions'] if kind=='procedural' else ['function'])},
                        'observations':selected},ensure_ascii=False),
            [(f'HISTORICAL {v["view_type"]}',v['path']) for v in selected])
        metadata=parse_json(text)
        required={'when_to_use','when_not_to_use','visible_cues','verification_cue','evidence_purpose'}
        if set(metadata)!=required:raise ValueError('Annotation fields differ from schema')
        node['evidence_state']={**metadata,'state_id':f'{kind}_{node["id"]}',
            'annotation_view_ids':[v['view_id'] for v in selected],'available_views':views}
        # No annotation is silently generalized to other historical instances.
        return node['id']
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for identity in pool.map(annotate,jobs):print('ANNOTATED',identity,flush=True)
    bank['evidence_protocol']='source_grounded_b2_b3_v1'
    bank['source_memory']=str(source.resolve())
    write(folder/'memory.json',bank)
    write(folder/'construction_protocol.json',{'system_prompt':ANNOTATE,'model':args.model,
        'model_url':args.model_url,'workers':args.workers,'nodes':len(jobs),
        'note':'Conditions grounded in annotation_view_ids only; no evaluation result consulted.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--memory',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-url',default='http://127.0.0.1:8177/v1');p.add_argument('--model',default='qwen3-vl-8b-instruct')
    p.add_argument('--workers',type=int,default=6);build(p.parse_args())
