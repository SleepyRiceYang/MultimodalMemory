"""Audit completed checkpoints and actual B2/B3 multimodal request isolation."""
import argparse,hashlib,json
from pathlib import Path
from .common import read,write
from .cold_parallel import memory_digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    root=Path(args.run);configs={c:read(root/c/'config.json') for c in ['B2','B3'] if (root/c/'config.json').exists()}
    first=next(iter(configs.values()));rows=[]
    for c,config in configs.items():
        assert config['memory_sha256']==first['memory_sha256']
        assert config['tasks']==first['tasks'] and config['budgets']==first['budgets']
        assert memory_digest(read(config['evidence_memory']))==config['memory_sha256']
        for key in ['model_url','model','temperature','max_tokens','seed','selection_prompt','use_prompt']:
            assert config[key]==first[key],('Condition confound',key)
        v3=config.get('protocol')=='multistate_subtask_v3'
        if v3:
            for key in ['protocol','stationary_policy','subtask_pre_threshold','subtask_goal_threshold',
                        'subtask_top_k','subtask_prompt','applicability_prompt','source_sha256']:
                assert config[key]==first[key],('Condition confound',key)
            snapshot=read(root/c/'input_memory.json')
            assert snapshot['stationary']==read(snapshot['source_memory'])['stationary']
        checked=0;requests=0;complete=0;loaded=0;actor_checked=0;stationary_loaded=0
        for task in config['tasks']:
            folder=root/c/'tasks'/task;cp=folder/'checkpoint.json'
            if cp.exists():
                e=read(cp)['episode'];assert e['success']==(e['score']==1)
                assert e['task_id']==task and e['budget']==config['budgets'][task]
                assert len(e['transitions'])<=e['budget'];complete+=1
            for p in sorted(folder.glob('step_*/*_evidence.json')):
                role=p.stem.removesuffix('_evidence');evidence=read(p)
                call=p.parent/role/'request.json'
                if not call.exists():continue
                actual=[x[1] for x in read(call)['images']]
                assert Path(actual[0]).resolve()==(p.parent/'before.png').resolve()
                assert actual[1:]==evidence['loaded_images']
                if c=='B2':assert actual[1:]==[]
                else:assert actual[1:]==evidence['selected_images']
                assert len(actual)<=(7 if config.get('protocol')=='multistate_subtask_v3' else 3)
                if config.get('protocol')=='multistate_subtask_v3':
                    from .evidence_eval_v3 import resolve_selection
                    views,cards=resolve_selection(evidence['decision'],evidence['state_metadata'])
                    assert [v['path'] for v in views]==evidence['selected_images']
                    assert cards==evidence['selected_state_cards']
                checked+=1;loaded+=len(actual)-1
            for p in folder.glob('step_*/*/request.json'):
                request=read(p);assert request['model']=='qwen3-vl-8b-instruct';requests+=1
                if v3 and p.parent.name=='actor':
                    step=p.parent.parent
                    trace=step/'stationary_retrieval.json'
                    selected=read(trace)['selected'] if trace.exists() else []
                    nodes={n['id']:n for n in snapshot['stationary']}
                    expected=[next(im['path'] for im in nodes[s['id']]['images'] if im['id']==s['image_id']) for s in selected]
                    assert [x[1] for x in request['images']][1:]==expected
                    assert not (step/'actor_evidence.json').exists()
                    assert not (step/'actor_evidence_selection').exists()
                    actor_checked+=1;stationary_loaded+=len(expected)
        rows.append({'condition':c,'completed':complete,'expected':len(config['tasks']),
                     'checked_planner_actor_requests':checked,'all_requests':requests,'loaded_history':loaded,
                     'gate_scope':'procedural_planner_only' if v3 else 'planner_and_actor',
                     'checked_original_stationary_actor_requests':actor_checked,'loaded_stationary_history':stationary_loaded})
    result={'complete':len(rows)==2 and all(r['completed']==r['expected'] for r in rows),'conditions':rows,
            'checks':'Frozen memory, same task/budget/prompts, official scores, actual request image isolation'}
    write(root/'audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
