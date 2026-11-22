"""Construction-only MAGNET diagnostics; no held-out task is selected or sent to models."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np

from ..config import ROOT,load_config,write_json
from ..memory.magnet_v2 import Builder,complete_link
from .run_100_100_nomemory import heartbeat


def prepare(benchmark):
    old=next((ROOT/'artifacts'/benchmark).glob('*_four_memory_v1'))
    root=ROOT/'artifacts/diagnostics/magnet_v2_20261025'/benchmark
    root.mkdir(parents=True,exist_ok=True)
    config=load_config(ROOT/'config'/f'qwen3_vl_8b_{benchmark}_four_memory_baselines.json')
    config['memory_construction']={**config['memory_construction'],'magnet_protocol':'described_complete_link_v2'}
    config['baselines']=['magnet_adapted']
    snapshot=root/'config.snapshot.json'
    if snapshot.exists() and json.loads(snapshot.read_text())!=config:
        raise ValueError('Diagnostic configuration changed')
    write_json(snapshot,config)
    manifest=json.loads((old/'dataset_manifest.json').read_text())
    write_json(root/'construction_manifest.json',dict(benchmark=benchmark,memory=manifest['memory'],test=[],
        provenance=str(old),purpose='construction-only diagnostics'))
    episodes=[]
    for task in manifest['memory']:
        source=old/'histories'/task['id']
        e=json.loads((source/'episode.json').read_text())
        for a in e['images']:
            dest=root/'histories'/task['id']/a['path']
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(source/a['path'],dest)
            a['path']=str(dest.relative_to(root))
        episodes.append(e)
    return root,old,config,episodes


def calibrate(root,benchmark,records,similarities):
    labels=json.loads((ROOT/'config/magnet_v2_validation_pairs.json').read_text())[benchmark]
    indices={r['id']:i for i,r in enumerate(records)}
    rows=[]
    for item in labels:
        if item['left'] not in indices or item['right'] not in indices:
            raise ValueError('Label refers to missing construction descriptor')
        rows.append({**item,'similarity':float(similarities[indices[item['left']],indices[item['right']]])})
    training=[r for r in rows if r['split']=='calibration']
    thresholds=sorted({0.,1.,*[r['similarity'] for r in training if r['similarity']>=0],
                       *[r['similarity']+1e-5 for r in training if 0<=r['similarity']<1]})
    def metrics(items,t):
        tp=sum(r['same_workflow'] and r['similarity']>=t for r in items)
        fp=sum(not r['same_workflow'] and r['similarity']>=t for r in items)
        fn=sum(r['same_workflow'] and r['similarity']<t for r in items)
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        return dict(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
                    f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    threshold=max(thresholds,key=lambda t:(metrics(training,t)['f1'],metrics(training,t)['precision'],t))
    config=dict(benchmark=benchmark,threshold=threshold,pairs=rows,
        calibration=metrics(training,threshold),validation=metrics([r for r in rows if r['split']=='validation'],threshold),
        limitation='Small purposive construction-only diagnostic labels; not benchmark-level performance estimates')
    groups=complete_link(similarities,[r['id'] for r in records],threshold)
    config['clusters']=[[records[i]['id'] for i in g] for g in groups]
    config['memberships']=sum(map(len,groups))
    write_json(root/'magnet_v2/calibration.json',config)
    return threshold


def integration(root,old,config,episodes):
    from ..memory.representations import build_bank
    from ..memory.bank import MemoryBank
    from ..memory.retrieval import DenseRetriever
    from .pilot import execute
    memory_ids=['ChartQA_test_000949_d78b49136054','ChartQA_test_001018_99d9c456abfe','ChartQA_test_002183_b6e7675ec75e']
    query_id='ChartQA_test_002169_8d544876cccd'
    child=root/'integration_multimodal'
    child.mkdir(parents=True,exist_ok=True)
    config=json.loads(json.dumps(config))
    config['memory_construction']['magnet_v2_threshold']=json.loads((root/'magnet_v2/calibration.json').read_text())['threshold']
    selected=[]
    for e in episodes:
        if e['id'] not in memory_ids:continue
        for a in e['images']:
            target=child/a['path'];target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(root/a['path'],target)
        selected.append(e)
    manifest=json.loads((old/'dataset_manifest.json').read_text())
    query=next(t for t in manifest['memory'] if t['id']==query_id)
    # This is a construction-only leave-one-source-out diagnostic, not held-out evaluation.
    write_json(child/'dataset_manifest.json',dict(benchmark='vision2code',memory=[t for t in manifest['memory'] if t['id'] in memory_ids],test=[query],
        purpose='All three tasks are from the original memory-construction split; no held-out data'))
    write_json(child/'config.snapshot.json',config)
    if not (child/'bank.json').exists():build_bank(child,config,selected)
    bank=MemoryBank(child/'bank.json',config)
    context=bank.context('magnet_adapted','vision2code',query_id,query['query'],Path(query['images'][0]),config['memory'],task=query)
    folder=child/'query_execution'
    result_path=folder/'result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else execute(query,context,folder,config,child)
    write_json(child/'integration_validation.json',dict(bank_frozen=bank.verify_frozen(),nodes=len(bank.data['methods']['magnet_adapted']),
        retrieved=len(context['selected']),images=len(context['images']),execution_status=result['status'],render_success=result['evaluation'].get('render_success'),
        score=result['score'],source_ids=memory_ids,query_id=query_id,original_query_split='memory',held_out_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True,choices=['mathvision','vision2code','mpdocvqa'])
    p.add_argument('--stage',choices=['describe','calibrate','workflows','replay','visual','integration','all'],default='describe')
    args=p.parse_args()
    root,old,config,episodes=prepare(args.benchmark)
    builder=Builder(root,config)
    with heartbeat('MAGNET v2 '+args.benchmark+' '+args.stage):
        if args.stage in ['describe','all']:
            records,similarities=builder.describe_sources(episodes)
        else:
            records=json.loads((root/'magnet_v2/source_descriptions.json').read_text())
            similarities=np.load(root/'magnet_v2/similarities.npy')
        if args.stage in ['calibrate','workflows','all']:
            threshold=calibrate(root,args.benchmark,records,similarities)
        if args.stage in ['workflows','all']:
            builder.workflows(records,similarities,threshold)
        if args.stage in ['visual','all']:
            # Real positive and failure paths without reconstructing the whole bank.
            selected=[e for e in episodes if e['id'] in json.loads((ROOT/'config/magnet_v2_visual_samples.json').read_text())[args.benchmark]]
            values=builder.visual_candidates(selected)
            groups,audit=builder.deduplicate(values,'inspected_visual_diagnostic')
            write_json(root/'magnet_v2/visual_validation.json',dict(candidates=values,**audit))
        if args.stage in ['replay','all'] and args.benchmark=='vision2code':
            original=json.loads((old/'construction_calls/vision2code_magnet_workflow_groups.request.json').read_text())['payload']['candidates']
            groups,audit=builder.deduplicate(original,'legacy118')
            write_json(root/'magnet_v2/legacy118_replay.json',dict(**audit,count_before=len(original),count_after=len(groups)))
        if args.stage=='integration' and args.benchmark=='vision2code':
            integration(root,old,config,episodes)
    print('COMPLETE',root,flush=True)


if __name__=='__main__':main()
