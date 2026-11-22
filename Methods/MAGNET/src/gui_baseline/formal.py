"""Build a GUI procedural memory bank from exactly 50 frozen No Memory trajectories."""
from __future__ import annotations
import argparse,fcntl
from pathlib import Path
from .build import build_bank
from .common import read,write
from .prompts import BUILD,VERIFY


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',choices=('androidworld','osworld'),required=True)
    p.add_argument('--workspace',default='/root/VLM-Memory')
    p.add_argument('--split-file',required=True)
    p.add_argument('--source-run',default='formal_50_construction_v1')
    p.add_argument('--bank-run',default='formal_50_bank_v1')
    p.add_argument('--model-url')
    p.add_argument('--embedding-url',default='http://127.0.0.1:8087')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    args.model_url=args.model_url or ('http://127.0.0.1:8104/v1' if args.benchmark=='androidworld' else 'http://127.0.0.1:8105/v1')
    manifest=read(args.split_file)
    if manifest['benchmark']!=args.benchmark:raise ValueError('Wrong benchmark split')
    ids=manifest['memory_construction']
    expected=set(ids) if args.benchmark=='androidworld' else {t for rows in ids.values() for t in rows}
    test=manifest['test'];heldout=set(test) if args.benchmark=='androidworld' else {t for rows in test.values() for t in rows}
    if len(expected)!=50 or len(heldout)!=50 or expected&heldout:raise ValueError('Invalid 50/50 split')
    base=Path(args.workspace)/'Experiments/memory_baselines/artifacts/gui_magnet_procedural'/args.benchmark
    source=base/args.source_run;bankroot=base/'banks'/args.bank_run
    if args.dry_run:
        print({'benchmark':args.benchmark,'source':str(source),'source_tasks':len(expected),
               'bank':str(bankroot),'test_tasks':len(heldout),'model_url':args.model_url,
               'embedding_url':args.embedding_url})
        return
    source_config=read(source/'config.json')
    if source_config.get('benchmark')!=args.benchmark or source_config.get('split_role')!='memory_construction':
        raise ValueError('Source run has wrong benchmark/role')
    if read(source/'split.snapshot.json')!=manifest:raise ValueError('Source run split differs from frozen manifest')
    if set(source_config['tasks'])!=expected:raise ValueError('Source run tasks differ from construction manifest')
    bankroot.mkdir(parents=True,exist_ok=True)
    with (bankroot/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        build_config={'benchmark':args.benchmark,'source_run':args.source_run,'source_tasks':sorted(expected),
                      'bank_run':args.bank_run,'model_url':args.model_url,'embedding_url':args.embedding_url,
                      'split':manifest,'build_prompt':BUILD,'verify_prompt':VERIFY}
        snapshot=bankroot/'build_config.json'
        if snapshot.exists() and read(snapshot)!=build_config:raise ValueError('Bank build config changed; use new --bank-run')
        write(snapshot,build_config)
        bank=build_bank(args.workspace,args.benchmark,bankroot,args.model_url,args.embedding_url,
                        source_run=args.source_run,expected_task_ids=expected)
        if set(bank['source_task_ids'])!=expected:raise ValueError('Bank source IDs are incomplete or contaminated')
        print(f'BUILT {args.benchmark} source_tasks={len(expected)} verified_nodes={len(bank["nodes"])} bank={bankroot}',flush=True)

if __name__=='__main__':main()
