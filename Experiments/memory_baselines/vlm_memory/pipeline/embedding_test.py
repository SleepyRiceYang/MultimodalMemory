"""Evaluate existing frozen banks with cosine ranking in an isolated resumable run."""
from __future__ import annotations

import argparse
import copy
import fcntl
import shutil
from pathlib import Path

from ..config import ROOT, load_config, write_json
from ..memory.bank import MemoryBank, inside
from .formal_100_100 import run_evaluation
from .pilot import read
from .run_100_100_nomemory import snapshot_run


def prepare(source: Path, root: Path, config: dict) -> dict:
    if source.resolve() == root.resolve():
        raise ValueError('New retrieval protocol requires a separate result directory')
    previous = read(source / 'config.snapshot.json')
    old, new = copy.deepcopy(previous), copy.deepcopy(config)
    old_options, new_options = old.pop('retrieval'), new.pop('retrieval')
    if old != new:
        raise ValueError('Only retrieval ranking may change when reusing a bank')
    for options in (old_options, new_options):
        for key in ('ranking', 'bounded_rerank'):
            options.pop(key, None)
    if old_options != new_options or config['retrieval'].get('ranking') != 'embedding_cosine':
        raise ValueError('Embedding/index/query settings must remain identical')
    bank = MemoryBank(source / 'bank.json')
    index = read(source / 'retrieval/index.json')
    if index['signature']['options'] != previous['retrieval']:
        raise ValueError('Source index options differ from the source snapshot')
    manifest = read(source / 'dataset_manifest.json')
    provenance = {'source_memory_run': str(source.resolve()), 'bank_sha256': bank.sha256,
                  'ranking': 'embedding_cosine', 'reused_test_results': False}
    marker = root / 'retrieval_reuse.json'
    if marker.exists() and read(marker) != provenance:
        raise ValueError('Retrieval reuse provenance changed')
    snapshot_run(root, config)

    def preserve(original, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != original.read_bytes():
                raise ValueError(f'Preserved artifact changed: {target}')
        else:
            shutil.copy2(original, target)

    for name in ('bank.json', 'dataset_manifest.json'):
        preserve(source / name, root / name)
    for name in bank.asset_hashes:
        preserve(inside(source, name), root / name)
    for task in manifest['test']:
        relative = Path('runs/no_memory') / str(task['id']) / 'seed_0/attempt_0'
        # Reference controls are unchanged; memory-arm results are never imported.
        original, target = source / relative, root / relative
        if not (target / 'result.json').exists():
            shutil.copytree(original, target, dirs_exist_ok=True)
    for path in (source / 'evaluation_targets').glob('*.json'):
        preserve(path, root / 'evaluation_targets' / path.name)
    preserve(source / 'retrieval/vectors.npy', root / 'retrieval/vectors.npy')
    # Only ranking options change; node texts and vector bytes remain identical.
    index['signature']['options'] = config['retrieval']
    target_index = root / 'retrieval/index.json'
    if target_index.exists() and read(target_index) != index:
        raise ValueError('Preserved index changed')
    write_json(target_index, index)
    # Reuse query descriptions, but exclude every old rerank response and trace.
    for path in (source / 'retrieval/calls/queries').glob('*.json'):
        preserve(path, root / 'retrieval/calls/queries' / path.name)
    MemoryBank(root / 'bank.json').verify_frozen()
    write_json(marker, provenance)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--benchmark', required=True, choices=['mathvision', 'vision2code', 'mpdocvqa'])
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if config['benchmarks'] != [args.benchmark]:
        raise ValueError('Benchmark/config mismatch')
    locks = ROOT / 'artifacts/locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f'formal_{args.run}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source, root = [ROOT / 'artifacts' / args.benchmark / name for name in (args.source_run, args.run)]
        manifest = prepare(source, root, config)
        print(f'REUSE frozen bank and index: {source} -> {root}', flush=True)
        if not args.prepare_only:
            run_evaluation(root, manifest, config, args.retry_failed)


if __name__ == '__main__':
    main()
