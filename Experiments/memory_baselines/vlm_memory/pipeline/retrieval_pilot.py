"""Resumeable KSI 5/5 pilot reusing fixed no-memory trajectories and controls."""
import argparse
import fcntl
import json
import random
import shutil
import time
from pathlib import Path
from ..config import ROOT, load_config, write_json
from ..prompts import snapshot_prompts, get_task_prompt
from ..prompts import retrieval, ksi_style
from .run_100_100_nomemory import heartbeat
from .pilot import construct, evaluate
from ..memory.bank import MemoryBank
from ..memory.retrieval import DenseRetriever


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = ROOT / 'artifacts/vision2code' / config['run_name']
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with heartbeat("KSI vision2code 5/5"):
            run(root, config)


def run(root, config):
    source = ROOT / 'artifacts/vision2code' / config['source_run']
    snapshot = root / 'config.snapshot.json'
    if snapshot.exists() and read(snapshot) != config:
        raise ValueError('Configuration changed; use a new run')
    write_json(snapshot, config)
    write_json(root / 'model.json', {'name': config['model']['name'], 'path': '/data/pretrained_models/Qwen3-VL-8B-Instruct', 'embedding_path': config['retrieval']['embedding_model']})
    prompt_snapshots = {
        'base_prompts.snapshot.json': snapshot_prompts(config),
        'retrieval_prompts.snapshot.json': {'query': retrieval.QUERY_SYSTEM, 'index': retrieval.INDEX_SYSTEM,
            'compress': retrieval.COMPRESS_SUFFIX, 'rerank': retrieval.RERANK_SYSTEM,
            'fields_schema': retrieval.FIELDS, 'rank_schema': retrieval.RANK_SCHEMA},
        'construction_prompts.snapshot.json': {'review': ksi_style.REVIEW_SYSTEM,
            'distill': ksi_style.DISTILL_SYSTEM, 'cross_task': ksi_style.CROSS_TASK_SUFFIX},
    }
    for name, values in prompt_snapshots.items():
        saved = root / name
        if saved.exists() and read(saved) != values:
            raise ValueError('Preserved prompts changed; use a new run: ' + name)
        write_json(saved, values)
    original = read(source / 'dataset_manifest.json')
    source_config = read(source / 'config.snapshot.json')
    for key in ['name', 'temperature', 'max_tokens', 'image_max_side']:
        if source_config['model'][key] != config['model'][key]:
            raise ValueError('No-memory model configuration mismatch: ' + key)
    # Five largest strata; selection uses only public category and fixed random seed.
    categories = ['Charts&Plots', 'Science', 'Geometry', 'Documents', 'Graphs']
    rng = random.Random(config['sampling']['seed'])
    manifest = {**original, 'memory': [], 'test': [], 'selection': {'seed': config['sampling']['seed'], 'categories': categories, 'rule': 'one random task per category per experimental split; no score filtering', 'omitted_category': 'Spatial'}}
    for split in ['memory', 'test']:
        for category in categories:
            choices = sorted((t for t in original[split] if t['metadata']['category'] == category), key=lambda t:t['id'])
            manifest[split].append(rng.choice(choices))
    saved_manifest = root / 'dataset_manifest.json'
    if saved_manifest.exists() and read(saved_manifest) != manifest:
        raise ValueError('Frozen pilot split changed')
    write_json(saved_manifest, manifest)
    for split in ['memory', 'test']:
        for task in manifest[split]:
            old = source / 'runs/no_memory' / split / task['id'] / 'attempt_0'
            result = read(old / 'result.json')
            generation = read(old / 'calls/generation.json')
            expected_system, expected_user = get_task_prompt('vision2code', config.get('prompt_version'))
            if result['status'] != 'completed' or generation['system'] != expected_system or generation['prompt'] != expected_user.format(task=task['prompt']) or generation.get('requested_max_tokens') != config['model']['max_tokens']:
                raise ValueError('No-memory source incomplete or prompt mismatch')
            if split == 'memory':
                folder = root / 'histories' / task['id']
                if (folder / 'episode.json').exists():
                    continue
                episode = read(source / 'trajectories/memory' / (task['id']+'.json'))
                folder.mkdir(parents=True, exist_ok=True)
                for i, image in enumerate(episode['images']):
                    asset = Path(image['path'])
                    target = folder / 'images' / (str(i) + asset.suffix)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(asset, target)
                    image['path'] = str(target.relative_to(folder))
                write_json(folder / 'episode.json', episode)
                (folder / 'history.jsonl').write_text(json.dumps(episode, ensure_ascii=False)+'\n')
            else:
                target = root / 'runs/no_memory' / task['id'] / 'seed_0/attempt_0'
                if not target.exists():
                    shutil.copytree(old, target)
                    result.update(method='no_memory', reused_from=str(old))
                    write_json(target / 'result.json', result)
    print('PHASE construction: 5 fixed no-memory histories; no test feedback enters memory', flush=True)
    construct(root, manifest, config, True)
    print('PHASE frozen dense index', flush=True)
    bank = MemoryBank(root / 'bank.json', config)
    DenseRetriever(bank, config)
    started = time.monotonic()
    for i, task in enumerate(manifest['test']):
        evaluate(root, manifest, config, True, task_ids=[task['id']])
        elapsed = time.monotonic()-started
        print(f'PROGRESS {i+1}/5 elapsed={elapsed:.0f}s ETA={elapsed/(i+1)*(4-i):.0f}s', flush=True)
    from ..analysis.retrieval_report import report
    report(root)
    print('COMPLETE ' + str(root), flush=True)


if __name__ == '__main__':
    main()
