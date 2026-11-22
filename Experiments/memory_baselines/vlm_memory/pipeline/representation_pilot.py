"""Resumable single-source/single-test validation of the five representation baselines."""
import argparse
import fcntl
import json
import shutil
from pathlib import Path

from ..config import ROOT, load_config, write_json
from ..memory.bank import MemoryBank
from ..memory.representations import request, obj, array, STRING, assets
from ..memory.retrieval import DenseRetriever
from ..prompts import get_task_prompt, memory_audit
from .pilot import construct, evaluate, read
from .run_100_100_nomemory import snapshot_run, heartbeat


def prepare(root, config):
    benchmark = config['benchmarks'][0]
    source = ROOT / 'artifacts' / benchmark / config['source_run']
    original = read(source / 'dataset_manifest.json')
    sample = config['diagnostic_sample']
    manifest = {**original}
    for split, key in [('memory', 'construction_id'), ('test', 'test_id')]:
        manifest[split] = [t for t in original[split] if t['id'] == sample[key]]
        if len(manifest[split]) != 1:
            raise ValueError(f'Sample is not in the original {split} split')
    if sample['construction_id'] == sample['test_id']:
        raise ValueError('Construction/test overlap')
    manifest['selection'] = sample
    saved = root / 'dataset_manifest.json'
    if saved.exists() and read(saved) != manifest:
        raise ValueError('Diagnostic sample changed; use a new run')
    write_json(saved, manifest)
    task = manifest['memory'][0]
    old = source / 'runs/no_memory/memory' / task['id'] / 'attempt_0'
    result, generation = read(old / 'result.json'), read(old / 'calls/generation.json')
    system, user = get_task_prompt(benchmark, config.get('prompt_version'))
    source_config = read(source / 'config.snapshot.json')
    for key in ['name', 'temperature', 'max_tokens', 'image_max_side']:
        if source_config['model'][key] != config['model'][key]:
            raise ValueError('No-memory construction configuration mismatch: ' + key)
    if result['status'] != 'completed' or generation['system'] != system or generation['prompt'] != user.format(task=task['prompt']):
        raise ValueError('Construction source is incomplete or uses different prompts')
    folder = root / 'histories' / task['id']
    if not (folder / 'episode.json').exists():
        episode = read(source / 'trajectories/memory' / (task['id']+'.json'))
        episode['generation_truncated'] = result['generation_truncated']
        for i, image in enumerate(episode['images']):
            asset = Path(image['path'])
            target = folder / 'images' / (str(i)+asset.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset, target)
            image['path'] = str(target.relative_to(folder))
        write_json(folder / 'episode.json', episode)
        (folder / 'history.jsonl').write_text(json.dumps(episode, ensure_ascii=False)+'\n')
    return manifest


def audit(root, config, manifest):
    bank = MemoryBank(root / 'bank.json', config)
    episodes = []
    for task in manifest['memory']:
        folder = root / 'histories' / task['id']
        e = read(folder / 'episode.json')
        for a in e['images']:
            a['path'] = str(folder.relative_to(root)/a['path'])
        episodes.append(e)
    nodes = bank.data['methods'].get('ksi_visual', bank.data['methods'].get('ksi_style', [])) + bank.data['methods'].get('magnet_adapted', [])
    audit_rows = []
    if nodes:
        schema = obj(dict(assessments=array(obj(dict(id=dict(enum=[n['id'] for n in nodes]),
            status=dict(enum=['supported', 'unsupported', 'uncertain']), reason=STRING, evidence=STRING)), len(nodes), len(nodes))))
        result = request(root, config, 'memory_audit', memory_audit.AUDIT_SYSTEM, dict(episodes=episodes, nodes=nodes),
            assets(root, episodes) + [(f"Memory patch {n['id']}", root/a['path']) for n in nodes for a in n['images']], schema)
        audit_rows = result['assessments']
        ids = [r['id'] for r in audit_rows]
        if len(set(ids)) != len(ids) or set(ids) != {n['id'] for n in nodes}:
            raise ValueError('Audit omitted or invented a node')
    report = dict(assessments=audit_rows, node_counts={m:len(v) for m,v in bank.data['methods'].items()},
        raw_memory='verbatim source task, trajectory and measured feedback; no inferred knowledge',
        ksi_pair='same node IDs, source IDs, text and index; different image payload only',
        limitation='same local model audit; manual source/patch review remains necessary; not proof of correctness',
        bank_frozen=bank.verify_frozen())
    write_json(root / 'memory_audit.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare','build','audit','index','evaluate','all'], default='all')
    args = parser.parse_args()
    config = load_config(args.config)
    root = ROOT / 'artifacts' / config['benchmarks'][0] / config['run_name']
    root.mkdir(parents=True, exist_ok=True)
    with (root/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot_run(root, config)
        manifest = prepare(root, config)
        if args.stage in ['build','all']:
            with heartbeat('Five-baseline memory construction'):
                construct(root, manifest, config, True)
        if args.stage in ['audit','all']:
            with heartbeat('Memory evidence audit'):
                audit(root, config, manifest)
        if args.stage in ['index','all']:
            with heartbeat('Frozen dense retrieval index'):
                DenseRetriever(MemoryBank(root/'bank.json',config),config)
        if args.stage in ['evaluate','all']:
            with heartbeat('Five baselines on one held-out task'):
                evaluate(root, manifest, config, True)
            results = read(root/'summary.json')
            contexts = {m:read(root/'runs'/m/manifest['test'][0]['id']/'seed_0/attempt_0/memory_trace.json')
                        for m in ['ksi_style','ksi_visual']}
            paired = (contexts['ksi_style']['text'] == contexts['ksi_visual']['text'] and
                [e['id'] for e in contexts['ksi_style']['selected']] == [e['id'] for e in contexts['ksi_visual']['selected']])
            if not paired:
                raise ValueError('Actual KSI injection violates paired comparison')
            write_json(root/'diagnostic_summary.json',dict(results=results,paired_ksi_text_and_selection=paired,
                interpretation='One deliberately selected diagnostic example; no population performance conclusion'))
        print('COMPLETE '+str(root),flush=True)


if __name__ == '__main__':
    main()
