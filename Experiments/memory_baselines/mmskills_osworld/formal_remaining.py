"""Run mapped Visual-Skill on the mapped pool minus the frozen 40."""
import argparse
import json
from pathlib import Path
import shlex
import sys

import formal_three as monitor

ROOT = Path(__file__).resolve().parents[3]
CONDITIONS = ('visual_skill',)


def remaining_tasks():
    mapping = json.loads((ROOT / 'Methods/MMSkills_for_Visual_Agents_baseline/task_skill_mappings/task_skill_mapping.json').read_text())
    excluded = json.loads(monitor.SPLIT.read_text())
    pool = {(d, t) for d, ids in mapping.items() for t in ids}
    old = {(d, t) for d, ids in excluded.items() for t in ids}
    if len(pool) != 360 or len(old) != 40 or not old <= pool:
        raise ValueError('Expected 360 mapped tasks containing the frozen 40')
    tasks = {d: sorted(set(ids) - set(excluded.get(d, []))) for d, ids in sorted(mapping.items())}
    assert sum(map(len, tasks.values())) == 320
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'Experiments/memory_baselines/artifacts/mmskills_osworld/formal_remaining_320_visual_20261112_v1')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--base-url', default='http://127.0.0.1:8170/v1')
    p.add_argument('--vm-path', required=True, help='Docker host-visible Ubuntu.qcow2 path')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.workers < 1:
        p.error('--workers must be positive')
    tasks = remaining_tasks()
    output = a.output.resolve()
    split = output / 'task_split_320.json'
    command = ['bash', str(ROOT / 'Experiments/memory_baselines/scripts/experiments/run_mmskills_four_conditions.sh'),
               '--suite', 'formal', '--task-split', str(split), '--conditions', *CONDITIONS,
               '--workers', str(a.workers), '--base-url', a.base_url, '--vm-path', a.vm_path,
               '--max-steps', '20', '--action-wait', '2', '--max-tokens', '65536',
               '--context-window', '65536', '--output', str(output)]
    print('Mapped pool 360 minus frozen 40 = 320 tasks; mapped Visual-Skill only = 320 episodes.')
    print(json.dumps({d: len(ids) for d, ids in tasks.items()}))
    print(shlex.join(command), flush=True)
    if a.dry_run:
        return
    import requests
    response = requests.get(a.base_url.rstrip('/') + '/models', timeout=15)
    response.raise_for_status()
    if not any(m['id'] == 'qwen3-vl-8b-instruct' and m.get('max_model_len', 65536) >= 65536
               for m in response.json()['data']):
        raise ValueError('Expected qwen3-vl-8b-instruct with a 65536 context window')
    output.mkdir(parents=True, exist_ok=True)
    if split.exists() and json.loads(split.read_text()) != tasks:
        raise ValueError('Existing task split differs; use a fresh output directory')
    split.write_text(json.dumps(tasks, indent=2) + '\n')
    monitor.CONDITIONS = CONDITIONS
    sys.exit(monitor.supervise(command, output, tasks))


if __name__ == '__main__':
    main()
