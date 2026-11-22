"""Launch the three-arm 40-task evaluation and persist progress every 15 minutes."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CONDITIONS = ('no_skill', 'text_skill', 'visual_skill')
SPLIT = ROOT / 'Experiments/memory_baselines/splits/mmskills_40_seed20261110/test_40.json'


def snapshot(output, tasks, phase, pid=None, exit_code=None):
    groups, active = {}, []
    for condition in CONDITIONS:
        completed, errors, scores = 0, 0, []
        for domain, ids in tasks.items():
            for task in ids:
                folder = output / condition / domain / task
                path = folder / 'result.json'
                try:
                    row = json.loads(path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if row['status'] == 'completed':
                    completed += 1
                    scores.append(row['score'])
                elif row['status'] == 'running':
                    item = {'condition': condition, 'domain': domain, 'task': task}
                    progress = folder / 'progress.jsonl'
                    if progress.exists():
                        with progress.open('rb') as stream:
                            stream.seek(max(0, progress.stat().st_size - 16384))
                            lines = stream.read().splitlines()
                        if lines:
                            try:
                                item['latest_event'] = json.loads(lines[-1])
                            except json.JSONDecodeError:
                                pass
                    active.append(item)
                else:
                    errors += 1
        planned = sum(map(len, tasks.values()))
        groups[condition] = {'planned': planned, 'completed': completed, 'errors': errors,
            'remaining_including_active': planned - completed - errors,
            'mean_score_completed_only': sum(scores) / len(scores) if scores else None}
    return {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'phase': phase, 'runner_pid': pid, 'exit_code': exit_code,
            'planned': sum(g['planned'] for g in groups.values()),
            'completed': sum(g['completed'] for g in groups.values()),
            'errors': sum(g['errors'] for g in groups.values()), 'conditions': groups, 'active': active}


def supervise(command, output, tasks, interval=900):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'supervisor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output / 'run.log').open('a', buffering=1) as runlog, (output / 'progress.log').open('a', buffering=1) as progress:
            child = subprocess.Popen(command, cwd=ROOT, stdout=runlog, stderr=subprocess.STDOUT, start_new_session=True)
            (output / 'runner.pid').write_text(str(child.pid) + '\n')
            (output / 'supervisor.pid').write_text(str(os.getpid()) + '\n')
            def emit(phase, code=None):
                row = snapshot(output, tasks, phase, child.pid, code)
                progress.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f"{row['time_utc']} {phase}: completed={row['completed']}/{row['planned']} errors={row['errors']}", flush=True)
            old = {}
            def forward(signum, _frame):
                emit('stop_requested')
                if child.poll() is None:
                    os.killpg(child.pid, signum)
            for sig in (signal.SIGINT, signal.SIGTERM):
                old[sig] = signal.signal(sig, forward)
            try:
                emit('started')
                deadline = time.monotonic() + interval
                while child.poll() is None:
                    if time.monotonic() >= deadline:
                        emit('running')
                        deadline += interval
                    time.sleep(min(2, max(.01, deadline - time.monotonic())))
                code = child.wait()
                emit('completed' if code == 0 else 'exited_with_error', code)
                (output / 'exit_code.txt').write_text(str(code) + '\n')
                return code
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'Experiments/memory_baselines/artifacts/mmskills_osworld/formal_40_three_20261110_v1')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--base-url', default='http://127.0.0.1:8170/v1')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    tasks = json.loads(SPLIT.read_text())
    if sum(map(len, tasks.values())) != 40:
        raise ValueError('Expected the frozen 40-task split')
    command = ['bash', str(ROOT / 'Experiments/memory_baselines/scripts/experiments/run_mmskills_four_conditions.sh'),
        '--suite', 'formal', '--conditions', *CONDITIONS, '--workers', str(args.workers),
        '--base-url', args.base_url, '--max-steps', '20', '--action-wait', '2',
        '--max-tokens', '65536', '--context-window', '65536', '--output', str(args.output.resolve())]
    print(shlex.join(command), flush=True)
    print(f'Progress every 900 seconds: {args.output.resolve() / "progress.log"}', flush=True)
    if args.dry_run:
        return
    import requests
    response = requests.get(args.base_url.rstrip('/') + '/models', timeout=15)
    response.raise_for_status()
    if not any(m['id'] == 'qwen3-vl-8b-instruct' and m.get('max_model_len', 65536) >= 65536
               for m in response.json()['data']):
        raise ValueError('Expected qwen3-vl-8b-instruct with a 65536 context window')
    raise SystemExit(supervise(command, args.output.resolve(), tasks))


if __name__ == '__main__':
    main()
