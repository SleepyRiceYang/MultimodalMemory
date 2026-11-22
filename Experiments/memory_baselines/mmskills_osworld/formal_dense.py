"""Fourth-arm 40-task evaluation with MiniCPM recall and 15-minute progress logs."""
import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
from pathlib import Path
import shlex
import sys

import requests
import run as native
import dense_retrieval as dense
from run_dense_smoke import retrieve_dense
import formal_three as monitoring

CONDITION = 'visual_skill_retrieval'


def dense_worker(job):
    # Spawn creates a fresh interpreter: apply the retrieval binding in every worker.
    native.retrieve = retrieve_dense
    return native.worker(job)


def run_tasks(options, tasks):
    from mm_agents.skill_loader import SkillLoader
    output = options.output.resolve()
    args = dict(output=str(output), base_url=options.base_url, model='qwen3-vl-8b-instruct',
        max_tokens=65536, context_window=65536, max_steps=20, request_timeout=300,
        reset_wait=60, evaluate_wait=20, action_wait=2, top_k=6,
        skills_dir=str(native.ROOT / 'Datasets/MMSkills/ubuntu'),
        vm_path='/data/yangshengjie/workspace/VLM-Memory/Benchmark/OSWorld/docker_vm_data/Ubuntu.qcow2',
        embedding_url=options.embedding_url, embedding_timeout=600, retrieval_top_k=10,
        embedding_index_dir=str(native.ROOT / 'Experiments/memory_baselines/artifacts/mmskills_osworld/minicpm_index'))
    info = dense.service_info(args)
    spec = {**args, 'tasks': tasks, 'condition': CONDITION, 'workers': options.workers,
        'policy': dense.POLICY, 'max_selected_skills': 3,
        'split_sha256': native.digest(monitoring.SPLIT), 'embedding_model': {k: info[k] for k in
            ('model', 'revision', 'dimension', 'normalized', 'max_tokens', 'pooling', 'query_prefix')},
        'source_sha256': {str(p.relative_to(native.ROOT)): native.digest(p) for p in
            [Path(__file__), Path(native.__file__), Path(dense.__file__),
             Path(__file__).with_name('run_dense_smoke.py'), Path(__file__).with_name('evaluation_audit.py')]},
        'prompt_source_sha256': {str(p.relative_to(native.BASELINE)): native.digest(p)
                                for p in (native.BASELINE / 'mm_agents').rglob('*.py')},
        'retrieval_inputs': 'task text and skill metadata only; no task mapping, screenshot or evaluator'}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = output / 'config.json'
        if config.exists() and native.read(config) != spec:
            raise ValueError('Output configuration changed; use a fresh run directory')
        native.write(config, spec)
        # Build/reuse the shared immutable index once before spawning GUI environments.
        metadata = SkillLoader(skills_library_dir=args['skills_dir']).discover_all_skills()
        _, _, index = dense.load_index(metadata, args)
        native.write(output / 'embedding_index.json', index)
        jobs = [dict(args=args, condition=CONDITION, domain=d, task=t) for d, ids in tasks.items() for t in ids]
        rows, failures = [], []
        with concurrent.futures.ProcessPoolExecutor(max_workers=options.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(dense_worker, job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    failures.append({'domain': job['domain'], 'task': job['task'], 'error': repr(exc)})
                    native.write(output / 'worker_errors.json', failures)
                native.write(output / 'progress.json', {'planned': len(jobs), 'finished': len(rows),
                    'worker_errors': failures, 'rows': rows})
        valid = [r for r in rows if r['status'] == 'completed']
        summary = {'planned': len(jobs), 'completed': len(valid),
            'errors': len(rows) - len(valid) + len(failures), 'policy': dense.POLICY,
            'mean_score_completed_only': sum(r['score'] for r in valid) / len(valid) if valid else None,
            'mean_score_full_suite': sum(r['score'] for r in valid) / len(jobs) if len(valid) == len(jobs) else None,
            'worker_errors': failures, 'rows': rows}
        native.write(output / 'summary.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2), flush=True)
        return 1 if summary['errors'] else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=native.ROOT / 'Experiments/memory_baselines/artifacts/mmskills_osworld/formal_40_dense_20261111_v1')
    p.add_argument('--base-url', default='http://127.0.0.1:8171/v1')
    p.add_argument('--embedding-url', default='http://127.0.0.1:8187')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--execute-run', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.workers < 1:
        p.error('--workers must be positive')
    tasks = native.read(monitoring.SPLIT)
    if sum(map(len, tasks.values())) != 40:
        raise ValueError('Expected the shared frozen 40-task split')
    if a.execute_run:
        raise SystemExit(run_tasks(a, tasks))
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute-run',
        '--base-url', a.base_url, '--embedding-url', a.embedding_url,
        '--workers', str(a.workers), '--output', str(a.output.resolve())]
    print(shlex.join(command), flush=True)
    print('Condition: visual_skill_retrieval; 40 tasks; 20 steps; action wait 2s; MiniCPM top10 -> LLM <=3/null', flush=True)
    print('Progress: every 900 seconds ->', a.output.resolve() / 'progress.log', flush=True)
    if a.dry_run:
        return
    response = requests.get(a.base_url.rstrip('/') + '/models', timeout=15)
    response.raise_for_status()
    if not any(m['id'] == 'qwen3-vl-8b-instruct' and m.get('max_model_len', 65536) >= 65536
               for m in response.json()['data']):
        raise ValueError('Expected qwen3-vl-8b-instruct with 65536 context')
    dense.service_info({'embedding_url': a.embedding_url})
    # Configure only this process's monitor; the running three-arm monitor is unaffected.
    monitoring.CONDITIONS = (CONDITION,)
    raise SystemExit(monitoring.supervise(command, a.output.resolve(), tasks, interval=900))


if __name__ == '__main__':
    main()
