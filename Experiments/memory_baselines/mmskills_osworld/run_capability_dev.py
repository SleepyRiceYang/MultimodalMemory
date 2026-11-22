"""Paired local-model development runs; author-mapped candidates, native worker."""
import argparse
import json
import concurrent.futures
import multiprocessing
from pathlib import Path

import run as native

ORIGINAL_MAKE_AGENT = native.make_agent


def make_development_agent(condition, args):
    if condition == 'visual_skill':
        return ORIGINAL_MAKE_AGENT(condition, args)
    from capability_agent import CapabilitySkillAgent
    from capability_v2_agent import CapabilityV2Agent
    agent_class = CapabilityV2Agent if condition.endswith('_v2') else CapabilitySkillAgent
    return agent_class(model=args['model'], api_model=args['model'],
        base_url=args['base_url'], api_key='EMPTY', max_tokens=args['max_tokens'],
        temperature=0, top_p=.9, action_space='pyautogui', observation_type='screenshot',
        coordinate_type='relative', max_trajectory_length=3, save_conversation_json=True,
        llm_request_timeout=args['request_timeout'], llm_max_retries=1,
        skills_library_dir=args['skills_dir'], skill_mode='multimodal',
        capability_dir=args['capability_dir'], capability_mode='text' if condition.startswith('cap_text') else 'cap')


def development_worker(job):
    native.make_agent = make_development_agent
    return native.worker(job)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sidecars', type=Path, required=True)
    p.add_argument('--vm-path', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8170/v1')
    p.add_argument('--task-split', type=Path, default=native.SPLITS / 'smoke_3.json')
    p.add_argument('--conditions', nargs='+', choices=['visual_skill', 'cap_text', 'cap', 'cap_text_v2', 'cap_v2'],
                   default=['visual_skill', 'cap_text', 'cap'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    a.sidecars = a.sidecars.resolve()
    if a.workers < 1:
        p.error('--workers must be positive')
    tasks = native.read(a.task_split)
    frozen = native.read(native.SPLITS / 'test_40.json')
    if not tasks or not all(ids and set(ids) <= set(frozen[d]) for d, ids in tasks.items()):
        p.error('Development tasks must be a nonempty subset of the frozen 40')
    mapping = native.read(native.BASELINE / 'task_skill_mappings/task_skill_mapping.json')
    skills = sorted({s for d, ids in tasks.items() for t in ids for s in mapping[d][t]})
    sidecars = {s: native.read(a.sidecars / (s + '.json')) for s in skills}
    for s, row in sidecars.items():
        source = next((native.ROOT / 'Datasets/MMSkills/ubuntu').glob('*/' + s))
        for filename, expected in row['source_sha256'].items():
            if native.digest(source / filename) != expected:
                raise ValueError(f'Stale annotation: {s}/{filename}')
    options = dict(output=str(a.output.resolve()), base_url=a.base_url,
        model='qwen3-vl-8b-instruct', max_tokens=65536, context_window=65536,
        max_steps=20, request_timeout=300, reset_wait=60, evaluate_wait=20,
        action_wait=2, top_k=6, shortlist_size=20,
        skills_dir=str(native.ROOT / 'Datasets/MMSkills/ubuntu'), vm_path=a.vm_path,
        capability_dir=str(a.sidecars))
    spec = dict(options=options, tasks=tasks, conditions=a.conditions, workers=a.workers,
        source_sha256={f: native.digest(Path(__file__).with_name(f)) for f in
                       ('run_capability_dev.py', 'capability_agent.py', 'capability_v2_agent.py', 'run.py')},
        sidecar_sha256={s: native.digest(a.sidecars / (s + '.json')) for s in skills},
        mapping_sha256=native.digest(native.BASELINE / 'task_skill_mappings/task_skill_mapping.json'),
        task_model_only=True, author_mapping=True, development_only=True)
    print(json.dumps(spec, indent=2))
    if a.dry_run:
        return
    # Fresh output prevents mixing different prompt versions or interrupted attempts.
    a.output.mkdir(parents=True, exist_ok=False)
    native.write(a.output / 'config.json', spec)
    rows = []
    jobs = [dict(args=options, condition=c, domain=d, task=t)
            for d, ids in tasks.items() for t in ids for c in a.conditions]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(development_worker, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            native.write(a.output / 'progress.json', dict(planned=len(jobs), rows=rows))
    native.write(a.output / 'summary.json', {'rows': rows})


if __name__ == '__main__':
    main()
