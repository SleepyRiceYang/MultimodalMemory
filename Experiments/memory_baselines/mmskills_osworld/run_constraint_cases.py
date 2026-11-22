"""Four-arm failure-case comparison isolating retrieval and mapped utilization."""
import argparse
import concurrent.futures
import multiprocessing
import json
import shutil
from pathlib import Path

import run as native
from constrained_retrieval import retrieve_frozen
from capability_v3_agent import CapabilityV3Agent

ORIGINAL_AGENT = native.make_agent
CONDITIONS = ('visual_skill', 'visual_skill_retrieval', 'cap_retrieval', 'cap_utilization')
SPLIT = native.ROOT / 'Experiments/memory_baselines/splits/capability_constraints_cases_v1'


def make_agent(condition, args):
    if condition != 'cap_utilization':
        return ORIGINAL_AGENT('visual_skill', args)
    return CapabilityV3Agent(model=args['model'], api_model=args['model'], base_url=args['base_url'],
        api_key='EMPTY', max_tokens=args['max_tokens'], temperature=0, top_p=.9,
        action_space='pyautogui', observation_type='screenshot', coordinate_type='relative',
        max_trajectory_length=3, save_conversation_json=True,
        llm_request_timeout=args['request_timeout'], llm_max_retries=1,
        skills_library_dir=args['skills_dir'], skill_mode='multimodal',
        capability_dir=args['capability_dir'], capability_mode='cap')


def worker(job):
    native.make_agent = make_agent
    native.retrieve = retrieve_frozen
    return native.worker(job)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8170/v1')
    p.add_argument('--vm-path', required=True)
    p.add_argument('--sidecars', type=Path, default=native.ROOT / 'Experiments/memory_baselines/artifacts/capability_constraints/sidecars')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--split-dir', type=Path, default=SPLIT)
    p.add_argument('--retrieval-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.workers < 1:
        p.error('--workers must be positive')
    split = a.split_dir.resolve()
    tasks, pools = native.read(split / 'tasks.json'), native.read(split / 'frozen_top10.json')
    mapping = native.read(native.BASELINE / 'task_skill_mappings/task_skill_mapping.json')
    skills = {s for d, ids in tasks.items() for t in ids for s in mapping[d][t]}
    sidecar_hashes = {}
    if not a.retrieval_only:
        for name in skills:
            path = a.sidecars / (name + '.json')
            row = native.read(path)
            source = next((native.ROOT / 'Datasets/MMSkills/ubuntu').glob('*/' + name))
            for filename, expected in row['source_sha256'].items():
                if native.digest(source / filename) != expected:
                    raise ValueError('Stale capability annotation: ' + name)
            sidecar_hashes[name] = native.digest(path)
    options = dict(output=str(a.output.resolve()), base_url=a.base_url, model='qwen3-vl-8b-instruct',
        max_tokens=65536, context_window=65536, max_steps=20, request_timeout=300,
        reset_wait=60, evaluate_wait=20, action_wait=2, top_k=6, shortlist_size=20,
        skills_dir=str(native.ROOT / 'Datasets/MMSkills/ubuntu'), vm_path=a.vm_path,
        capability_dir=str(a.sidecars.resolve()), operation_catalog_path=str(split / 'operation_catalog.json'))
    jobs = []
    for domain, ids in tasks.items():
        for task in ids:
            config = native.read(native.OSWORLD / 'evaluation_examples/examples' / domain / (task + '.json'))
            source = native.ROOT / pools[task]['source']
            old = native.read(source)
            if old['candidates'] != pools[task]['candidates']:
                raise ValueError('Frozen candidate source changed')
            original_instruction = old['instruction'] if 'instruction' in old else json.loads(
                native.read(source.parent / 'api/request_000.json')['messages'][1]['content'])['instruction']
            if original_instruction != config['instruction']:
                raise ValueError('Task differs from original retrieval query')
            for condition in CONDITIONS:
                if a.retrieval_only and condition not in ('visual_skill_retrieval', 'cap_retrieval'):
                    continue
                args = dict(options, candidate_policy='retrieval' if 'retrieval' in condition else 'mapping',
                            retrieval_mode='cap' if condition == 'cap_retrieval' else 'baseline',
                            frozen_candidates=pools[task]['candidates'])
                jobs.append(dict(args=args, condition=condition, domain=domain, task=task))
    spec = dict(options=options, tasks=tasks, conditions=list(CONDITIONS), planned=len(jobs),
        retrieval_only=a.retrieval_only, workers=a.workers, fixed_top10=True,
        source_sha256={f:native.digest(Path(__file__).with_name(f)) for f in
            ('run_constraint_cases.py','constrained_retrieval.py','capability_agent.py',
             'capability_v2_agent.py','capability_v3_agent.py','run.py')},
        split_sha256=native.digest(split / 'tasks.json'), pools_sha256=native.digest(split / 'frozen_top10.json'),
        sidecar_sha256=sidecar_hashes, mapping_sha256=native.digest(native.BASELINE / 'task_skill_mappings/task_skill_mapping.json'))
    spec['operation_catalog_sha256'] = native.digest(split / 'operation_catalog.json')
    print(f"Planned {len(jobs)} episodes/selection diagnostics", flush=True)
    if a.dry_run:
        return
    a.output.mkdir(parents=True, exist_ok=False)
    native.write(a.output / 'config.json', spec)
    frozen = a.output / 'frozen_source'
    frozen.mkdir()
    for name in spec['source_sha256']:
        shutil.copyfile(Path(__file__).with_name(name), frozen / name)
    inputs = a.output / 'frozen_inputs'
    inputs.mkdir()
    for name in ('tasks.json', 'frozen_top10.json', 'operation_catalog.json'):
        shutil.copyfile(split / name, inputs / name)
    rows = []
    if a.retrieval_only:
        for job in jobs:
            folder = a.output / job['condition'] / job['domain'] / job['task']
            transport = native.RecordedTransport(job['args'], folder)
            instruction = native.read(native.OSWORLD / 'evaluation_examples/examples' / job['domain'] / (job['task'] + '.json'))['instruction']
            names, selection = retrieve_frozen(instruction, job['args'], transport)
            native.write(folder / 'selection.json', {**selection, 'selected_skills': names})
            print(job['condition'], job['task'], names, flush=True)
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            native.write(a.output / 'progress.json', dict(planned=len(jobs), rows=rows))
    native.write(a.output / 'summary.json', {'rows':rows})


if __name__ == '__main__':
    main()
