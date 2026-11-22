"""Single-task dense-retrieval smoke, isolated from the running formal evaluator."""
import argparse
import fcntl
import json
from pathlib import Path

import run as native
import dense_retrieval as dense


def retrieve_dense(instruction, args, transport):
    from mm_agents.skill_loader import SkillLoader
    metadata = SkillLoader(skills_library_dir=args['skills_dir']).discover_all_skills()
    candidates, index = dense.shortlist(instruction, metadata, args)
    result = transport.complete(model=args['model'], temperature=0, top_p=.9,
        messages=[{'role': 'system', 'content':
            'Select reusable desktop skills BEFORE task execution using only the task instruction and '
            'the candidate skill descriptions. Select zero to three directly helpful skills, ordered by relevance. '
            'Do not select unrelated skills to fill a quota. Match the specific task constraints rather than '
            'only the application or topic. Return ONLY a JSON object with "skills": '
            '[exact skill directory names] or null, and "reason": a brief explanation.'},
            {'role': 'user', 'content': json.dumps({'instruction': instruction, 'candidates': candidates})}])
    names, decision = dense.parse_decision(result.choices[0].message.content or '', candidates)
    return names, {'policy': dense.POLICY, **index, 'candidates': candidates,
                   'decision': decision, 'max_selected_skills': 3, 'mapping_used': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8170/v1')
    p.add_argument('--embedding-url', default='http://127.0.0.1:8187')
    p.add_argument('--task', default='386dbd0e-0241-4a0a-b6a2-6704fba26b1c')
    p.add_argument('--retrieval-only', action='store_true')
    p.add_argument('--max-steps', type=int, default=20)
    a = p.parse_args()
    tasks = native.read(native.SPLITS / 'smoke_3.json')
    domains = [d for d, ids in tasks.items() if a.task in ids]
    if len(domains) != 1:
        p.error('--task must be one of the fixed smoke tasks')
    if not 1 <= a.max_steps <= 20:
        p.error('--max-steps must be in 1..20')
    domain = domains[0]
    args = dict(output=str(a.output.resolve()), base_url=a.base_url, model='qwen3-vl-8b-instruct',
        max_tokens=65536, context_window=65536, max_steps=a.max_steps, request_timeout=300,
        reset_wait=60, evaluate_wait=20, action_wait=2, top_k=6,
        skills_dir=str(native.ROOT / 'Datasets/MMSkills/ubuntu'),
        vm_path='/data/yangshengjie/workspace/VLM-Memory/Benchmark/OSWorld/docker_vm_data/Ubuntu.qcow2',
        embedding_url=a.embedding_url, embedding_timeout=600, retrieval_top_k=10,
        embedding_index_dir=str(native.ROOT / 'Experiments/memory_baselines/artifacts/mmskills_osworld/minicpm_index'))
    a.output.mkdir(parents=True, exist_ok=True)
    spec = {**args, 'condition': 'visual_skill_retrieval', 'task': a.task, 'domain': domain,
            'retrieval_only': a.retrieval_only, 'policy': dense.POLICY,
            'runner_sha256': native.digest(__file__), 'native_runner_sha256': native.digest(native.__file__),
            'dense_retrieval_sha256': native.digest(dense.__file__),
            'embedding_service': dense.service_info(args),
            'retrieval_inputs': 'task text and skill metadata only; no task mapping, screenshot or evaluator',
            'shared_vlm': True, 'concurrency': 1}
    with (a.output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (a.output / 'config.json').exists():
            raise ValueError('Use a fresh output directory to preserve prior smoke results')
        native.write(a.output / 'config.json', spec)
        # This assignment affects only this interpreter; formal run.py and its workers are untouched.
        native.retrieve = retrieve_dense
        if a.retrieval_only:
            config = native.read(native.OSWORLD / 'evaluation_examples/examples' / domain / (a.task + '.json'))
            transport = native.RecordedTransport(args, a.output)
            names, selection = retrieve_dense(config['instruction'], args, transport)
            native.write(a.output / 'selection.json', {**selection, 'selected_skills': names})
        else:
            row = native.worker({'args': args, 'condition': 'visual_skill_retrieval', 'domain': domain, 'task': a.task})
            native.write(a.output / 'summary.json', {'policy': dense.POLICY, 'rows': [row],
                'completed': int(row['status'] == 'completed'), 'errors': int(row['status'] != 'completed')})
            if row['status'] != 'completed':
                raise SystemExit(1)


if __name__ == '__main__':
    main()
