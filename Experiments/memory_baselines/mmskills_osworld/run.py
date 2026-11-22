"""Four-arm MMSkills evaluation; native prompts, isolated OSWorld per process."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / 'Methods/MMSkills_for_Visual_Agents_baseline'
OSWORLD = ROOT / 'Benchmark/OSWorld'
SPLITS = ROOT / 'Experiments/memory_baselines/splits/mmskills_40_seed20261110'
CONDITIONS = ('no_skill', 'text_skill', 'visual_skill', 'visual_skill_retrieval', 'visual_skill_without_images')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(seed=20261110):
    """Proportional stratified random sample, with deterministic largest remainders."""
    mapping = read(BASELINE / 'task_skill_mappings/task_skill_mapping.json')
    total = sum(map(len, mapping.values()))
    counts = {d: math.floor(40 * len(ts) / total) for d, ts in mapping.items()}
    remainder = sorted(mapping, key=lambda d: (-(40 * len(mapping[d]) / total - counts[d]), d))
    for domain in remainder[:40 - sum(counts.values())]:
        counts[domain] += 1
    rng = random.Random(seed)
    selected = {d: sorted(rng.sample(sorted(ts), counts[d])) for d, ts in sorted(mapping.items())}
    # Cover an office workflow, image editing, and media settings. All are in the 40-task sample.
    smoke = {d: [selected[d][0]] for d in ('libreoffice_calc', 'gimp', 'vlc')}
    payloads = {'test_40.json': selected, 'smoke_3.json': smoke,
                'manifest.json': {'seed': seed, 'sampling': 'proportional_stratified_largest_remainder',
                    'pool_tasks': total, 'sample_tasks': 40, 'counts': counts,
                    'mapping_sha256': digest(BASELINE / 'task_skill_mappings/task_skill_mapping.json'),
                    'smoke_selection': 'first sorted sampled ID in Calc/GIMP/VLC; subset, development use'}}
    for name, data in payloads.items():
        path = SPLITS / name
        if path.exists() and read(path) != data:
            raise ValueError(f'Refusing to change frozen split: {path}')
        write(path, data)
    return selected, smoke


class RecordedTransport:
    """Keep native agent prompts; only budget requests and record API interactions."""
    def __init__(self, args, output):
        import openai
        self.args, self.output, self.calls = args, Path(output), []
        self.client = openai.OpenAI(base_url=args['base_url'], api_key='EMPTY', max_retries=0,
                                   timeout=args['request_timeout'])

    def complete(self, **kwargs):
        import requests
        messages = kwargs['messages']
        response = requests.post(self.args['base_url'].removesuffix('/v1') + '/tokenize',
            json={'model': self.args['model'], 'messages': messages, 'add_generation_prompt': True},
            timeout=self.args['request_timeout'])
        response.raise_for_status()
        tokenized = response.json()
        context = min(self.args['context_window'], tokenized['max_model_len'])
        # Reserve room beyond tokenizer count for processor/version differences.
        effective = min(self.args['max_tokens'], context - tokenized['count'] - 256)
        if effective < 1:
            raise ValueError('Input exceeds context window; refusing silent history truncation')
        number = len(self.calls)
        request = {**kwargs, 'max_tokens': effective}
        request.pop('timeout', None)
        write(self.output / f'api/request_{number:03d}.json', request)
        record = {'call': number, 'prompt_tokens_estimate': tokenized['count'],
                  'configured_max_tokens': self.args['max_tokens'], 'effective_max_tokens': effective,
                  'context_window': context, 'status': 'running'}
        self.calls.append(record)
        write(self.output / 'api/calls.json', self.calls)
        start = time.monotonic()
        try:
            result = self.client.chat.completions.create(**request)
            record.update(status='completed', seconds=time.monotonic() - start,
                          usage=result.usage.model_dump() if result.usage else None)
            write(self.output / f'api/response_{number:03d}.json', result.model_dump())
            return result
        except Exception as exc:
            record.update(status='error', seconds=time.monotonic() - start, error=str(exc))
            raise
        finally:
            write(self.output / 'api/calls.json', self.calls)


def bm25_shortlist(instruction, metadata, limit=20):
    """Task text only; no task ID, task domain, evaluator, or task-to-skill mapping."""
    tokenize = lambda s: re.findall(r'[a-z0-9]+', s.lower())
    docs = [tokenize(Path(m.directory).name + ' ' + m.name + ' ' + m.description) for m in metadata]
    freq = collections.Counter(t for doc in docs for t in set(doc))
    avg = sum(map(len, docs)) / max(1, len(docs))
    query = set(tokenize(instruction))
    scores = []
    for m, doc in zip(metadata, docs):
        tf = collections.Counter(doc)
        score = sum(math.log(1 + (len(docs) - freq[t] + .5) / (freq[t] + .5)) *
                    tf[t] * 2.5 / (tf[t] + 1.5 * (.25 + .75 * len(doc) / avg))
                    for t in query if tf[t])
        scores.append((score, Path(m.directory).name, m))
    return sorted(scores, key=lambda x: (-x[0], x[1]))[:limit]


def retrieve(instruction, args, transport):
    from mm_agents.skill_loader import SkillLoader
    metadata = SkillLoader(skills_library_dir=args['skills_dir']).discover_all_skills()
    shortlist = bm25_shortlist(instruction, metadata, args['shortlist_size'])
    candidates = [{'skill': name, 'description': m.description, 'bm25': score} for score, name, m in shortlist]
    result = transport.complete(model=args['model'], temperature=0, top_p=.9,
        messages=[{'role': 'system', 'content':
            'Select reusable desktop skills BEFORE task execution using only the task instruction and '
            'candidate skill descriptions. Select zero to ' + str(args['top_k']) +
            ' directly helpful skills, ordered by relevance. Do not select unrelated skills to fill a quota. '
            'Return ONLY a JSON object with "skills": [exact skill directory names] and "reason": a brief explanation.'},
            {'role': 'user', 'content': json.dumps({'instruction': instruction, 'candidates': candidates})}])
    raw = result.choices[0].message.content or ''
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    decision = json.loads(cleaned)
    names = decision['skills']
    valid = {x['skill'] for x in candidates}
    if not isinstance(names, list) or len(names) > args['top_k'] or len(names) != len(set(names)) or any(n not in valid for n in names):
        raise ValueError('Invalid retrieval selection; refusing mapping fallback')
    return names, {'policy': 'task_only_bm25_shortlist_then_llm_select_v1', 'library_size': len(metadata),
                   'candidates': candidates, 'decision': decision, 'mapping_used': False}


def make_agent(condition, args):
    from mm_agents.general_agent import GeneralAgent
    from mm_agents.general_text_skill_agent import GeneralTextSkillAgent
    from mm_agents.mm_skill_agent import MMSkillAgent
    common = dict(model=args['model'], api_model=args['model'], base_url=args['base_url'], api_key='EMPTY',
                  max_tokens=args['max_tokens'], temperature=0, top_p=.9, action_space='pyautogui',
                  observation_type='screenshot', coordinate_type='relative', max_trajectory_length=3,
                  save_conversation_json=True, llm_request_timeout=args['request_timeout'], llm_max_retries=1)
    if condition == 'no_skill':
        return GeneralAgent(**common)
    if condition == 'text_skill':
        # Public facade delegates reads but not assignments: install transport on its implementation.
        return GeneralTextSkillAgent(**common, skills_library_dir=args['skills_dir'], text_skill_mode='branch_planner')._impl
    if condition == 'visual_skill_without_images':
        from without_images import MMSkillWithoutImagesAgent
        return MMSkillWithoutImagesAgent(**common, skills_library_dir=args['skills_dir'], skill_mode='multimodal')
    return MMSkillAgent(**common, skills_library_dir=args['skills_dir'], skill_mode='multimodal')


def worker(job):
    args, condition, domain, task = job['args'], job['condition'], job['domain'], job['task']
    folder = Path(args['output']) / condition / domain / task
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=folder / 'runtime.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s', force=True)
    import fcntl
    with (folder / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'result.json').exists():
            existing = read(folder / 'result.json')
            if existing['status'] == 'completed':
                return existing
            raise RuntimeError(f'Previous attempt is incomplete/failed: {folder}; use a new run output')
        if (folder / 'api/calls.json').exists():
            raise RuntimeError(f'Incomplete previous attempt: {folder}; use a new run output')
        start, env = time.monotonic(), None
        row = {'condition': condition, 'domain': domain, 'task': task, 'status': 'running'}
        write(folder / 'result.json', row)
        try:
            from desktop_env.desktop_env import DesktopEnv
            from mm_agents import general_agent
            config = read(OSWORLD / 'evaluation_examples/examples' / domain / (task + '.json'))
            transport = RecordedTransport(args, folder)
            agent = make_agent(condition, args)
            # Preserve upstream prompt assembly and response/action parsing exactly.
            def call(system_text, contents):
                response = transport.complete(model=args['model'], temperature=agent.temperature, top_p=agent.top_p,
                    messages=agent._build_openai_messages(system_text, contents),
                    **({'extra_body': agent._build_openai_extra_body()} if agent._build_openai_extra_body() else {}))
                content = agent._extract_openai_message_text(response.choices[0].message)
                if not content:
                    raise ValueError('Empty model response')
                return content
            agent._call_llm_openai = call
            def progress(**event):
                with (folder / 'progress.jsonl').open('a') as stream:
                    stream.write(json.dumps({'time': time.time(), **event}) + '\n')
            agent.set_progress_callback(progress)
            if condition == 'visual_skill_retrieval' or args.get('candidate_policy') == 'retrieval':
                names, selection = retrieve(config['instruction'], args, transport)
            elif condition == 'no_skill':
                names, selection = [], {'policy': 'none', 'mapping_used': False}
            else:
                from mm_agents.task_skill_resolver import resolve_task_skill_names
                names = resolve_task_skill_names(domain=domain, task_id=task,
                    skills_library_dir=args['skills_dir'], mapping_root=str(BASELINE / 'task_skill_mappings/task_skill_mapping.json'),
                    top_k=args['top_k'])
                selection = {'policy': 'author_task_mapping', 'mapping_used': True}
            write(folder / 'selection.json', {**selection, 'selected_skills': names})
            progress(stage='environment_start')
            from evaluation_audit import episode_cache, evaluate_checked
            env = DesktopEnv(provider_name='docker', path_to_vm=args['vm_path'], headless=True,
                             action_space='pyautogui', require_a11y_tree=False, screen_size=(1920, 1080),
                             cache_dir=episode_cache(folder), enable_proxy=False)
            progress(stage='environment_reset')
            obs = env.reset(task_config=config)
            agent.reset(logging.getLogger('mmskills'), vm_ip=env.vm_ip, result_dir=str(folder))
            general_agent.logger = logging.getLogger('mmskills')
            if condition != 'no_skill':
                agent.set_task_skills(names)
            time.sleep(args['reset_wait'])
            obs = dict(env._get_obs())
            obs['env_feedback'] = 'None'
            done, steps, actions_count = False, 0, 0
            # OSWorld has its own lib_run_single.py; load the author's helper explicitly.
            helper_spec = importlib.util.spec_from_file_location('mmskills_single_helper',
                BASELINE / 'osworld_integration/lib_run_single.py')
            helper = importlib.util.module_from_spec(helper_spec)
            helper_spec.loader.exec_module(helper)
            _summarize_env_feedback = helper._summarize_env_feedback
            for step in range(args['max_steps']):
                progress(stage='predict', step=step + 1)
                (folder / f'step_{step:02d}_before.png').write_bytes(obs['screenshot'])
                response, actions = agent.predict(config['instruction'], obs)
                if not actions:
                    raise ValueError('Agent returned no actions')
                for action in actions:
                    prev = obs
                    obs, reward, done, info = env.step(action, args['action_wait'])
                    obs = dict(obs)
                    obs['env_feedback'] = _summarize_env_feedback(action, reward, done, info, prev, obs)
                    actions_count += 1
                    (folder / f'action_{actions_count:03d}_after.png').write_bytes(obs['screenshot'])
                    with (folder / 'traj.jsonl').open('a') as stream:
                        stream.write(json.dumps({'step': step + 1, 'action': action, 'response': response,
                                                 'reward': reward, 'done': done, 'info': info}) + '\n')
                    if done:
                        break
                steps = step + 1
                print(f'{condition}/{domain}/{task} step={steps} done={done}', flush=True)
                if done:
                    break
            time.sleep(args['evaluate_wait'])
            progress(stage='evaluate')
            (folder / 'final_before_evaluation.png').write_bytes(env._get_obs()['screenshot'])
            evaluation = evaluate_checked(env, folder)
            score = evaluation['score']
            if evaluation['status'] == 'valid':
                (folder / 'result.txt').write_text(str(score) + '\n')
            row.update(status='completed' if evaluation['status'] == 'valid' else 'evaluation_error',
                       score=score, raw_score=evaluation['raw_score'], evaluation_status=evaluation['status'],
                       evaluation_issues=evaluation['issues'], env_cache_dir=str(env.cache_dir),
                       termination='agent_terminated' if done else 'step_budget_exhausted',
                       steps=steps, actions=actions_count, agent_done=bool(done),
                       model_calls=len(transport.calls), selected_skills=names,
                       token_usage={key: sum((c.get('usage') or {}).get(key, 0) for c in transport.calls)
                                    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')})
            api_errors = [c for c in transport.calls if c['status'] != 'completed']
            if api_errors:
                row.update(status='error', error='One or more API calls failed; score retained for diagnosis',
                           api_errors=len(api_errors))
        except Exception as exc:
            row.update(status='error', error=str(exc), traceback=traceback.format_exc())
            print(f'ERROR {condition}/{task}: {exc}', flush=True)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    row['cleanup_error'] = traceback.format_exc()
            row['seconds'] = time.monotonic() - start
            write(folder / 'result.json', row)
        return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--suite', choices=('smoke', 'formal'), default='smoke')
    p.add_argument('--task-split', type=Path, help='Explicit domain-to-task-ID JSON; overrides suite selection')
    p.add_argument('--limit-tasks', type=int, help='Limit the number of tasks for a diagnostic run')
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS[:4]))
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--output', type=Path)
    p.add_argument('--base-url', default='http://127.0.0.1:8166/v1')
    p.add_argument('--model', default='qwen3-vl-8b-instruct')
    p.add_argument('--max-tokens', type=int, default=65536)
    p.add_argument('--context-window', type=int, default=65536)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--request-timeout', type=int, default=300)
    p.add_argument('--reset-wait', type=float, default=60)
    p.add_argument('--evaluate-wait', type=float, default=20)
    p.add_argument('--action-wait', type=float, default=0)
    p.add_argument('--top-k', type=int, default=6)
    p.add_argument('--shortlist-size', type=int, default=20)
    p.add_argument('--skills-dir', default=str(ROOT / 'Datasets/MMSkills/ubuntu'))
    # Docker daemon runs on the host; this workspace's /root is a bind mount.
    # Use the same host-side qcow2 path as the existing gui_baseline framework.
    p.add_argument('--vm-path', default='/data/yangshengjie/workspace/VLM-Memory/Benchmark/OSWorld/docker_vm_data/Ubuntu.qcow2')
    a = p.parse_args()
    selected, smoke = prepare() if a.task_split is None else (None, None)
    if a.prepare_only:
        print(SPLITS)
        return
    if not a.output:
        p.error('--output required for execution')
    a.max_steps = a.max_steps or (5 if a.suite == 'smoke' else 20)
    if min(a.workers, a.max_steps, a.max_tokens, a.top_k, a.shortlist_size) < 1:
        p.error('Budgets and worker count must be positive')
    tasks = read(a.task_split) if a.task_split else (smoke if a.suite == 'smoke' else selected)
    if not isinstance(tasks, dict) or not tasks or any(
        not isinstance(ids, list) or any(not isinstance(t, str) for t in ids)
        or len(ids) != len(set(ids)) for ids in tasks.values()
    ) or not any(tasks.values()):
        p.error('Task split must contain nonempty domain-to-unique-task-ID lists')
    if a.limit_tasks is not None:
        if a.limit_tasks < 1:
            p.error('--limit-tasks must be positive')
        remaining = a.limit_tasks
        limited = {}
        for domain, ids in tasks.items():
            if remaining <= 0:
                break
            limited[domain] = ids[:remaining]
            remaining -= len(limited[domain])
        tasks = limited
    args = json.loads(json.dumps(vars(a), default=str))
    args['output'] = str(a.output.resolve())
    spec = {**args, 'tasks': tasks,
            'runner_sha256': digest(__file__),
            'evaluation_audit_sha256': digest(Path(__file__).with_name('evaluation_audit.py')),
            'without_images_sha256': digest(Path(__file__).with_name('without_images.py')),
            'mapping_sha256': digest(BASELINE / 'task_skill_mappings/task_skill_mapping.json'),
            'prompt_source_sha256': {str(f.relative_to(BASELINE)): digest(f) for f in (BASELINE / 'mm_agents').rglob('*.py')},
            'retrieval': 'task_only_bm25_shortlist_then_llm_select_v1; local implementation of abstract PreRecall',
            'budget_policy': 'min(configured_max_tokens, context_window - tokenize_count - 256)'}
    a.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (a.output / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (a.output / 'config.json').exists() and read(a.output / 'config.json') != spec:
            raise ValueError('Output config differs; use a fresh directory')
        write(a.output / 'config.json', spec)
        jobs = [dict(args=args, condition=c, domain=d, task=t)
                for d, ts in tasks.items() for t in ts for c in a.conditions]
        rows = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                write(a.output / 'progress.json', {'planned': len(jobs), 'finished': len(rows), 'rows': rows})
        summary = {'planned': len(jobs), 'completed': sum(r['status'] == 'completed' for r in rows),
                   'errors': sum(r['status'] != 'completed' for r in rows), 'conditions': {}, 'rows': rows}
        for c in a.conditions:
            group = [r for r in rows if r['condition'] == c]
            valid = [r for r in group if r['status'] == 'completed']
            summary['conditions'][c] = {'planned': len(group), 'completed': len(valid),
                'mean_score_completed': sum(r['score'] for r in valid) / len(valid) if valid else None,
                'mean_score_full_suite': sum(r['score'] for r in valid) / len(group) if len(valid) == len(group) else None}
        write(a.output / 'summary.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
        if summary['errors']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
