"""Five screenshot-only no-memory tasks through AndroidWorld's official HTTP API."""
from __future__ import annotations

import argparse
import base64
import fcntl
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
from openai import OpenAI
from PIL import Image

from ..config import ROOT, write_json
from ..prompts.gui_tasks import ANDROID_SYSTEM


def validate_action(value, width, height):
    value = dict(value)
    value.pop('reason', None)
    action = value.get('action_type')
    allowed = {'click': {'x', 'y'}, 'long_press': {'x', 'y'}, 'input_text': {'text'},
               'keyboard_enter': set(), 'scroll': {'direction'}, 'navigate_back': set(),
               'navigate_home': set(), 'open_app': {'app_name'}, 'wait': set(),
               'status': {'goal_status'}}
    if action not in allowed or set(value) != {'action_type', *allowed[action]}:
        raise ValueError(f'Unexpected action fields: {value}')
    if action in ('click', 'long_press'):
        for key, size in [('x', width), ('y', height)]:
            if type(value[key]) not in (int, float) or not 0 <= value[key] <= 1000:
                raise ValueError('Coordinates must be normalized to [0,1000]')
            value[key] = min(size - 1, round(value[key] * size / 1000))
    if action == 'scroll' and value['direction'] not in ('up', 'down', 'left', 'right'):
        raise ValueError('Invalid scroll direction')
    if action == 'status' and value['goal_status'] not in ('complete', 'infeasible'):
        raise ValueError('Invalid status')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='qwen3_vl_8b_androidworld_nomemory_5_20261026_png_v1')
    parser.add_argument('--base-url', default='http://127.0.0.1:8104/v1')
    parser.add_argument('--env-url', default='http://127.0.0.1:15000')
    parser.add_argument('--model', default='qwen3-vl-8b-instruct')
    parser.add_argument('--max-steps', type=int, default=30)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--screenshot-backend', choices=['adb_png', 'http'], default='adb_png')
    parser.add_argument('--android-container', default='android_world_smoke')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'artifacts/androidworld' / args.run
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env_lock = (ROOT / 'artifacts/locks/androidworld_http.lock').open('w')
    fcntl.flock(env_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    split = json.loads((ROOT / 'splits/gui_smoke_5_seed20261026/androidworld.json').read_text())
    config = {**{k:v for k,v in vars(args).items() if k != 'retry_failed'}, 'split': split, 'system': ANDROID_SYSTEM,
              'baseline': 'no_memory', 'coordinate_type': 'normalized_1000',
              'history': 'last_10_actions_text; current_screenshot_only',
              'settle_seconds':2, 'scoring':'TaskEval.is_successful gated by explicit agent termination'}
    snapshot = root / 'config.snapshot.json'
    if snapshot.exists() and json.loads(snapshot.read_text()) != config:
        raise ValueError('Configuration changed; use a new run')
    write_json(snapshot, config)
    http = requests.Session()
    http.trust_env = False

    def api(method, route, **kwargs):
        r = http.request(method, args.env_url + route, timeout=180, **kwargs)
        r.raise_for_status()
        return r.json()

    def screenshot():
        if args.screenshot_backend == 'adb_png':
            capture = subprocess.run(['docker', 'exec', args.android_container, 'adb',
                                      'exec-out', 'screencap', '-p'],
                                     capture_output=True, check=True, timeout=30)
            return Image.open(io.BytesIO(capture.stdout)).convert('RGB')
        pixels = api('GET', '/screenshot', params={'wait_to_stabilize':True})['pixels']
        return Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert('RGB')

    api('GET', '/health')
    api('GET', '/suite/reinitialize', params={'seed': split['seed'], 'n_task_combinations': 1})
    client = OpenAI(base_url=args.base_url, api_key='EMPTY', timeout=180, max_retries=1)
    started = time.monotonic()
    results = []
    for index, task in enumerate(split['tasks'], 1):
        folder = root / task
        result_path = folder / 'result.json'
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous['status'] == 'completed' or not args.retry_failed:
                results.append(previous)
                continue
            archive = folder / 'errors' / f'{time.time_ns()}.json'
            archive.parent.mkdir(exist_ok=True)
            result_path.rename(archive)
        folder.mkdir(exist_ok=True)
        params = {'task_type': task, 'task_idx': split['task_index']}
        initialized = False
        task_started = time.monotonic()
        history = []
        predicted_done = False
        agent_done = False
        print(f'START AndroidWorld {task} {index}/5', flush=True)
        try:
            api('POST', '/reset', params={'go_home': True})
            api('POST', '/task/initialize', params=params)
            initialized = True
            goal = api('GET', '/task/goal', params=params)['goal']
            write_json(folder / 'task.json', {'task_type': task, 'task_index': 0, 'goal': goal, 'seed': split['seed']})
            for step in range(args.max_steps):
                step_dir = folder / f'step_{step:03d}'
                step_dir.mkdir(exist_ok=True)
                im = screenshot()
                width, height = im.size
                im.save(step_dir / 'screen.png')
                im.thumbnail((1280, 1280))
                buffer = io.BytesIO()
                im.save(buffer, format='PNG')
                payload = {'task': goal, 'previous_actions': history[-10:], 'step': step + 1,
                           'steps_remaining': args.max_steps - step, 'coordinates': 'normalized 0..1000'}
                write_json(step_dir / 'request.json', {'system': ANDROID_SYSTEM, 'input': payload,
                           'image': str(step_dir / 'screen.png'), 'model': args.model, 'max_tokens': args.max_tokens})
                response = client.chat.completions.create(model=args.model, temperature=0,
                    max_tokens=args.max_tokens, response_format={'type': 'json_object'},
                    messages=[{'role':'system', 'content':ANDROID_SYSTEM},
                              {'role':'user', 'content':[{'type':'text', 'text':json.dumps(payload)},
                               {'type':'image_url', 'image_url':{'url':'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}}]}])
                write_json(step_dir / 'response.json', response.model_dump())
                try:
                    if response.choices[0].finish_reason == 'length':
                        raise ValueError('Action generation truncated')
                    value = json.loads(response.choices[0].message.content)
                    action = validate_action(value, width, height)
                    write_json(step_dir / 'action.json', action)
                    print(f'ACTION {task} {step + 1}/{args.max_steps}: {action}', flush=True)
                    if action['action_type'] == 'status':
                        agent_done = True
                        predicted_done = action['goal_status'] == 'complete'
                        history.append({'action':value, 'outcome':'agent_terminated'})
                        break
                    outcome = api('POST', '/execute_action', json=action)
                    history.append({'action':value, 'outcome':outcome})
                    time.sleep(config['settle_seconds'])
                except (ValueError, requests.RequestException) as exc:
                    write_json(step_dir / 'action_error.json', {'error':str(exc)})
                    history.append({'error':str(exc)})
                    print(f'ACTION_ERROR {task}: {exc}', flush=True)
            score = api('GET', '/task/score', params=params)['score']
            result = {'status':'completed', 'task_id':task, 'goal':goal,
                      'task_state_score':score, 'score':score if agent_done else 0.0,
                      'success':score == 1.0 and agent_done, 'predicted_done':predicted_done,
                      'agent_done':agent_done,
                      'steps':len(history), 'seconds':time.monotonic() - task_started,
                      'evaluator':'AndroidWorld TaskEval.is_successful', 'baseline':'no_memory'}
            screenshot().save(folder / 'final.png')
        except Exception as exc:
            result = {'status':'error', 'task_id':task, 'error':f'{type(exc).__name__}: {exc}',
                      'steps':len(history), 'seconds':time.monotonic() - task_started}
            print(f'TASK_ERROR {task}: {result["error"]}', flush=True)
        finally:
            if initialized:
                try:
                    api('POST', '/task/tear_down', params=params)
                except Exception as exc:
                    write_json(folder / 'cleanup_error.json', {'error':str(exc)})
        write_json(result_path, result)
        results.append(result)
        write_json(root / 'summary.json', results)
        elapsed = time.monotonic() - started
        print(f'PROGRESS AndroidWorld {index}/5 score={result.get("score")} elapsed={elapsed:.0f}s ETA={elapsed/index*(5-index):.0f}s', flush=True)
    write_json(root / 'summary.json', results)


if __name__ == '__main__':
    main()
