"""Observe official OSWorld evaluation without changing task scoring rules."""
import hashlib
import json
import math
from pathlib import Path


def episode_cache(folder):
    return str((Path(folder) / 'env_cache').resolve())


def evaluate_checked(env, folder):
    folder = Path(folder)
    issues, getters, transfers = [], [], []
    report = {'status': 'running', 'raw_score': None, 'score': None,
              'cache_dir': str(env.cache_dir), 'issues': issues, 'getters': getters,
              'file_transfers': transfers, 'evaluator': env.evaluator}
    originals = {}
    original_get_file = getattr(env.controller, 'get_file', None)
    def get_file(path, *args, **kwargs):
        record = {'guest_path': str(path)}
        transfers.append(record)
        try:
            content = original_get_file(path, *args, **kwargs)
            record.update(returned_none=content is None, bytes=len(content) if content is not None else None)
            return content
        except Exception as exc:
            record['error'] = repr(exc)
            raise
    def files_in(value):
        if isinstance(value, (list, tuple)):
            return [f for x in value for f in files_in(x)]
        if isinstance(value, dict):
            return [f for x in value.values() for f in files_in(x)]
        if isinstance(value, str) and len(value) < 4096:
            path = Path(value)
            if path.is_file():
                return [{'path': str(path.resolve()), 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
        return []
    def wrap(getter, role, index=None):
        def observed(*args, **kwargs):
            record = {'role': role, 'index': index, 'getter': getattr(getter, '__name__', str(getter))}
            getters.append(record)
            try:
                value = getter(*args, **kwargs)
                record.update(returned_none=value is None, files=files_in(value))
                if value is None:
                    issues.append({'type': 'getter_input_unavailable', 'role': role, 'index': index})
                return value
            except Exception as exc:
                record['error'] = repr(exc)
                issues.append({'type': 'getter_exception', 'role': role, 'index': index, 'error': repr(exc)})
                raise
        return observed
    try:
        for role in ('result_getter', 'expected_getter'):
            value = getattr(env, role, None)
            if not value:
                continue
            originals[role] = value
            setattr(env, role, [wrap(f, role, i) for i, f in enumerate(value)]
                    if isinstance(value, list) else wrap(value, role))
        if callable(original_get_file):
            env.controller.get_file = get_file
        raw = float(env.evaluate())
        report['raw_score'] = raw
        if not math.isfinite(raw) or not 0 <= raw <= 1:
            issues.append({'type': 'invalid_metric_value', 'value': str(raw)})
        report['status'] = 'evaluation_error' if issues else 'valid'
        report['score'] = raw if not issues else None
    except Exception as exc:
        issues.append({'type': 'evaluation_exception', 'error': repr(exc)})
        report.update(status='evaluation_error', error=repr(exc))
    finally:
        for role, value in originals.items():
            setattr(env, role, value)
        if callable(original_get_file):
            env.controller.get_file = original_get_file
        (folder / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
    return report
