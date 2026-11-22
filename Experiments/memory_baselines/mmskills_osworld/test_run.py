"""Check split isolation, task-only retrieval, and the context budget boundary."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('mmskills_run', Path(__file__).with_name('run.py'))
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
from evaluation_audit import episode_cache, evaluate_checked


class ProtocolTests(unittest.TestCase):
    def test_cache_paths_are_isolated_by_episode_and_condition(self):
        self.assertNotEqual(episode_cache('/tmp/run/text_skill/gimp/task'),
                            episode_cache('/tmp/run/visual_skill/gimp/task'))

    def test_missing_evaluation_input_is_not_a_valid_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            env = types.SimpleNamespace(controller=types.SimpleNamespace(), cache_dir=temp,
                evaluator={'func': 'check_config_status'}, result_getter=lambda *_: None, expected_getter=None)
            def evaluate():
                return 0 if env.result_getter(env, {}) is None else 1
            env.evaluate = evaluate
            result = evaluate_checked(env, temp)
            self.assertEqual(result['status'], 'evaluation_error')
            self.assertEqual(result['raw_score'], 0)
            self.assertIsNone(result['score'])

    def test_valid_failure_stays_zero_and_evidence_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / 'config'
            artifact.write_text('(theme "Dark")')
            getter = lambda *_: str(artifact)
            env = types.SimpleNamespace(controller=types.SimpleNamespace(), cache_dir=temp,
                evaluator={'func': 'check_config_status'}, result_getter=getter, expected_getter=None)
            env.evaluate = lambda: 0 if Path(env.result_getter(env, {})).read_text() else 1
            result = evaluate_checked(env, temp)
            self.assertEqual(result['status'], 'valid')
            self.assertEqual(result['score'], 0)
            self.assertEqual(result['getters'][0]['files'][0]['bytes'], artifact.stat().st_size)
            self.assertIs(env.result_getter, getter)

    def test_swallowed_getter_exception_still_invalidates_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            def getter(*_):
                raise FileNotFoundError('missing')
            env = types.SimpleNamespace(controller=types.SimpleNamespace(), cache_dir=temp,
                evaluator={}, result_getter=getter, expected_getter=None)
            def evaluate():
                try:
                    env.result_getter(env, {})
                except FileNotFoundError:
                    return 0
            env.evaluate = evaluate
            result = evaluate_checked(env, temp)
            self.assertEqual(result['status'], 'evaluation_error')
            self.assertEqual(result['issues'][0]['type'], 'getter_exception')

    def test_transport_hook_reaches_all_native_agent_implementations(self):
        sys.path.insert(0, str(run.OSWORLD))
        args = {'model': 'qwen3-vl-8b-instruct', 'base_url': 'http://localhost:8166/v1',
                'max_tokens': 65536, 'request_timeout': 300,
                'skills_dir': str(run.ROOT / 'Datasets/MMSkills/ubuntu')}
        for condition in run.CONDITIONS:
            with self.subTest(condition=condition):
                agent = run.make_agent(condition, args)
                agent._call_llm_openai = lambda **kwargs: 'transport_hook_reached'
                self.assertEqual(agent.call_llm(system_text='test', contents=[]), 'transport_hook_reached')

    def test_stratified_split_is_fixed_subset_of_author_mapping(self):
        selected, smoke = run.prepare()
        mapping = run.read(run.BASELINE / 'task_skill_mappings/task_skill_mapping.json')
        self.assertEqual(sum(map(len, selected.values())), 40)
        self.assertEqual(len(selected), 10)
        self.assertEqual(sum(map(len, smoke.values())), 3)
        for domain, tasks in selected.items():
            self.assertEqual(len(set(tasks)), len(tasks))
            self.assertTrue(set(tasks) <= set(mapping[domain]))
            self.assertTrue(set(smoke.get(domain, [])) <= set(tasks))
        self.assertEqual((selected, smoke), run.prepare())

    def test_task_only_shortlist_finds_matching_procedure_across_library(self):
        meta = lambda n, d: types.SimpleNamespace(directory='/skills/' + n, name=n, description=d)
        library = [meta('audio', 'VLC global hotkey pause video'), meta('image', 'GIMP theme light dark'),
                   meta('sheet', 'Calc spreadsheet merge cells header')]
        self.assertEqual(run.bm25_shortlist('Merge spreadsheet header cells', library, 1)[0][1], 'sheet')
        self.assertEqual(run.bm25_shortlist('GIMP light theme', library, 1)[0][1], 'image')

    def test_configured_65536_is_bounded_by_remaining_context(self):
        with tempfile.TemporaryDirectory() as temp:
            transport = object.__new__(run.RecordedTransport)
            transport.args = {'base_url': 'http://local/v1', 'model': 'test', 'context_window': 65536,
                              'max_tokens': 65536, 'request_timeout': 1}
            transport.output, transport.calls = Path(temp), []
            captured = {}
            result = types.SimpleNamespace(usage=None, model_dump=lambda: {'ok': True})
            def create(**kwargs):
                captured.update(kwargs)
                return result
            transport.client = types.SimpleNamespace(chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)))
            tokenized = types.SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {'max_model_len': 65536, 'count': 4096})
            with patch('requests.post', return_value=tokenized):
                transport.complete(model='test', messages=[{'role': 'user', 'content': 'input'}])
            self.assertEqual(captured['max_tokens'], 65536 - 4096 - 256)
            self.assertEqual(transport.calls[0]['configured_max_tokens'], 65536)


if __name__ == '__main__':
    unittest.main()
