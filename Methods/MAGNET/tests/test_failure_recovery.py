"""Regression checks for malformed GUI actions and interrupted fixture downloads."""
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from types import SimpleNamespace
import unittest

from gui_baseline.runner import Android


class RecoveryTests(unittest.TestCase):
    def test_missing_action_type_records_error_and_continues(self):
        adapter = Android.__new__(Android)
        adapter.args = SimpleNamespace(max_steps=2, max_tokens=128)
        replies = iter([{'reason': 'Malformed action'},
                        {'action_type': 'status', 'goal_status': 'complete'}])
        adapter.model = SimpleNamespace(generate=lambda *a: next(replies))
        adapter.screenshot = lambda path: (1, 1)
        adapter.api = lambda *a, **kw: {'score': 1.0}
        adapter.params = {}
        with TemporaryDirectory() as folder:
            result = adapter.run(Path(folder), 'goal', {'text': '', 'images': []})
            self.assertTrue((Path(folder) / 'step_000/action_error.json').exists())
            self.assertTrue(result['agent_done'])
            self.assertEqual(result['steps'], 2)

    def test_interrupted_download_retries_and_caches_complete_file(self):
        source = Path(__file__).resolve().parents[3] / 'Benchmark/OSWorld/desktop_env/evaluators/getters/file.py'
        spec = importlib.util.spec_from_file_location('fixture_getters', source)
        getters = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(getters)
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append(self.path)
                self.send_response(200)
                self.send_header('Content-Length', '8')
                self.end_headers()
                self.wfile.write(b'bad' if len(calls) == 1 else b'complete')
                self.close_connection = True

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as folder:
                env = SimpleNamespace(cache_dir=folder)
                config = {'path': f'http://127.0.0.1:{server.server_port}/fixture', 'dest': 'gold.bin'}
                path = getters.get_cloud_file(env, config)
                self.assertEqual(Path(path).read_bytes(), b'complete')
                self.assertEqual(getters.get_cloud_file(env, config), path)
                self.assertEqual(len(calls), 2)
                self.assertEqual(list(Path(folder).glob('*.tmp.*')), [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
