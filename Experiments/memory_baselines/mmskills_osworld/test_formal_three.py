import json
from pathlib import Path
import sys
import tempfile
import unittest

from formal_three import snapshot, supervise


class MonitoringTests(unittest.TestCase):
    def test_error_and_incomplete_are_not_valid_completed_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for condition, status in [('no_skill', 'completed'), ('text_skill', 'evaluation_error'), ('visual_skill', 'running')]:
                p = root / condition / 'domain' / 'task'
                p.mkdir(parents=True)
                (p / 'result.json').write_text(json.dumps({'status': status, 'score': 0 if status == 'completed' else None}))
            row = snapshot(root, {'domain': ['task']}, 'running')
            self.assertEqual((row['planned'], row['completed'], row['errors']), (3, 1, 1))
            self.assertEqual(len(row['active']), 1)

    def test_periodic_logs_and_exit_status_without_starting_osworld(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = supervise([sys.executable, '-c', 'import time; print("fixture", flush=True); time.sleep(.12)'],
                             root, {'domain': ['task']}, interval=.03)
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in (root / 'progress.log').read_text().splitlines()]
            self.assertEqual(rows[0]['phase'], 'started')
            self.assertIn('running', [r['phase'] for r in rows])
            self.assertEqual(rows[-1]['phase'], 'completed')
            self.assertIn('fixture', (root / 'run.log').read_text())


if __name__ == '__main__':
    unittest.main()
