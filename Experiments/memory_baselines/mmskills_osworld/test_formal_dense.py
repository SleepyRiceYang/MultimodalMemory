import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import formal_dense


class FormalDenseTests(unittest.TestCase):
    def test_spawn_worker_installs_dense_retriever(self):
        previous = formal_dense.native.retrieve
        try:
            with patch.object(formal_dense.native, 'worker', return_value={'status': 'completed'}) as worker:
                job = {'task': 'fixture'}
                formal_dense.dense_worker(job)
                self.assertIs(formal_dense.native.retrieve, formal_dense.retrieve_dense)
                worker.assert_called_once_with(job)
        finally:
            formal_dense.native.retrieve = previous

    def test_monitor_reports_only_the_fourth_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'visual_skill_retrieval/domain/task'
            folder.mkdir(parents=True)
            (folder / 'result.json').write_text(json.dumps({'status': 'completed', 'score': 1}))
            with patch.object(formal_dense.monitoring, 'CONDITIONS', ('visual_skill_retrieval',)):
                report = formal_dense.monitoring.snapshot(root, {'domain': ['task']}, 'running')
            self.assertEqual(report['planned'], 1)
            self.assertEqual(report['completed'], 1)
            self.assertEqual(set(report['conditions']), {'visual_skill_retrieval'})


if __name__ == '__main__':
    unittest.main()
