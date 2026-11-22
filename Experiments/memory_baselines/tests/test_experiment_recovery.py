import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vlm_memory.config import write_json
from vlm_memory.models.local import image_hash
from vlm_memory.pipeline.formal_100_100 import prepare, prepare_evaluation_targets
from vlm_memory.pipeline.pilot import call, evaluate
from vlm_memory.pipeline.run_100_100_nomemory import snapshot_run


class RecoveryTests(unittest.TestCase):
    def test_document_references_are_copied_only_to_the_scorer_and_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,destination=Path(tmp)/'source',Path(tmp)/'formal'
            manifest=dict(memory=[dict(id='train')],test=[dict(id='heldout')])
            reference=dict(answers=['PRIVATE_REFERENCE'],answer_page_idx=1)
            write_json(source/'evaluation_targets/heldout.json',reference)
            write_json(source/'evaluation_targets/train.json',reference)
            prepare_evaluation_targets(source,destination,manifest)
            self.assertEqual(json.loads((destination/'evaluation_targets/heldout.json').read_text()),reference)
            self.assertFalse((destination/'evaluation_targets/train.json').exists())
            self.assertFalse((destination/'histories').exists())
            self.assertNotIn('PRIVATE_REFERENCE',json.dumps(manifest))
            write_json(destination/'evaluation_targets/heldout.json',dict(answers=['changed']))
            with self.assertRaisesRegex(ValueError,'scoring reference changed'):
                prepare_evaluation_targets(source,destination,manifest)

    def test_memory_only_matrix_does_not_count_reused_nomemory_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for method in ['no_memory','magnet_adapted']:
                write_json(root/'runs'/method/'query/seed_0/attempt_0/result.json',dict(status='completed',method=method))
            manifest=dict(benchmark='vision2code',memory=[dict(id='source')],test=[dict(id='query',image_hashes=[])])
            bank=Mock()
            bank.data=dict(sources=[dict(id='source',benchmark='vision2code',image_hashes=[])])
            bank.verify_frozen.return_value=True
            with patch('vlm_memory.pipeline.pilot.MemoryBank',return_value=bank):
                evaluate(root,manifest,dict(baselines=['magnet_adapted']),True)
            self.assertEqual(len(json.loads((root/'summary.json').read_text())),1)
            validation=json.loads((root/'validation.json').read_text())
            self.assertEqual(validation['completed_results'],validation['expected_results'])

    def test_timeout_can_change_without_changing_frozen_sampling_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"model": {"name": "local", "timeout": 600, "max_tokens": 32678}}
            snapshot_run(root, config)
            config["model"]["timeout"] = 1800
            snapshot_run(root, config)
            self.assertEqual(json.loads((root / "config.snapshot.json").read_text())["model"]["timeout"], 600)
            self.assertEqual(json.loads((root / "runtime_transport.json").read_text())["timeout"], 1800)
            config["model"]["max_tokens"] = 8192
            with self.assertRaisesRegex(ValueError, "configuration/prompt changed"):
                snapshot_run(root, config)

    def test_preserved_generation_is_reused_and_changed_budget_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            response = {"system": "system", "prompt": "question", "images": [],
                        "requested_max_tokens": 4096, "text": "answer"}
            first = SimpleNamespace(config={"max_tokens": 4096}, generate=Mock(return_value=response))
            call(first, folder, "generation", "system", "question", [])
            second = SimpleNamespace(config={"max_tokens": 4096}, generate=Mock())
            self.assertEqual(call(second, folder, "generation", "system", "question", []), response)
            second.generate.assert_not_called()
            second.config["max_tokens"] = 8192
            with self.assertRaisesRegex(ValueError, "request differs"):
                call(second, folder, "generation", "system", "question", [])

    def test_formal_import_uses_memory_episodes_and_keeps_test_out_of_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / "common", root / "formal"
            config = {"prompt_version": "base_v1", "model": {"name": "qwen3.8-27b", "timeout": 600}}
            write_json(source / "config.snapshot.json", config)
            config["model"]["timeout"] = 1800
            image = root / "source-image.png"
            image.write_bytes(b"image fixture")
            memory = {"id": "train", "benchmark": "mathvision"}
            test = {"id": "test", "benchmark": "mathvision"}
            manifest = {"memory": [memory], "test": [test]}
            write_json(source / "dataset_manifest.json", manifest)
            for split, task_id in (("memory", "train"), ("test", "test")):
                write_json(source / "runs/no_memory" / split / task_id / "attempt_0/result.json",
                           {"status": "completed"})
            episode = {"id": "train", "split": "memory", "prompt_version": "base_v1",
                       "images": [{"path": str(image), "sha256": image_hash(image), "role": "source"}],
                       "provenance": {}}
            write_json(source / "trajectories/memory/train.json", episode)
            with patch("vlm_memory.pipeline.formal_100_100.verify_manifest", return_value=manifest):
                prepare(source, destination, "mathvision", config)
            self.assertTrue((destination / "histories/train/history.jsonl").exists())
            self.assertFalse((destination / "histories/test").exists())
            self.assertTrue((destination / "runs/no_memory/test/seed_0/attempt_0/result.json").exists())
            metadata = json.loads((destination / "common_trajectory_provenance.json").read_text())
            self.assertEqual(metadata["memory_source_ids"], ["train"])
            self.assertEqual(metadata["test_source_ids"], [])

    def test_formal_import_still_rejects_changed_output_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            write_json(source / "config.snapshot.json", {"model": {"max_tokens": 32678, "timeout": 600}})
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                prepare(source, Path(tmp) / "destination", "mathvision",
                        {"model": {"max_tokens": 8192, "timeout": 1800}})


if __name__ == "__main__":
    unittest.main()
