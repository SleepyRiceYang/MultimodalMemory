import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vlm_memory.config import load_config
from vlm_memory.memory.bank import MemoryBank, inside
from vlm_memory.memory.build import build_memories
from vlm_memory.models.local import image_hash


class MemoryContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = self.root / "history.png"
        self.current = self.root / "current.png"
        Image.new("RGB", (24, 24), "blue").save(self.reference)
        Image.new("RGB", (24, 24), "red").save(self.current)
        entry = {"id": "entry", "benchmark": "mathvision", "source_ids": ["history_1"],
                 "index_text": "count objects", "text": "PRIVATE_HISTORICAL_ANSWER_17",
                 "images": [{"path": "history.png", "role": "before"}]}
        self.data = {"sources": [{"id": "history_1", "benchmark": "mathvision",
                                  "image_hashes": [image_hash(self.reference)]}],
                     "methods": {"raw_text": [{**entry, "images": []}], "raw_visual": [entry],
                                 "raw_multimodal": [entry]}}
        self.budget = load_config()["memory"]

    def tearDown(self):
        self.temp.cleanup()

    def bank(self):
        path = self.root / "bank.json"
        path.write_text(json.dumps(self.data))
        return MemoryBank(path)

    def context(self, method, **kw):
        return self.bank().context(method, "mathvision", kw.get("task_id", "eval_1"),
                                   "count objects", kw.get("image", self.current), self.budget)

    def test_text_and_visual_controls_have_distinct_payloads(self):
        text = self.context("raw_text")
        visual = self.context("raw_visual")
        multimodal = self.context("raw_multimodal")
        self.assertIn("PRIVATE_HISTORICAL_ANSWER", text["text"])
        self.assertFalse(text["images"])
        self.assertNotIn("PRIVATE_HISTORICAL_ANSWER", visual["text"])
        self.assertEqual(len(visual["images"]), 1)
        self.assertTrue(multimodal["images"])
        self.assertIn("PRIVATE_HISTORICAL_ANSWER", multimodal["text"])

    def test_exact_image_and_source_id_leakage_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "task ID"):
            self.context("raw_text", task_id="history_1")
        with self.assertRaisesRegex(ValueError, "image"):
            self.context("raw_text", image=self.reference)

    def test_no_memory_retains_no_historical_payload(self):
        context = self.context("no_memory")
        self.assertFalse(context["text"])
        self.assertFalse(context["images"])
        self.assertFalse(context["selected"])

    def test_budget_excludes_whole_entries_and_images(self):
        self.budget["max_text_chars"] = 0
        self.budget["max_images"] = 0
        context = self.context("raw_multimodal")
        self.assertFalse(context["selected"])
        self.assertFalse(context["images"])
        self.assertTrue(context["truncated"])

    def test_assets_cannot_escape_memory_root(self):
        with self.assertRaises(ValueError):
            inside(self.root, "../outside.png")

    def test_freeze_detects_changed_skill_or_image_asset(self):
        self.data["asset_hashes"] = {"history.png": image_hash(self.reference)}
        bank = self.bank()
        Image.new("RGB", (24, 24), "green").save(self.reference)
        with self.assertRaisesRegex(ValueError, "asset changed"):
            bank.verify_frozen()

    def test_darwinian_risk_filter_uses_observed_outcomes(self):
        base = self.data["methods"]["raw_multimodal"][0]
        self.data["methods"]["darwinian_adapted"] = [
            {**base, "id": "reliable", "statistics": {"successes": 3, "failures": 0, "uses": 3, "steps": 2}},
            {**base, "id": "risky", "statistics": {"successes": 0, "failures": 12, "uses": 12, "steps": 1}},
        ]
        context = self.context("darwinian_adapted")
        self.assertEqual([entry["id"] for entry in context["selected"]], ["reliable"])

    def test_raw_only_build_exports_provenance_without_model_calls(self):
        episode = {"id": "history_1", "benchmark": "mathvision", "task": "count objects",
                   "provenance": "independent_test", "images": [{"path": "history.png", "role": "before"}],
                   "steps": [{"text": "observed answer"}], "feedback": {"correct": True}, "success": True}
        (self.root / "history.jsonl").write_text(json.dumps(episode) + "\n")
        config = load_config()
        config.update(baselines=["raw_text"], benchmarks=["mathvision"])
        with patch("vlm_memory.memory.build.LocalModel") as model:
            output = build_memories(self.root, config)
            model.return_value.generate.assert_not_called()
        bank = MemoryBank(output)
        self.assertEqual(set(bank.data["methods"]), {"raw_text"})
        self.assertEqual(bank.data["sources"][0]["id"], "history_1")
        self.assertTrue(bank.data["asset_hashes"])
        self.assertFalse(bank.data["methods"]["raw_text"][0]["images"])


if __name__ == "__main__":
    unittest.main()
