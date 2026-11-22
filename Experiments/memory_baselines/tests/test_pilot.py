import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vlm_memory.pipeline.data import math_task, select
from vlm_memory.pipeline.pilot import document_schema


class PilotProtocolTests(unittest.TestCase):
    def test_math_public_input_excludes_reference_answer_and_solution(self):
        task = math_task({"id": "x", "question": "count", "options": [], "answer": "SECRET_GOLD",
                          "solution": "SECRET_SOLUTION", "subject": "counting", "level": 1, "image": "x.png"})
        self.assertNotIn("SECRET", json.dumps(task))
        self.assertNotIn("answer", task)

    def test_document_schema_constrains_not_found_and_page_indices(self):
        found, missing = document_schema([2, 3])["anyOf"]
        self.assertEqual(found["properties"]["answer_page_idx"]["enum"], [2, 3])
        self.assertEqual(missing["properties"]["answer"]["const"], "")
        self.assertEqual(missing["properties"]["answer_page_idx"]["type"], "null")

    def test_document_split_excludes_same_document_and_keeps_gold_in_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "Benchmark/MP-DocVQA-Framework/data/MP-DocVQA/images"
            images.mkdir(parents=True)
            def make(i, doc):
                path = images / f"{doc}_p0.jpg"
                if not path.exists():
                    Image.new("RGB", (16, 16), (i * 12, 20, 40)).save(path)
                return {"questionId": i, "question": "question", "doc_id": doc, "page_ids": [f"{doc}_p0"],
                        "answers": ["SECRET_GOLD"], "answer_page_idx": 0}
            train = [make(i, f"doc{i}") for i in range(1, 12)]
            val = [make(40, "doc1")] + [make(i, f"doc{i}") for i in range(12, 19)]
            config = {"sampling": {"seed": 42, "history_count": 10, "test_count": 5, "mpdocvqa_max_pages": 6}}
            with patch("vlm_memory.pipeline.data.WORKSPACE", root), patch(
                    "vlm_memory.pipeline.data.mp_rows", side_effect=lambda split: train if split == "train" else val):
                manifest = select("mpdocvqa", config, root / "run")
                repeated = select("mpdocvqa", config, root / "repeat")
            self.assertEqual(manifest, repeated)
            self.assertEqual(len(manifest["memory"]), 10)
            self.assertEqual(len(manifest["test"]), 5)
            memory_docs = {t["metadata"]["doc_id"] for t in manifest["memory"]}
            self.assertFalse(memory_docs & {t["metadata"]["doc_id"] for t in manifest["test"]})
            self.assertNotIn("SECRET_GOLD", json.dumps(manifest))
            targets = list((root / "run/evaluation_targets").glob("*.json"))
            self.assertEqual(len(targets), 15)
            self.assertTrue(all(json.loads(p.read_text())["answers"] == ["SECRET_GOLD"] for p in targets))


if __name__ == "__main__":
    unittest.main()
