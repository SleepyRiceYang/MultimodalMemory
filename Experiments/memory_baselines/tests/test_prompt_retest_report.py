import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_memory.config import write_json
from vlm_memory.pipeline.prompt_retest import make_report


class RetestReportTests(unittest.TestCase):
    def test_report_keeps_equal_budget_results_separate_from_historical_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root, previous = workspace / "new", workspace / "previous"
            config = {"model": {"name": "local", "temperature": 0, "max_tokens": 32678,
                                 "image_max_side": 1024}, "fixed_split_dir": "splits"}
            write_json(root / "base_prompts.snapshot.json", {
                v: {b: {"system": f"Saved fixture {v}/{b}"} for b in ("mathvision", "vision2code")}
                for v in ("base_v1", "base_v2")
            })
            history = {b: {"old": {"mean": 0.6}, "base_v1": {"mean": 0.8}}
                       for b in ("mathvision", "vision2code")}
            write_json(previous / "analysis.json", {"benchmarks": history})
            for benchmark in history:
                ids = [str(i) for i in range(5)]
                write_json(root / benchmark / "sample_ids.json", ids)
                write_json(workspace / "splits" / benchmark / "dataset_manifest.json", {
                    "memory": [{"id": i, "metadata": {"level": 2, "subject": "logic", "category": "Documents"}} for i in ids]})
                for version in ("base_v1", "base_v2"):
                    for task_id in ids:
                        folder = root / benchmark / version / task_id
                        write_json(folder / "result.json", {
                            "status": "completed", "score": 0 if version == "base_v1" else 1,
                            "evaluation": {"answer": "A", "render_success": False},
                            "usage": {"completion_tokens": 100, "prompt_tokens": 200},
                            "seconds": 1, "generation_truncated": False,
                            "prediction": "reasoning</think>Final answer"})
                        write_json(folder / "calls/generation.json", {"finish_reason": "stop"})
                        write_json(folder / "evaluation.json", {"status": "test render error"})
            with patch("vlm_memory.pipeline.prompt_retest.WORKSPACE", workspace):
                result = make_report(root, previous, config)
            self.assertEqual(result["max_tokens"], 32678)
            self.assertEqual(result["benchmarks"]["mathvision"]["delta"], 1)
            self.assertTrue(Path(result["report"]).is_file())
            self.assertEqual(len((root / "paired_results.csv").read_text().splitlines()), 21)
            text = Path(result["report"]).read_text()
            self.assertIn("没有复用上一轮响应", text)
            self.assertIn("32,678", text)
            self.assertIn("test render error", text)


if __name__ == "__main__":
    unittest.main()
