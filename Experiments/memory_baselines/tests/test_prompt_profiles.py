import json
import tempfile
import unittest
from pathlib import Path

from vlm_memory.prompts import BASE_TASKS, DOCUMENT_PROMPTS, get_task_prompt, snapshot_prompts
from vlm_memory.pipeline.run_100_100_nomemory import _matches_saved_prompts, snapshot_run


class PromptProfileTests(unittest.TestCase):
    def test_only_three_current_benchmark_definitions_are_available(self):
        self.assertEqual(set(BASE_TASKS), {"mathvision", "vision2code", "mpdocvqa"})
        self.assertEqual(BASE_TASKS["mathvision"]["version"], "mathvision_concise_v1")
        self.assertEqual(get_task_prompt("mathvision"), get_task_prompt("mathvision", "mathvision_concise_v1"))
        for version in ("old", "base_v1", "base_v2", "formal_100_100"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "not current"):
                get_task_prompt("mathvision", version)

    def test_document_stages_render_existing_question_and_reading_payloads(self):
        system, template = get_task_prompt("mpdocvqa", "formal_100_100", "read_pages")
        self.assertEqual(system, DOCUMENT_PROMPTS["read_pages"])
        self.assertEqual(template.format(question="Count?", page_indices=[2, 3]),
                         "Question: Count?\nSupplied global page indices: [2, 3].")
        _, template = get_task_prompt("mpdocvqa", stage="aggregate")
        self.assertEqual(json.loads(template.format(question_json='"Count?"', readings_json='[]')),
                         {"question": "Count?", "page_batch_readings": []})
        with self.assertRaisesRegex(ValueError, "Select"):
            get_task_prompt("mpdocvqa")

    def test_snapshot_ignores_unused_historical_profiles_but_checks_selected_text(self):
        current = snapshot_prompts({"benchmarks": ["vision2code"], "prompt_version": "formal_100_100"})
        saved = json.loads(json.dumps(current))
        saved["removed_historical_profile"] = {"mathvision": {"system": "old fixture"}}
        self.assertTrue(_matches_saved_prompts(saved, current))
        saved["formal_100_100"]["vision2code"]["system"] = "changed fixture"
        self.assertFalse(_matches_saved_prompts(saved, current))

    def test_existing_document_snapshot_remains_compatible(self):
        current = snapshot_prompts({"benchmarks": ["mpdocvqa"], "prompt_version": "formal_100_100"})
        self.assertTrue(_matches_saved_prompts({"__document_prompts__": DOCUMENT_PROMPTS}, current))
        self.assertFalse(_matches_saved_prompts({"__document_prompts__": {}}, current))

    def test_new_snapshot_contains_only_selected_current_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"benchmarks": ["mathvision"], "prompt_version": "mathvision_concise_v1",
                      "model": {"name": "local", "timeout": 10}}
            snapshot_run(root, config)
            saved = json.loads((root / "base_prompts.snapshot.json").read_text())
            self.assertEqual(set(saved), {"mathvision_concise_v1"})
            self.assertEqual(set(saved["mathvision_concise_v1"]), {"mathvision"})


if __name__ == "__main__":
    unittest.main()
