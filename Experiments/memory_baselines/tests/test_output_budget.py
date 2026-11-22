import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vlm_memory.models.local import LocalModel


class OutputBudgetTests(unittest.TestCase):
    def model(self, cap):
        model = LocalModel.__new__(LocalModel)
        model.config = {"base_url": "http://localhost:8077/v1", "name": "qwen3.8-27b",
                        "api_key_env": "TEST_UNUSED_KEY", "max_tokens": cap,
                        "timeout": 5, "temperature": 0, "image_max_side": 1024}
        event = SimpleNamespace(id="test", choices=[SimpleNamespace(
            index=0, delta=SimpleNamespace(content="answer"), finish_reason="stop")], usage=None)
        response = Mock()
        response.__iter__ = Mock(return_value=iter([event]))
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(return_value=response))))
        return model

    def test_large_output_budget_reserves_exact_input_tokens(self):
        model = self.model(262144)
        body = io.BytesIO(json.dumps({"count": 567, "max_model_len": 262144}).encode())
        with patch("vlm_memory.models.local.urllib.request.urlopen", return_value=body):
            result = model.generate("system", "question", [])
        self.assertEqual(result["effective_max_tokens"], 261577)
        self.assertEqual(model.client.chat.completions.create.call_args.kwargs["max_tokens"], 261577)

    def test_small_explicit_budget_remains_unchanged(self):
        model = self.model(262144)
        with patch("vlm_memory.models.local.urllib.request.urlopen") as tokenize:
            result = model.generate("system", "question", [], max_tokens=3072)
        tokenize.assert_not_called()
        self.assertEqual(result["effective_max_tokens"], 3072)

    def test_exhausted_context_does_not_issue_generation(self):
        model = self.model(262144)
        body = io.BytesIO(json.dumps({"count": 262144, "max_model_len": 262144}).encode())
        with patch("vlm_memory.models.local.urllib.request.urlopen", return_value=body):
            with self.assertRaisesRegex(ValueError, "Input exhausts"):
                model.generate("system", "question", [])
        model.client.chat.completions.create.assert_not_called()

    def test_32768_output_budget_is_checked_against_real_input(self):
        model=self.model(32768)
        body=io.BytesIO(json.dumps({'count':38742,'max_model_len':65536}).encode())
        with patch('vlm_memory.models.local.urllib.request.urlopen',return_value=body):
            result=model.generate('system','large grouping request',[])
        self.assertEqual(result['effective_max_tokens'],26794)
        self.assertEqual(result['context_limit'],65536)

    def test_small_checked_requests_reject_oversized_inputs_before_generation(self):
        model=self.model(1536)
        model.config.update(check_context_budget=True,max_input_tokens=16384)
        body=io.BytesIO(json.dumps({'count':18000,'max_model_len':65536}).encode())
        with patch('vlm_memory.models.local.urllib.request.urlopen',return_value=body):
            with self.assertRaisesRegex(ValueError,'split the request'):
                model.generate('system','too much evidence',[])
        model.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
