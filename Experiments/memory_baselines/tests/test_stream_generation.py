import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from vlm_memory.models.local import LocalModel
from vlm_memory.pipeline.pilot import call


def event(content=None, reason=None, usage=None, reasoning=None):
    choices = [] if usage else [SimpleNamespace(
        index=0, delta=SimpleNamespace(content=content, reasoning_content=reasoning),
        finish_reason=reason)]
    return SimpleNamespace(id="request-test", choices=choices, usage=usage)


class Stream:
    def __init__(self, events, error=None):
        self.events, self.error, self.closed = events, error, False

    def __iter__(self):
        yield from self.events
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


class StreamingTests(unittest.TestCase):
    def model(self, stream):
        model = LocalModel.__new__(LocalModel)
        model.config = {"name": "local", "max_tokens": 32678,
                        "temperature": 0, "image_max_side": 1024}
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(return_value=stream))))
        return model

    def test_content_reasoning_and_final_usage_are_preserved(self):
        usage = Mock()
        usage.model_dump.return_value = {"completion_tokens": 7, "prompt_tokens": 3,
                                        "total_tokens": 10}
        stream = Stream([event(reasoning="think"), event("an"), event("swer"),
                         event(reason="stop"), event(usage=usage)])
        model = self.model(stream)
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / "generation.partial.json"
            result = model.generate("system", "question", [], progress_path=progress)
            saved = json.loads(progress.read_text())
        self.assertEqual(result["text"], "answer")
        self.assertEqual(result["reasoning"], "think")
        self.assertEqual(result["usage"]["completion_tokens"], 7)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["request_id"], "request-test")
        self.assertTrue(stream.closed)
        args = model.client.chat.completions.create.call_args.kwargs
        self.assertTrue(args["stream"])
        self.assertEqual(args["max_tokens"], 32678)

    def test_transport_failure_keeps_partial_output_and_closes_stream(self):
        stream = Stream([event("unfinished")], TimeoutError("stalled"))
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp) / "generation.partial.json"
            with self.assertRaises(TimeoutError):
                self.model(stream).generate("system", "question", [], progress_path=progress)
            saved = json.loads(progress.read_text())
        self.assertEqual(saved["status"], "error")
        self.assertEqual(saved["text"], "unfinished")
        self.assertTrue(stream.closed)

    def test_stream_without_terminal_finish_is_rejected(self):
        stream = Stream([event("truncated")])
        with self.assertRaisesRegex(RuntimeError, "without a finish_reason"):
            self.model(stream).generate("system", "question", [])
        self.assertTrue(stream.closed)

    def test_failed_stream_is_not_reused_as_a_completed_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            failed = self.model(Stream([event("discard this partial")], TimeoutError("stalled")))
            with self.assertRaises(TimeoutError):
                call(failed, folder, "generation", "system", "question", [])
            self.assertFalse((folder / "calls/generation.json").exists())
            recovered = self.model(Stream([event("complete answer", reason="stop")]))
            response = call(recovered, folder, "generation", "system", "question", [])
            saved = json.loads((folder / "calls/generation.json").read_text())
            self.assertEqual(saved["text"], "complete answer")
            self.assertEqual(response["text"], "complete answer")
            history = list((folder / "calls/partial_history").glob("generation_*.json"))
            self.assertEqual(len(history), 1)
            self.assertEqual(json.loads(history[0].read_text())["text"], "discard this partial")


if __name__ == "__main__":
    unittest.main()
