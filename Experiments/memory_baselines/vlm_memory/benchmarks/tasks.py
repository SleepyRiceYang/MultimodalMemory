from __future__ import annotations

import json
import sys
from pathlib import Path

from ..prompts.base_tasks import VISION2CODE_TASK

from ..config import WORKSPACE


def load_task(benchmark: str, index: int = 0) -> dict:
    if benchmark == "mathvision":
        repo = WORKSPACE / "Benchmark/MATH-Vision"
        rows = [json.loads(line) for line in (repo / "data/testmini.jsonl").read_text().splitlines()]
        row = rows[index]
        prompt = row["question"]
        if row.get("options") and "".join(row["options"]) != "ABCDE":
            prompt += "\n" + "\n".join(f"({chr(65 + i)}) {v}" for i, v in enumerate(row["options"]))
        # No reference answer/solution enters the task request or memory retrieval.
        return {"id": str(row["id"]), "benchmark": benchmark, "prompt": prompt,
                "query": "visual mathematics " + row["subject"] + " " + prompt,
                "image": repo / row["image"], "split": "testmini", "metadata": {"subject": row["subject"], "level": row["level"]}}
    if benchmark != "vision2code":
        raise ValueError(benchmark)
    repo = WORKSPACE / "Benchmark/Vision2Code"
    sys.path.insert(0, str(repo))
    from vision2code.benchmark.run_benchmark import selected_rows, source_image_path
    data = repo / "data/kaggle/vision2code"
    row = selected_rows(data, "test_mini", index + 1)[index]
    return {"id": row["question_folder"], "benchmark": benchmark,
            "prompt": VISION2CODE_TASK,
            "query": "recreate image visual code chart layout " + row["category"],
            "image": source_image_path(data, row), "split": "test_mini", "metadata": row}


def evaluate_math(output_dir: Path, task: dict, response: str, python: str) -> dict:
    import subprocess
    from ..config import write_json
    repo = WORKSPACE / "Benchmark/MATH-Vision"
    answer_file = (output_dir / "official_answers.jsonl").resolve()
    answer_file.write_text(json.dumps({"id": task["id"], "response": response}) + "\n")
    code = ("import sys; sys.path.insert(0,'evaluation'); "
            "from evaluate import evaluate,math_level_subject_acc; "
            "evaluate(sys.argv[1],True); math_level_subject_acc(sys.argv[1])")
    completed = subprocess.run([python, "-c", code, str(answer_file)], cwd=repo,
                               capture_output=True, text=True, timeout=90)
    (output_dir / "evaluator.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"MATH-Vision evaluator failed: {completed.stderr[-1500:]}")
    result = json.loads(answer_file.read_text().splitlines()[0])
    write_json(output_dir / "official_result.json", result)
    return {"status": "evaluated", "correct": result["correct"], "answer": result["model_answer"]}


def evaluate_vision(output_dir: Path, task: dict, response: str, model_config: dict,
                    evaluation_config: dict) -> dict:
    from argparse import Namespace
    from types import SimpleNamespace
    sys.path.insert(0, str(WORKSPACE / "Benchmark/Vision2Code"))
    from openai import OpenAI
    from vision2code.benchmark.run_benchmark import rate_sample, normalized_similarity
    from vision2code.generation.normalize_code import normalize_generated_code
    from vision2code.rendering.render_python import render_matplotlib_code
    from PIL import Image
    from ..config import write_json
    from ..prompts.evaluation import vision2code_rater_prompts
    import os
    import time

    code = normalize_generated_code(response)
    (output_dir / "generated_code.py").write_text(code)
    rendered = output_dir / "rendered_image.png"
    started = time.monotonic()
    result = render_matplotlib_code(code, rendered, timeout_sec=evaluation_config["render_timeout"])
    result["render_seconds"] = time.monotonic() - started
    write_json(output_dir / "render_result.json", result)
    if not result["render_success"]:
        return {**result, "rating_status": "render_failed", "rating_0_to_5": 0.0}
    with Image.open(task["image"]) as source, Image.open(rendered) as candidate:
        result["pixel_similarity_diagnostic"] = normalized_similarity(source, candidate)
    args = Namespace(rubric="dataset", rater_provider="local_vllm", rater_model=model_config["name"],
                     rater_temperature=0, rater_max_tokens=evaluation_config["rater_max_tokens"],
                     rater_api_timeout_sec=180, rater_reasoning=False, rater_repair_attempts=1,
                     rater_max_retries=1, rater_retry_sleep_sec=1, rater_retry_jitter_sec=0)
    client = OpenAI(base_url=model_config["base_url"], api_key=os.getenv(model_config["api_key_env"], "EMPTY"),
                    timeout=180, max_retries=1)
    rater_calls = []
    rater_prompts = vision2code_rater_prompts()
    def recorded_create(**kwargs):
        # Same official judge text, selected through the central prompt entry.
        kwargs["messages"] = [{"role": "system", "content": rater_prompts["system"]},
                              *kwargs["messages"][1:]]
        response = client.chat.completions.create(**kwargs)
        rater_calls.append({"usage": response.usage.model_dump() if response.usage else {},
                            "finish_reason": response.choices[0].finish_reason})
        return response
    recording_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorded_create)))
    started = time.monotonic()
    try:
        rating, raw = rate_sample(client=recording_client, source_image=task["image"], rendered_image=rendered,
                                  metadata=task["metadata"], dataset_name=task["metadata"]["source_dataset"], args=args)
        write_json(output_dir / "official_rubric_rating.json", rating)
        (output_dir / "rater_response.txt").write_text(raw)
        result.update(rating_status="evaluated", rating_0_to_5=rating["final_rating_0_to_5"])
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        parse_failure = isinstance(exc, json.JSONDecodeError) or (
            isinstance(exc, ValueError) and any(marker in str(exc).lower() for marker in (
                "parse response as a json", "parse rubric response", "json object")))
        if parse_failure:
            # An invalid/truncated judge response is a failed evaluation sample,
            # not a pipeline infrastructure failure. Keep the run moving and
            # conservatively assign the rubric floor.
            result.update(rating_status="invalid_json", rating_0_to_5=0.0,
                          rating_error=error_text, rating_failure_reason="invalid_rater_json")
            write_json(output_dir / "rating_failure.json", {
                "status": "invalid_json", "score": 0.0,
                "reason": "Rater response remained unparsable after its configured repair attempt.",
                "error": error_text, "rater_calls": rater_calls,
            })
        else:
            result.update(rating_status="error", rating_error=error_text)
    result["rater_seconds"] = time.monotonic() - started
    result["rater_calls"] = rater_calls
    result["judge_model"] = model_config["name"]
    result["judge_is_same_as_solver"] = True
    return result
