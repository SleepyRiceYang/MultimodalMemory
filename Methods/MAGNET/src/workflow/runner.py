from __future__ import annotations

import json
import shutil
import threading
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tqdm import tqdm

_AUX_SUFFIXES = (
    "_0observe.json",
    "_1think.json",
    "_1think_action.json",
    "_reflect.json",
)
_AUX_ERROR_DIRS = {"action_desc_errors", "action_predict_errors"}


def _safe_copy(src: Path, dst: Path) -> None:
    """Copy a file while tolerating missing extended-attribute support."""
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        print(f"[WorkflowRunner] copy2 failed for {src} -> {dst}: {exc}; falling back to copy().")
        shutil.copy(src, dst)


def should_skip_action_json(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    if any(parent in _AUX_ERROR_DIRS for parent in path.parts):
        return True
    if any(name.endswith(suffix) for suffix in _AUX_SUFFIXES):
        return True
    return False


def safe_load_action_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"[WorkflowRunner] Failed to read JSON at {path}: {exc}")
        return None
    return payload if isinstance(payload, dict) else None


def record_has_plan_fields(record: dict[str, Any]) -> bool:
    action_think = record.get("action_think")
    if not action_think:
        return False
    action_desc = record.get("action_desc")
    if not isinstance(action_desc, dict) or not action_desc:
        return False
    desc_type = action_desc.get("type")
    desc_text = action_desc.get("description")
    return bool(desc_type) and bool(desc_text)


def record_has_prediction(record: dict[str, Any]) -> bool:
    predictions = record.get("action_predict")
    if not isinstance(predictions, dict) or not predictions:
        return False
    for action_mode in predictions.values():
        if not isinstance(action_mode, dict):
            continue
        for screen_entry in action_mode.values():
            if not isinstance(screen_entry, dict):
                continue
            action_value = screen_entry.get("ACTION")
            if isinstance(action_value, str) and action_value.strip():
                return True
            if isinstance(action_value, dict):
                nested = action_value.get("ACTION")
                if isinstance(nested, str) and nested.strip():
                    return True
    return False


class WorkflowRunner:
    """Lightweight orchestrator for running specific COAT stages."""

    def __init__(
        self,
        agent,
        dataset: Sequence,
        work_dir: Path,
        limit: int = 0,
        desc: str = "",
        num_workers: int = 1,
        agent_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.work_dir = Path(work_dir)
        self.limit = limit
        self.desc = desc or "Stage"
        self.num_workers = max(1, num_workers)
        self.agent_factory = agent_factory
        self._thread_local = threading.local()

        if self.num_workers == 1:
            if agent is None:
                if self.agent_factory is None:
                    raise ValueError("Sequential execution requires an agent instance or agent_factory.")
                agent = self.agent_factory()
            self._single_agent = agent
            self._single_handlers: dict[str, Callable] = {}
            self._shared_agent = None
            self._shared_handlers: dict[str, Callable] = {}
        else:
            if agent is None and self.agent_factory is None:
                raise ValueError("Parallel execution requires an agent_factory to create the shared agent.")
            self._single_agent = agent
            self._single_handlers = {}
            if self._single_agent is not None:
                self._shared_agent = self._single_agent
            else:
                self._shared_agent = self.agent_factory()
            self._shared_handlers: dict[str, Callable] = {}

    def _summarize_step(self, step: Any) -> str:
        def _maybe_get(key: str):
            if isinstance(step, dict):
                return step.get(key)
            return getattr(step, key, None)

        summary_parts = []
        for key in ("subset", "episode_id", "step_id"):
            value = _maybe_get(key)
            if value is not None:
                summary_parts.append(f"{key}={value}")

        if not summary_parts:
            raw = repr(step)
            return raw[:200] + ("..." if len(raw) > 200 else "")
        return ", ".join(summary_parts)

    def _record_failure(self, stage: str, step: Any, exc: Exception) -> str:
        step_info = self._summarize_step(step)
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(f"[WorkflowRunner] Error during '{stage}' for step [{step_info}]:\n{trace}")
        return step_info

    def _report_failures(self, stage: str, failures: list[str]) -> None:
        if not failures:
            return
        sample = ", ".join(failures[:5])
        print(
            f"[WorkflowRunner] {len(failures)} step(s) failed during '{stage}'. "
            f"Sample failures: {sample}"
        )

    def _report_stage_summary(self, stage: str, total: int, failure_count: int) -> None:
        success = max(0, total - failure_count)
        print(
            f"[WorkflowRunner] Stage '{stage}' summary: total={total}, "
            f"success={success}, failed={failure_count}"
        )

    def _report_field_counts(self, stage: str) -> None:
        counter: Counter[str] = Counter()
        total = 0
        for json_path in self.work_dir.rglob("*.json"):
            if should_skip_action_json(json_path):
                continue
            record = safe_load_action_json(json_path)
            if record is None:
                continue
            total += 1
            for key, value in record.items():
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                if isinstance(value, (list, dict)) and not value:
                    continue
                counter[key] += 1

        if total == 0:
            print(f"[WorkflowRunner] No JSON artifacts found for stage '{stage}'.")
            return

        print(f"[WorkflowRunner] Stage '{stage}' JSON records: {total}")
        for key in sorted(counter):
            print(f"[WorkflowRunner]   field '{key}': {counter[key]}")

    def _build_iterator(self):
        total = len(self.dataset)
        iterator: Iterable = iter(self.dataset)
        if self.limit and self.limit > 0:
            total = min(total, self.limit)
            iterator = islice(iterator, 0, self.limit)
        return iter(iterator), total

    def _expected_path(self, stage: str):
        return self.work_dir / f".{stage}_expected"

    def _write_stage_expected(self, stage: str, total: int):
        path = self._expected_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(total), encoding="utf-8")

    def _select_handler(self, agent, stage: str):
        if stage == "flow":
            return agent.flow_with_memory if getattr(agent, "procedural_memory", None) else agent.flow
        if stage == "predict":
            return agent.predict_with_memory if getattr(agent, "stationary_memory", None) else agent.predict
        if stage == "reflect":
            return agent.reflect
        raise ValueError(f"Unknown stage '{stage}'.")

    def _get_single_handler(self, stage: str):
        handler = self._single_handlers.get(stage)
        if handler is None:
            handler = self._select_handler(self._single_agent, stage)
            self._single_handlers[stage] = handler
        return handler

    def _get_parallel_handler(self, stage: str):
        if self._shared_agent is not None:
            handler = self._shared_handlers.get(stage)
            if handler is None:
                handler = self._select_handler(self._shared_agent, stage)
                self._shared_handlers[stage] = handler
            return handler
        state = getattr(self._thread_local, "state", None)
        if state is None:
            agent = self.agent_factory()
            state = {"agent": agent, "handlers": {}}
            self._thread_local.state = state
        handler = state["handlers"].get(stage)
        if handler is None:
            handler = self._select_handler(state["agent"], stage)
            state["handlers"][stage] = handler
        return handler

    def _run_stage(self, stage: str):
        if self.num_workers == 1:
            self._run_stage_sequential(stage)
        else:
            self._run_stage_parallel(stage)

    def _run_stage_sequential(self, stage: str):
        iterator, total = self._build_iterator()
        self._write_stage_expected(stage, total)
        handler = self._get_single_handler(stage)
        failures: list[str] = []
        for step in tqdm(iterator, total=total, desc=self.desc):
            # handler(step, str(self.work_dir))
            try:
                handler(step, str(self.work_dir))
            except Exception as exc:
                failures.append(self._record_failure(stage, step, exc))
        self._report_failures(stage, failures)
        self._report_stage_summary(stage, total, len(failures))
        self._report_field_counts(stage)

    def _run_stage_parallel(self, stage: str):
        iterator, total = self._build_iterator()
        self._write_stage_expected(stage, total)
        work_dir = str(self.work_dir)
        failures: list[str] = []

        def task(step):
            handler = self._get_parallel_handler(stage)
            try:
                handler(step, work_dir)
            except Exception as exc:
                return self._record_failure(stage, step, exc)
            return None

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor, tqdm(total=total, desc=self.desc) as progress:
            pending = set()
            try:
                while len(pending) < self.num_workers:
                    step = next(iterator)
                    pending.add(executor.submit(task, step))
            except StopIteration:
                pass

            if not pending:
                self._report_failures(stage, failures)
                self._report_stage_summary(stage, total, len(failures))
                self._report_field_counts(stage)
                return

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover
                        result = self._record_failure(stage, "unknown", exc)
                    if result is not None:
                        failures.append(result)
                    progress.update()
                    # step = next(iterator)
                    try:
                        step = next(iterator)
                    except StopIteration:
                        continue
                    pending.add(executor.submit(task, step))
        self._report_failures(stage, failures)
        self._report_stage_summary(stage, total, len(failures))
        self._report_field_counts(stage)

    def run_flow(self):
        self._run_stage("flow")

    def run_predict(self):
        self._run_stage("predict")

    def run_reflect(self):
        """Re-run the reflection step for each saved JSON (if missing)."""
        self._run_stage("reflect")


def ensure_flow_dir(flow_dir: Path, output_dir: Path):
    """Copy flow artifacts when reusing cached outputs."""
    flow_dir = Path(flow_dir)
    output_dir = Path(output_dir)
    if flow_dir == output_dir:
        return
    for src_path in flow_dir.rglob("*"):
        rel = src_path.relative_to(flow_dir)
        dst_path = output_dir / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.suffix.lower() != ".json":
            _safe_copy(src_path, dst_path)
            continue
        try:
            with src_path.open("r", encoding="utf-8") as handle:
                src_payload = json.load(handle)
        except Exception:
            _safe_copy(src_path, dst_path)
            continue
        if not isinstance(src_payload, dict):
            _safe_copy(src_path, dst_path)
            continue
        preserved_predict = None
        preserved_result = None
        if dst_path.exists():
            try:
                with dst_path.open("r", encoding="utf-8") as handle:
                    dst_payload = json.load(handle)
            except Exception:
                dst_payload = None
            if isinstance(dst_payload, dict):
                preserved_predict = dst_payload.get("action_predict")
                preserved_result = dst_payload.get("action_result")
        if preserved_predict:
            src_payload["action_predict"] = preserved_predict
        if preserved_result:
            src_payload["action_result"] = preserved_result
        with dst_path.open("w", encoding="utf-8") as handle:
            json.dump(src_payload, handle, ensure_ascii=False, indent=4)
