"""Current task prompts for all three benchmarks; no historical prompt text."""
from __future__ import annotations

# MathVision 系统提示词：使用图像和题目给定数值解答当前视觉数学问题。
# 要求逐步思考但只展示必要推理，不重复计算或重启解题；不确定时给出证据最充分的答案。
# 最终只输出一个 \boxed{} 并停止；选择题的框内只写选项字母。
MATHVISION_SYSTEM = (
    'Solve the current visual mathematics problem using the image and given values. Think step by '
    'step, but show only the essential reasoning. Do not repeat calculations or restart the '
    'solution. If unsure, give the best-supported answer. Finish with exactly one \\boxed{} and '
    'stop. For multiple choice, put only the option letter in the box.'
)

# Vision2Code 系统提示词：用可执行的 Python matplotlib/numpy 代码重现当前参考图。
# 只返回代码；OUTPUT_PATH 由执行框架提供，必须调用 plt.savefig(OUTPUT_PATH) 或 fig.savefig(OUTPUT_PATH)。
# 不得重新赋值 OUTPUT_PATH，不得读取外部文件或访问网络。
VISION2CODE_SYSTEM = (
    'Recreate the CURRENT reference image with executable Python matplotlib/numpy code. Return code'
    ' only. OUTPUT_PATH is supplied by the execution harness. You must call '
    'plt.savefig(OUTPUT_PATH) or fig.savefig(OUTPUT_PATH). Do not reassign OUTPUT_PATH. Do not read'
    ' external files or use network access.'
)

# Vision2Code 用户任务补充：生成重现当前参考图的 Python 代码。
VISION2CODE_TASK = (
    'Generate Python code to recreate the current reference image.'
)

# MP-DocVQA 页面读取阶段系统提示词：只能依据提供的文档页面回答，简要逐步推理。
# 使用提供的从 0 开始的页面索引，而不是页面上印刷的页码。仅返回 JSON，字段为答案、答案页索引、支持证据和是否找到。
# 如果文档中没有答案，答案使用空字符串、页索引使用 null，并将 found 设为 false。
MPDOCVQA_READ_SYSTEM = (
    'Answer the question using only the supplied document pages. Think step by step, keeping the '
    'reasoning brief. Use the supplied zero-based page indices, not printed page numbers. Return '
    'JSON only: {"answer":"short exact answer","answer_page_idx":0,"evidence":"brief supporting '
    'text","found":true}. If the answer is absent, use an empty answer, null page index, and '
    'found=false.'
)

# MP-DocVQA 汇总阶段系统提示词：从各页面批次的读取结果中选择证据最充分的答案，简要推理且不得编造。
# 仅返回 JSON，包含简短准确答案、全局 0 起始页索引（或 null）、简要支持证据和 found 字段。
MPDOCVQA_AGGREGATE_SYSTEM = (
    'Choose the answer best supported by the supplied page-batch readings. Think step by step, '
    'keeping the reasoning brief. Do not invent information. Return JSON only with answer (short '
    'exact string), answer_page_idx (supplied global zero-based index, or null), evidence (brief '
    'supporting text), and found.'
)

# MP-DocVQA 页面读取阶段用户模板：给出问题以及本次提供的全局页面索引。
MPDOCVQA_READ_USER = "Question: {question}\nSupplied global page indices: {page_indices}."
# MP-DocVQA 汇总阶段用户模板：以 JSON 提供问题和各页面批次的读取结果。
MPDOCVQA_AGGREGATE_USER = '{{"question": {question_json}, "page_batch_readings": {readings_json}}}'

DOCUMENT_PROMPTS = {"read_pages": MPDOCVQA_READ_SYSTEM, "aggregate": MPDOCVQA_AGGREGATE_SYSTEM}

BASE_TASKS = {
    "mathvision": {"version": "mathvision_concise_v1", "system": MATHVISION_SYSTEM, "user": "{task}"},
    "vision2code": {"version": "formal_100_100", "system": VISION2CODE_SYSTEM, "user": "{task}"},
    "mpdocvqa": {
        "version": "formal_100_100",
        "stages": {
            "read_pages": {"system": MPDOCVQA_READ_SYSTEM, "user": MPDOCVQA_READ_USER},
            "aggregate": {"system": MPDOCVQA_AGGREGATE_SYSTEM, "user": MPDOCVQA_AGGREGATE_USER},
        },
    },
}


def get_task_prompt(benchmark: str, version: str | None = None,
                    stage: str | None = None) -> tuple[str, str]:
    """Select current text; reject removed versions instead of silently changing a run."""
    if benchmark not in BASE_TASKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    spec = BASE_TASKS[benchmark]
    if version not in (None, "current", spec["version"]):
        raise ValueError(f"Prompt version {version!r} is not current for {benchmark}; "
                         f"use {spec['version']!r} and a new run ID. Historical text remains in saved snapshots.")
    if "stages" in spec:
        if stage not in spec["stages"]:
            raise ValueError(f"Select read_pages or aggregate for {benchmark}")
        selected = spec["stages"][stage]
    else:
        if stage is not None:
            raise ValueError(f"{benchmark} has no stage {stage!r}")
        selected = spec
    return selected["system"], selected["user"]


def snapshot_prompts(config: dict) -> dict:
    """Snapshot only the task prompts selected for this run."""
    result = {}
    for benchmark in config.get("benchmarks", []):
        spec = BASE_TASKS[benchmark]
        get_task_prompt(benchmark, config.get("prompt_version"),
                        "read_pages" if benchmark == "mpdocvqa" else None)
        result.setdefault(spec["version"], {})[benchmark] = spec
    return result
