"""Official Vision2Code judge prompt access; solver prompts live in base_tasks.

The dataset-specific rubric and image-content builders are maintained by the
local upstream benchmark. Re-export that current source instead of copying its
large rubric tables or introducing a different judge instruction.
"""
from __future__ import annotations

import sys

from ..config import WORKSPACE


def vision2code_rater_prompts() -> dict:
    # 官方评分提示词不在本目录重复定义；此处从 benchmark 源码动态导入，保持唯一来源。
    repo = str(WORKSPACE / "Benchmark/Vision2Code")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from vision2code.evaluation.dataset_rubrics import (
        RATER_SYSTEM_PROMPT, build_rater_user_content, build_rater_repair_prompt,
    )
    return {"system": RATER_SYSTEM_PROMPT, "user": build_rater_user_content,
            "repair": build_rater_repair_prompt,
            "source": "Benchmark/Vision2Code/vision2code/evaluation/dataset_rubrics.py"}
