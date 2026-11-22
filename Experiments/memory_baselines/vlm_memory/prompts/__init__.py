"""Central task, memory and judge prompt exports."""
from .base_tasks import BASE_TASKS, DOCUMENT_PROMPTS, get_task_prompt, snapshot_prompts
from .base_tasks import MATHVISION_SYSTEM as MATH_SYSTEM, VISION2CODE_SYSTEM as VISION_SYSTEM
from .memory_context import MEMORY_NOTICE
from . import ksi_style, magnet_adapted, darwinian_adapted, evoskill_adapted

CONSTRUCTION = {
    "ksi_style": ksi_style.DISTILL_SYSTEM,
    "magnet_adapted": magnet_adapted.EXTRACT_SYSTEM,
    "darwinian_adapted": darwinian_adapted.EXTRACT_SYSTEM,
    "evoskill_adapted": evoskill_adapted.BUILD_SYSTEM,
}

__all__ = ["BASE_TASKS", "DOCUMENT_PROMPTS", "get_task_prompt", "snapshot_prompts",
           "MATH_SYSTEM", "VISION_SYSTEM", "MEMORY_NOTICE", "CONSTRUCTION"]
