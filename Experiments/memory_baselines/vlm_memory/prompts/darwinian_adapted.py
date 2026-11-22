"""Prompt text for darwinian_adapted; no execution or memory update logic."""

# 将每条历史轨迹分解为可复用的子任务单元，并保留顺序和来源证据。
# 仅返回 JSON：每个单元包含来源 ID、前置条件、目标、轨迹步骤索引和验证方式。
# 步骤索引必须对应输入轨迹中的步骤；不得虚构成功动作。
EXTRACT_SYSTEM = (
    'Decompose each episode into reusable subtask units, preserving order and source evidence. '
    'Return JSON: {"units":[{"source_id":"...","precondition":"...","goal":"...","step_indices":[0]'
    ',"verification":"..."}]}. Each step index refers to a supplied trajectory step. Do not invent '
    'successful actions.'
)
