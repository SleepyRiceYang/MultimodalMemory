"""Prompt text for evoskill_adapted; no execution or memory update logic."""

# 根据提供的历史案例构建可迁移的结构化文本技能，仅返回 JSON。
# 每项技能包含来源 ID、意图、Markdown 操作流程、备选定位/策略、失败信号与恢复动作、失败案例。
# 根据反馈识别失败；不要把特定任务的常数写进可复用流程。
BUILD_SYSTEM = (
    'Build a transferable structured text skill from the supplied episodes. Return JSON: '
    '{"skills":[{"source_ids":["..."],"intent":"...","plan":"markdown '
    'procedure","backup":"alternative localization/strategy","recover":"failure signals and '
    'recovery actions","failure_cases":["..."]}]}. Use feedback to identify failures; keep task-'
    'specific constants out of reusable procedures.'
)

# 根据提供的反馈诊断可测量的执行失败，以 JSON 返回失败信号、恢复检查项和证据 ID。
# 不得声称修订方案已经过测试。
CRITIC_SYSTEM = (
    'Diagnose measurable execution failures from the supplied feedback. Return JSON with '
    'failure_signals, recovery_checks and evidence_ids. Do not claim that revisions have been '
    'tested.'
)

# 修订后缀：根据批评反馈修改初始技能。
REVISION_SUFFIX = (
    ' Revise the initial skill using the critic feedback.'
)
