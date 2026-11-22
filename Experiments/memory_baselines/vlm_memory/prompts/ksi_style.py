"""Prompt text for ksi_style; no execution or memory update logic."""

# 将提供的历史尝试及反馈提炼为有证据支撑的文本知识；给出具体操作，不要写泛泛建议。
# 仅返回 JSON 知识条目，每条包含主张、适用条件、不适用情形、证据 ID、被否定的假设和检查项。
# 每个证据 ID 必须指向输入案例；这些知识不会附带图像。
DISTILL_SYSTEM = (
    'Distill the supplied historical attempts and feedback into evidence-grounded textual '
    'knowledge. Use specific operations, not generic advice. Return JSON: '
    '{"knowledge":[{"claim":"...","conditions":["..."],"exclusions":["..."],"evidence_ids":["source'
    ' id"],"rejected_hypotheses":["..."],"checks":["..."]}]}. Every evidence id must refer to an '
    'input episode. No images will be injected with this knowledge.'
)

# 审阅提供的尝试及其测量反馈，以 JSON 给出具体主张、反例和证据来源 ID。
# 不要推断输入中没有观测到的结果。
REVIEW_SYSTEM = (
    'Review the supplied attempts and their measured feedback. Return JSON with concrete claims, '
    'counterexamples, and evidence source IDs. Do not infer unseen outcomes.'
)

# 跨任务补充指令：比较这 {task_count} 个任务中的主张，限定相互冲突的主张适用范围，并引用原始任务 ID。
# 保留具体检查项和反例；最多输出 10 条知识。
CROSS_TASK_SUFFIX = (
    ' Compare claims across these {task_count} tasks; scope conflicting claims and cite original '
    'source task IDs. Preserve concrete checks and counterexamples. Return at most 10 knowledge '
    'items.'
)

# Current paired protocol only; historical runs retain their original prompts above.
EVIDENCE_SUFFIX = '''
Memory is for DIFFERENT future tasks. A claim must be a reusable conditional operation or a failure-avoidance check, not a report that this particular answer/code was correct. Remove source-specific dates, names, answers and coordinates from the claim. Use source observations and actual generated output to check feedback: an aggregate success/score is not proof of correct layout, content or every intermediate step. Distinguish an observed error from a suggested remedy; never say an unexecuted remedy was verified. Conditions define the required task structure; exclusions describe where the operation does not apply; checks state concrete observable validation. Keep only supported nonredundant lessons, and return an empty list if none is supported. For example, a demonstrated overlap can justify checking text-block bounding boxes, but cannot justify claiming a particular new layout algorithm succeeded.'''

PAIRED_REVIEW_SYSTEM = '''Inspect the supplied historical task, executed code/answer, source observations, output images and feedback. First write observed_support: a short account of which concrete operations visibly worked or failed, with the supporting code or output detail. Then produce up to three claims about reusable operations or checks demonstrated by that evidence. A demonstrated successful operation is sufficient for a scoped lesson; multiple successful tasks are NOT required here. Preserve limitations of this single example. Each claim contains claim, evidence_ids and counterexample_or_limit. Do not merely grade the old task. If an operation failed, record the observed pitfall, not an untested repair. Return JSON with observed_support and claims; claims can be empty only when no supported operation or check exists.'''

PAIRED_DISTILL_SYSTEM = '''Create a small reusable knowledge memory from the supplied historical evidence and review. Write instructions to an agent solving a DIFFERENT future task. Keep at most TWO nonredundant lessons. Each lesson must be supported by an operation actually present in the supplied execution, or by a directly observed failure.
Fields: claim = specific transferable operation/check; conditions = required input structure, not instructions to reuse old data; exclusions = cases where the operation is inappropriate; evidence_ids = supplied sources; rejected_hypotheses = observed unsupported assumptions; checks = observable validation of the operation. Leave exclusions and rejected_hypotheses EMPTY unless the evidence supports a concrete restriction or refuted assumption. Do not invent API restrictions: rotation, different labels or unequal spacing alone do not invalidate tick-position/label mapping. Distinguish item positions from item labels; they need matching order, not necessarily identical values.
PARAMETERIZE every source-specific value: dates, years, number of items, labels, names, colors, measured values and pixel positions. Refer to "current categories", "current values", "reference styling", etc. Never require the old data series or copy old axis titles. API names and true mathematical constants can remain. Do not claim a high aggregate score proves every detail correct. Do not invent an executed repair. A check can recommend verifying an observed failure without claiming a remedy succeeded.
Example of abstraction (illustrative only, not evidence): "For a labeled sequence, derive tick positions from the current item count and map each current label to its corresponding item." Bad: "Use the original five labels or dates from the old task." Do not copy this example unless supported by the actual evidence. Return JSON knowledge array with the stated fields, or an empty array when evidence is insufficient.'''
