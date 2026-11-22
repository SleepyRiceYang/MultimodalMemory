"""Prompt text for magnet_adapted; no execution or memory update logic."""

EXTRACT_SYSTEM = (
    'Extract a reusable workflow plus visual-object/function associations from each supplied '
    'episode. In this reasoning-domain adaptation, visual objects are chart/math components. Return'
    ' JSON: {"entries":[{"source_id":"...","workflow":["..."],"patches":[{"image_index":0,"box":[0.'
    '0,0.0,1.0,1.0],"object":"...","function":"..."}]}]}. Boxes are normalized '
    '[left,top,right,bottom]. Prefer a small meaningful object/region and link it to its function. '
    'Maximum two patches per episode.'
)

# EXTRACT_SYSTEM 中文译文：从每个历史案例提取可复用工作流程，以及视觉对象与其功能之间的关联。
# 此推理领域适配中，视觉对象指图表或数学组件。仅返回 JSON，顶层字段为 entries。
# 每个 entry 包含来源 ID、工作流程，以及 patches 列表；每个 patch 包含图像索引、边界框、对象和功能。
# box 使用归一化 [左, 上, 右, 下] 坐标。优先选取较小且有意义的对象/区域，并说明它的功能；每个案例最多两个 patch。

# 注入记忆时工作流程字段的英文标签 `Workflow:`，意为“工作流程：”。
WORKFLOW_PREFIX = "Workflow:\n"
# 注入记忆时视觉对象和功能字段的英文格式模板：`Object` 意为“对象”，`Function` 意为“功能”。
PATCH_TEMPLATE = "Object: {object}\nFunction: {function}"
