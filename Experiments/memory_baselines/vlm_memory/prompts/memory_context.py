"""Shared instructions and labels for retrieved experience across baselines."""

# 共享记忆使用说明：以下为历史经验，来自不同任务，因此可能不适用。
# 应以当前任务的图像和数值为准，不要照搬历史答案或坐标。
MEMORY_NOTICE = (
    'Historical experience follows. It concerns different tasks and may not apply. Use the current '
    "task's image and values; do not copy historical answers or coordinates.\n"
)

# 记忆条目的展示模板：先给出条目 ID，再给出条目正文。
ENTRY_TEMPLATE = "\n[{entry_id}]\n{text}\n"
# 历史图像标签模板：标出编号、记忆条目 ID 和图像角色。
HISTORICAL_IMAGE_LABEL = "Historical image {number}, entry {entry_id}, {role}"
# 来源图像标签模板：标出来源 ID 和图像索引。
SOURCE_IMAGE_LABEL = "Source {source_id} image {index}"
# 当前任务参考图的标签。
CURRENT_IMAGE_LABEL = "CURRENT TASK reference"
# 当前文档页面标签模板，index 为从 0 开始的页面索引。
DOCUMENT_PAGE_LABEL = "CURRENT DOCUMENT PAGE INDEX {index}"
# 技能章节展示模板：章节名称后接对应内容。
SKILL_SECTION_TEMPLATE = "{name}:\n{content}"
