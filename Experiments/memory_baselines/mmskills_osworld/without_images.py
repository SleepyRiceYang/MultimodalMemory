"""MMSkills image ablation: retain runtime cards, selection and live observations."""
from mm_agents.mm_skill_agent import MMSkillAgent
from mm_agents.skill_loader import SkillLoader


class MetadataOnlySkillLoader(SkillLoader):
    def load_selected_state_views(self, skill_name, requested_items, *, runtime=True):
        # Resolve exact state/view identifiers, but never decode reference pixels.
        selections, missing = self.resolve_state_view_requests(
            skill_name, requested_items, runtime=runtime)
        for selection in selections:
            selection.loaded_views = []
        return selections, missing

    def load_single_skill_image(self, *args, **kwargs):
        raise RuntimeError('Reference-image loading is disabled by the W/O Images ablation')


class MMSkillWithoutImagesAgent(MMSkillAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skill_loader = MetadataOnlySkillLoader(
            skills_library_dir=kwargs['skills_library_dir'])

    def _build_selected_state_view_reference_content(self, skill_name, selected_selections):
        content = super()._build_selected_state_view_reference_content(skill_name, selected_selections)
        if content is not None:
            # Preserve selection reasons and card text; fail closed on pixel leakage.
            assert not any('inlineData' in part for part in content['parts'])
            content['parts'].append({'text':
                'Reference images are unavailable in this W/O Images condition. '
                'The selection record describes requested views, not observed reference pixels. '
                'Use the skill text, state descriptions and CURRENT screenshot for planning.'})
        return content
