import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import run
sys.path.insert(0, str(run.OSWORLD))
from without_images import MetadataOnlySkillLoader

SKILL = 'VLC_Load_Subtitles_And_Configure_Subtitle_Presentation'


class WithoutImagesTests(unittest.TestCase):
    def test_retains_cards_and_selection_without_loading_pixels(self):
        loader = MetadataOnlySkillLoader(str(run.ROOT / 'Datasets/MMSkills/ubuntu'))
        bundles = loader.load_state_bundles(SKILL, runtime=True)
        self.assertGreater(len(bundles.bundles), 1)
        text = loader.format_state_bundles_for_branch(bundles)
        for state in bundles.bundles:
            self.assertIn(state.state_id, text)
        request = [{'state_id': bundles.bundles[0].state_id, 'views': ['focus_crop'], 'reason': 'test'}]
        with patch.object(loader, 'load_single_skill_image', side_effect=AssertionError('pixel read')):
            selections, missing = loader.load_selected_state_views(SKILL, request)
        self.assertFalse(missing)
        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].loaded_views, [])
        self.assertEqual(loader.load_selected_state_views(SKILL, []), ([], []))

    def test_planner_keeps_live_screenshot_and_all_card_text(self):
        args = dict(model='qwen3-vl-8b-instruct', base_url='http://localhost:8170/v1',
                    max_tokens=65536, request_timeout=300,
                    skills_dir=str(run.ROOT / 'Datasets/MMSkills/ubuntu'))
        agent = run.make_agent('visual_skill_without_images', args)
        agent.set_task_skills([SKILL])
        bundle, _, error = agent._load_skill_for_branch(SKILL, set())
        self.assertIsNone(error)
        states = bundle['runtime_state_bundles'].bundles
        selections, missing = agent._skill_loader.load_selected_state_views(SKILL,
            [{'state_id': states[0].state_id, 'views': ['focus_crop'], 'reason': 'test'}])
        contents = agent._build_branch_planner_contents(
            instruction='Configure subtitles', trigger_skill_name=SKILL,
            main_trigger_response='LOAD_SKILL', processed_image='LIVE_SCREENSHOT',
            original_width=1920, original_height=1080, processed_width=1920,
            processed_height=1080, obs={}, branch_skill_payloads=[(SKILL, bundle)],
            selected_selections=selections)
        import json
        serialized = json.dumps(contents)
        for state in states:
            self.assertIn(state.state_id, serialized)
        images = [p['inlineData']['data'] for c in contents for p in c['parts'] if 'inlineData' in p]
        self.assertEqual(images, ['LIVE_SCREENSHOT'])
        normal = run.make_agent('visual_skill', args)
        normal.set_task_skills([SKILL])
        normal._load_skill_for_branch(SKILL, set())
        self.assertEqual(agent._build_main_system_message('test'), normal._build_main_system_message('test'))


if __name__ == '__main__':
    unittest.main()
