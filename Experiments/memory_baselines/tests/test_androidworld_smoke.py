import unittest

from vlm_memory.pipeline.androidworld_smoke import validate_action


class AndroidActionTests(unittest.TestCase):
    def test_coordinates_scale_and_clip_to_device_pixels(self):
        self.assertEqual(validate_action({'reason':'tap', 'action_type':'click', 'x':1000, 'y':500},1080,2400),
                         {'action_type':'click', 'x':1079, 'y':1200})

    def test_rejects_unknown_actions_and_malformed_coordinates(self):
        for value in [{'action_type':'shell', 'text':'ignored'},
                      {'action_type':'click', 'x':1001, 'y':0},
                      {'action_type':'click', 'x':True, 'y':0},
                      {'action_type':'click', 'x':0},
                      {'action_type':'scroll', 'direction':'diagonal'},
                      {'action_type':'status', 'goal_status':'maybe'}]:
            with self.assertRaises(ValueError):
                validate_action(value,1080,2400)

    def test_native_text_and_app_actions_are_preserved(self):
        for value in [{'action_type':'input_text', 'text':'A note'},
                      {'action_type':'open_app', 'app_name':'Settings'},
                      {'action_type':'status', 'goal_status':'complete'}]:
            self.assertEqual(validate_action(value,1080,2400),value)
