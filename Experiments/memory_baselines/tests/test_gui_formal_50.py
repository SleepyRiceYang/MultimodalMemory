"""Source-run and frozen-bank gates for the 50/50 GUI scripts."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'Methods/MAGNET/src'))
from gui_baseline.build import source_episodes
from gui_baseline.common import write
from gui_baseline import comparison

class FormalSources(unittest.TestCase):
 def test_os_actions_have_actual_before_after_and_all_50_sources(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'Experiments/memory_baselines/artifacts/gui_magnet_procedural/osworld/source'
   ids=[f'task{i:02d}' for i in range(50)]
   write(root/'config.json',{'benchmark':'osworld','split_role':'memory_construction','tasks':ids})
   for i,task in enumerate(ids):
    parent=root/task;folder=parent/'no_memory';folder.mkdir(parents=True)
    write(parent/'task.json',{'goal':'Goal '+task});write(folder/'result.json',{'status':'completed','score':float(i%2)})
   task=ids[0];folder=root/task/'no_memory'
   for name,color in [('screen_000.png','white'),('action_000_after.png','black'),('action_001_after.png','red')]:
    Image.new('RGB',(8,8),color).save(folder/name)
   write(folder/'action_000.json',{'action':'pyautogui.click(1, 2)'})
   write(folder/'action_001.json',{'action':'pyautogui.press("enter")'})
   episodes=source_episodes(temp,'osworld','source',ids)
   self.assertEqual(len(episodes),50)
   self.assertEqual([x['step'] for x in episodes[0]['steps']],[0,1])
   self.assertTrue(episodes[0]['steps'][0]['before'].endswith('screen_000.png'))
   self.assertTrue(episodes[0]['steps'][1]['after'].endswith('action_001_after.png'))
   with self.assertRaises(ValueError):source_episodes(temp,'osworld','source',ids[:-1]+['unknown'])
 def test_android_source_actions_exclude_termination(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'Experiments/memory_baselines/artifacts/gui_magnet_procedural/androidworld/source'
   ids=[f'Task{i:02d}' for i in range(50)]
   write(root/'config.json',{'benchmark':'androidworld','split_role':'memory_construction','tasks':ids})
   for i,task in enumerate(ids):
    parent=root/task;folder=parent/'no_memory';folder.mkdir(parents=True)
    write(parent/'task.json',{'goal':'Goal '+task})
    write(folder/'result.json',{'status':'completed','score':float(i%2),'task_state_score':float(i%2)})
   folder=root/ids[0]/'no_memory'
   for i,action in enumerate([{'action_type':'open_app','app_name':'Clock'},
                               {'action_type':'click','x':1,'y':2},
                               {'action_type':'status','goal_status':'complete'}]):
    step=folder/f'step_{i:03d}';step.mkdir()
    Image.new('RGB',(8,8),'white').save(step/'screen.png');write(step/'action.json',action)
   Image.new('RGB',(8,8),'white').save(folder/'final.png')
   episodes=source_episodes(temp,'androidworld','source',ids)
   self.assertEqual(len(episodes),50)
   self.assertEqual([x['step'] for x in episodes[0]['steps']],[0,1])
   self.assertTrue(episodes[0]['steps'][-1]['after'].endswith('step_002/screen.png'))
 def test_wrong_bank_source_ids_rejected_before_environment(self):
  split=Path(__file__).resolve().parents[1]/'splits/gui_50_50_seed20261026/androidworld.json'
  with tempfile.TemporaryDirectory() as temp:
   class WrongBank:
    def __init__(self,*args):self.data={'source_task_ids':['an old pilot task']}
   with patch.object(comparison,'ProceduralBank',WrongBank),patch.object(comparison,'Android',side_effect=AssertionError('Environment must not start')):
    with self.assertRaisesRegex(ValueError,'construction IDs'):
     comparison.main(['--benchmark','androidworld','--condition','magnet_procedural_text',
       '--workspace',temp,'--split-file',str(split),'--split-role','test',
       '--bank-dir',str(Path(temp)/'bank'),'--source-run','formal_50_construction_v1'])
   correct_ids=json.loads(split.read_text())['memory_construction']
   class WrongRun:
    def __init__(self,*args):self.data={'source_task_ids':correct_ids,'source_run':'old_diagnostic_run'}
   with patch.object(comparison,'ProceduralBank',WrongRun),patch.object(comparison,'Android',side_effect=AssertionError('Environment must not start')):
    with self.assertRaisesRegex(ValueError,'source run'):
     comparison.main(['--benchmark','androidworld','--condition','magnet_procedural_text',
       '--workspace',temp,'--split-file',str(split),'--split-role','test',
       '--bank-dir',str(Path(temp)/'bank'),'--source-run','formal_50_construction_v1'])

if __name__=='__main__':unittest.main()
