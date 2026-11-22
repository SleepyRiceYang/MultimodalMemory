"""Representation isolation, evidence validity, and frozen held-out memory checks."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'Methods/MAGNET/src'))
from gui_baseline.memory import ProceduralBank,validate_node,parameterize,payload_text
from gui_baseline.runner import memory_messages,redact,normalize_android_action
from gui_baseline.common import write

class ProceduralTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  for name in ['before.png','after.png']:Image.new('RGB',(20,20),'white').save(self.root/name)
  self.node={'id':'n1','title':'Create draft Alice','app':'Contacts','precondition':'Home','goal':'Create Alice draft',
    'parameters':[{'name':'contact_name','value':'Alice'}],
    'steps':[{'source_step':0,'instruction':'Open Contacts','expected_state':'Contacts visible'},
             {'source_step':1,'instruction':'Type Alice','expected_state':'Alice visible'}],
    'postcondition':'Alice draft visible','evidence':['before.png','after.png'],'source_task_id':'source'}
  self.source={'goal':'Create Alice draft','steps':[{'step':0,'action':{'open':'Contacts'},'before':str(self.root/'before.png'),'after':str(self.root/'after.png')},
    {'step':1,'action':{'type':'Alice'},'before':str(self.root/'before.png'),'after':str(self.root/'after.png')}]}
 def bank(self):
  write(self.root/'bank.json',{'benchmark':'androidworld','stationary_memory':False,'source_task_ids':['source'],'nodes':[self.node]})
  write(self.root/'index.json',{'texts':['Create draft']});np.save(self.root/'vectors.npy',np.array([[1,0]],dtype=np.float32))
  return ProceduralBank(self.root)
 def test_actual_steps_and_parameterization(self):
  validate_node(self.node,self.source)
  candidate=copy.deepcopy(self.node);candidate['steps'][1]['source_step']=3
  with self.assertRaises(ValueError):validate_node(candidate,self.source)
  value=payload_text(parameterize(self.node,self.source));self.assertNotIn('Alice',value);self.assertIn('<contact_name>',value)
 def test_no_cross_benchmark_or_source_leak(self):
  bank=self.bank()
  with self.assertRaises(ValueError):bank.select('osworld','test','goal')
  with self.assertRaises(ValueError):bank.select('androidworld','source','goal')
 def test_pairing_and_reserved_image_budget(self):
  bank=self.bank()
  with patch('gui_baseline.memory.encode',return_value=[[1,0]]):
   ctx=bank.select('androidworld','test','goal');self.assertEqual(len(ctx['selected']),1)
   self.assertEqual(bank.select('androidworld','test','goal',max_images=1)['selected'],[])
  text=bank.view(ctx,False);visual=bank.view(ctx,True)
  self.assertEqual(text['text'],visual['text']);self.assertEqual(text['selected'],visual['selected']);self.assertEqual(text['images'],[])
  messages=[{'role':'system','content':'native system'},{'role':'user','content':'CURRENT task'}]
  def strip_images(ms):
   return [{**m,'content':[b for b in m['content'] if b['type']!='image_url']} if isinstance(m['content'],list) else m for m in ms]
  self.assertEqual(strip_images(memory_messages(messages,text)),strip_images(memory_messages(messages,visual)))
  self.assertEqual(messages[0]['content'],'native system')
 def test_freeze_includes_evidence(self):
  bank=self.bank();self.assertTrue(bank.verify());(self.root/'after.png').write_bytes(b'changed')
  with self.assertRaises(ValueError):bank.verify()
 def test_termination_alias_is_unambiguous(self):
  self.assertEqual(normalize_android_action({'reason':'done','action_type':'status','status':'complete'}),{'action_type':'status','goal_status':'complete'})
  with self.assertRaises(ValueError):normalize_android_action({'action_type':'status','status':'complete','goal_status':'infeasible'})
 def test_raw_coordinate_payload_not_exposed(self):
  self.node['source_actions']=[{'x':123,'y':456}];value=payload_text(self.node)
  self.assertNotIn('source_actions',value);self.assertNotIn('source_step',value);self.assertNotIn('before.png',value)

if __name__=='__main__':unittest.main()

class ComparisonTests(unittest.TestCase):
 def test_no_memory_has_no_bank_or_injected_turn(self):
  from gui_baseline import comparison
  from gui_baseline.runner import memory_messages
  original=[{'role':'system','content':'native actor'},{'role':'user','content':'current screen'}]
  self.assertEqual(memory_messages(original,comparison.EMPTY),original)
  with tempfile.TemporaryDirectory() as temp:
   class FakeAndroid:
    def __init__(self,args):self.args=args;self.calls=0
    def setup(self,task):self.calls+=1;return 'public task goal'
    def run(self,folder,goal,context,max_steps=None):
     assert context==comparison.EMPTY
     return {'score':1.0,'steps':1,'task_state_score':1.0,'agent_done':True}
    def cleanup(self):pass
    def close(self):pass
   with patch.object(comparison,'AW',['task_a']),patch.object(comparison,'Android',FakeAndroid),patch.object(comparison,'ProceduralBank',side_effect=AssertionError('No Memory loaded a bank')):
    argv=['--benchmark','androidworld','--condition','no_memory','--workspace',temp]
    comparison.main(argv)
    comparison.main(argv)  # completed condition is resumed, not rerun
   root=Path(temp)/'Experiments/memory_baselines/artifacts/gui_magnet_procedural/androidworld/comparison_5_v1'
   self.assertEqual(len(json.loads((root/'summary.json').read_text())),1)
   self.assertEqual(json.loads((root/'task_a/no_memory/memory.json').read_text())['images'],[])

class PairedComparisonTests(unittest.TestCase):
 def test_independent_condition_commands_share_one_retrieval(self):
  from gui_baseline import comparison
  with tempfile.TemporaryDirectory() as temp:
   calls=[]
   class FakeAndroid:
    def __init__(self,args):pass
    def setup(self,task):return 'the identical public goal'
    def run(self,folder,goal,context,max_steps=None):return {'score':0.0,'steps':1,'task_state_score':0.0,'agent_done':True}
    def cleanup(self):pass
    def close(self):pass
   class FakeBank:
    def __init__(self,*args):pass
    def select(self,*args):
     calls.append('retrieved')
     return {'text':'shared workflow','images':['before.png','after.png'],'selected':[{'id':'node'}],
             'candidates':[],'ranking':'cosine','stationary_memory':False}
    def view(self,context,visual):
     result=copy.deepcopy(context)
     if not visual:result['images']=[]
     return result
    def verify(self):return True
   with patch.object(comparison,'AW',['task_a']),patch.object(comparison,'Android',FakeAndroid),patch.object(comparison,'ProceduralBank',FakeBank):
    args=['--benchmark','androidworld','--workspace',temp]
    comparison.main(args+['--condition','magnet_procedural_text'])
    comparison.main(args+['--condition','magnet_procedural_visual'])
   root=Path(temp)/'Experiments/memory_baselines/artifacts/gui_magnet_procedural/androidworld/comparison_5_v1/task_a'
   t=json.loads((root/'magnet_procedural_text/memory.json').read_text())
   v=json.loads((root/'magnet_procedural_visual/memory.json').read_text())
   self.assertEqual(calls,['retrieved']);self.assertEqual(t['text'],v['text'])
   self.assertEqual(t['selected'],v['selected']);self.assertEqual(t['images'],[])
   self.assertEqual(v['images'],['before.png','after.png'])
