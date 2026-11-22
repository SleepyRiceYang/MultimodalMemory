import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gui_baseline.cold_continual import Model, ROOT
from gui_baseline.cold_memory import DualMemory
from gui_baseline.evidence_eval_v3 import MultiStateModel, SubtaskEpisode, resolve_selection


class EvidenceV3Test(unittest.TestCase):
    def setUp(self):
        self.state={'state_id':'procedural_1/create','precondition':'Contacts list', 'goal':'Create contact',
                    'procedure':['Open contact form'], 'precondition_embedding':[1.,0.], 'goal_embedding':[1.,0.],
                    'available_views':[{'view_id':'v1','view_type':'focus_crop','path':'crop.png'},
                                       {'view_id':'v2','view_type':'after','path':'after.png'}]}
        self.manifest=[{'id':1,'states':[self.state]}]
        self.decision={'visual_reference_needed':True,'requests':[{'memory_id':1,'state_id':self.state['state_id'],
            'views':['focus_crop'],'evidence_goal':'locate_control','reason':'target ambiguity'}],'view_ids':['v1']}

    def test_gate_rejects_false_with_views_wrong_state_and_wrong_goal(self):
        resolve_selection(self.decision,self.manifest)
        for field,value in [('visual_reference_needed',False),('view_ids',[])]:
            bad=copy.deepcopy(self.decision);bad[field]=value
            with self.assertRaises(ValueError):resolve_selection(bad,self.manifest)
        for field,value in [('state_id','not-retrieved'),('evidence_goal','verify_after')]:
            bad=copy.deepcopy(self.decision);bad['requests'][0][field]=value
            with self.assertRaises(ValueError):resolve_selection(bad,self.manifest)
        resolve_selection({'visual_reference_needed':False,'requests':[],'view_ids':[]},self.manifest)

    def test_planner_b2_b3_gate_only_changes_procedural_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'procedural_retrieval.json').write_text(json.dumps({'query':{'goal':'Create contact'},'selected':[{'id':1,'state_ids':[self.state['state_id']]}]}))
            (root/'stationary_retrieval.json').write_text(json.dumps({'selected':[{'id':1}]}))
            calls=[]
            def call(model,folder,system,user,images=(),prefix=None,*args,**kwargs):
                calls.append({'role':Path(folder).name,'user':user,'images':images,'prefix':prefix})
                return json.dumps(self.decision) if Path(folder).name.endswith('_selection') else '{}'
            for condition in ['B2','B3']:
                model=object.__new__(MultiStateModel)
                model.args=SimpleNamespace(condition=condition)
                model.memory={'procedural':[{'id':1,'title':'create contact','states':[self.state]}]}
                with patch.object(Model,'call',call):
                    model.call(root/'planner','system','task',[('CURRENT','live.png')])
                self.assertEqual(calls[-1]['images'],[('CURRENT','live.png')]+([('HISTORICAL focus_crop','crop.png')] if condition=='B3' else []))
                self.assertIn('selected_state_cards',calls[-1]['user'])

    def test_actor_preserves_original_stationary_images_and_prompt_in_both_conditions(self):
        for condition in ['B2','B3']:
            model=object.__new__(MultiStateModel)
            model.args=SimpleNamespace(condition=condition)
            images=[('CURRENT','live.png'),('HISTORICAL function: open menu','original_patch.png')]
            with patch.object(Model,'call',return_value='{}') as call:
                model.call(Path('step_000/actor'),'original system','original user',images,prefix='original prefix')
            call.assert_called_once_with(Path('step_000/actor'),'original system','original user',images,'original prefix',True,None,None)

    def test_subtask_query_rejects_wrong_direction_and_passes_segment_only(self):
        bank=DualMemory({'counter':0,'procedural':[
            {'id':1,'title':'Whole contact workflow','states':[self.state],'access_count':0,'last_access':0},
            {'id':2,'title':'Delete contact','states':[{**self.state,'state_id':'delete','goal_embedding':[-1.,0.]}],'access_count':0,'last_access':0}]},
            lambda texts:[[1.,0.] for _ in texts],{})
        class FakeModel:
            def call(self,folder,system,user,images=(),**kwargs):
                if Path(folder).name=='subtask':
                    return json.dumps({'precondition':'Contacts list','goal':'Create contact','completion_cue':'form opens','mode':'execute','reason':'visible'})
                payload=json.loads(user)
                assert len(payload['candidates'])==1
                return json.dumps({'decisions':[{'state_id':'procedural_1/create','applicable':True,'reason':'matches'}]})
        args=SimpleNamespace(subtask_pre_threshold=.35,subtask_goal_threshold=.4,subtask_top_k=5)
        episode=SubtaskEpisode(args,None,FakeModel(),bank)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'step_000';folder.mkdir()
            workflows,trace=episode.retrieve_procedural(folder,'Create a new contact','list',[],None,'live.png')
            self.assertEqual(trace['query']['goal'],'Create contact')
            self.assertEqual([w['id'] for w in workflows],[1])
            self.assertEqual(workflows[0]['actions'][0]['steps'],['Open contact form'])
            self.assertEqual(bank.data['procedural'][1]['access_count'],0)

    def test_renderer_coordinate_spaces_and_escaping(self):
        spec=importlib.util.spec_from_file_location('renderer',ROOT/'.agents/skills/gui-agent-trajectory/scripts/render.py')
        renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
        image={'path':'/tmp/screen.png','width':1080,'height':2400}
        result=renderer.screenshot(image,Path('/tmp'),'screen',{'type':'click','coordinate_space':'normalized_1000','bbox':[100,200,300,400]})
        self.assertIn('cx="216.0" cy="720.0"',result)
        self.assertEqual(renderer.pretty('<script>'),'&lt;script&gt;')

    def test_report_preserves_unstarted_task_and_missing_condition(self):
        from gui_baseline.evidence_report_v3 import normalize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'B2').mkdir()
            (root/'B2/config.json').write_text(json.dumps({'tasks':['PendingTask']}))
            data=normalize(root)
            self.assertEqual(len(data['trajectories']),1)
            row=data['trajectories'][0]
            self.assertEqual(row['condition'],'B2')
            self.assertEqual(row['status'],'not_started')
            self.assertIsNone(row['score'])


if __name__=='__main__':unittest.main()
