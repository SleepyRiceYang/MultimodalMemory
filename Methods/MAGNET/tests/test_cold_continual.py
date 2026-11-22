"""Protocol tests: dual memory, functional view updates, actual online feedback."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase,mock
from tempfile import TemporaryDirectory

from PIL import Image

from gui_baseline.cold_memory import DualMemory,empty_memory
from gui_baseline.cold_continual import (ROOT,Episode,action_to_android,parse_json,
                                        task_selection,update_memory,validate_plan,actor_schema)

OPTIONS={'procedural_threshold':.4,'stationary_threshold':.7,
         'procedural_top_n':5,'procedural_top_k':2,'stationary_top_n':20,'stationary_top_k':2}


def encoder(texts):
    return [[0.,1.] if 'other' in t else [1.,0.] for t in texts]


class ColdProtocolTests(TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)
        self.bank=DualMemory(empty_memory(),encoder,OPTIONS)

    def test_empty_memory_stays_empty_on_retrieval(self):
        for kind in ('procedural','stationary'):
            selected,audit=self.bank.select(kind,'search')
            self.assertEqual(selected,[]);self.assertEqual(audit['selected'],[])
        self.assertEqual(self.bank.data['counter'],0)

    def test_function_adds_views_and_deduplicates_pixels(self):
        one=self.path/'one.png';two=self.path/'two.png'
        Image.new('RGB',(8,8),'red').save(one);Image.new('RGB',(8,8),'blue').save(two)
        ident,new=self.bank.add_visual('Search',one,0,'task_a');self.assertTrue(new)
        same,new=self.bank.add_visual('Search',two,1,'task_b');self.assertEqual(ident,same);self.assertTrue(new)
        _,new=self.bank.add_visual('Search',one,2,'task_c');self.assertFalse(new)
        self.bank.add_visual('other function',one,2,'task_c')
        self.assertEqual(len(self.bank.data['stationary']),2)
        self.assertEqual(len(self.bank.data['stationary'][0]['images']),2)
        chosen,_=self.bank.select('stationary','Search')
        self.assertEqual(chosen[0]['image'],str(two.resolve()))
        self.assertEqual(self.bank.data['stationary'][0]['images'][0]['access_count'],0)
        self.assertEqual(self.bank.data['stationary'][0]['images'][1]['access_count'],1)

    def test_retention_then_recency_not_cosine_only(self):
        self.bank.add_workflow('old',['search'],0,'a');self.bank.add_workflow('new',['different path'],1,'b')
        self.bank.data['counter']=10;old=self.bank.data['procedural'][0];old.update(access_count=5,last_access=10)
        nodes,_=self.bank.select('procedural','Search')
        self.assertEqual(nodes[0]['title'],'old')
        self.assertEqual(len(self.bank.data['procedural']),2)

    def test_screenshot_coordinates_and_answer(self):
        result=action_to_android({'ACTION':'click_element','ARGS':{'bbox':[100,200,300,400]}},1080,2400)
        self.assertEqual(result,{'action_type':'click','x':216,'y':720})
        with self.assertRaises(ValueError):action_to_android({'ACTION':'click_element','ARGS':{'bbox':[0,0,1001,9]}},1080,2400)
        self.assertEqual(action_to_android({'ACTION':'answer','ARGS':{'text':'Meeting'}},100,100),{'action_type':'answer','text':'Meeting'})
        self.assertEqual(parse_json('"stop", "ARGS": {"task_status":"success"}}','{"ACTION": '),{'ACTION':'stop','ARGS':{'task_status':'success'}})

    def test_out_of_space_planning_is_rejected_instead_of_reinterpreted(self):
        with self.assertRaisesRegex(ValueError,'invalid'):
            validate_plan({'Thought':'Open shade','Future Action Plan':[{'type':'DRAG','description':'Drag an item'}]})
        self.assertEqual(action_to_android({'ACTION':'swipe','ARGS':{'direction':'down'}},100,100),{'action_type':'swipe','direction':'down'})
        spec=actor_schema('CLICK_ELEMENT')
        self.assertEqual(spec['properties']['ACTION']['enum'],['click_element'])
        self.assertNotIn('drag',spec['properties']['ACTION']['enum'])

    def test_easy61_is_not_fifty_fifty_and_budgets_are_complexity_based(self):
        tasks,budgets=task_selection(SimpleNamespace(tasks=None))
        self.assertEqual(len(tasks),61);self.assertIn('SimpleSmsReply',tasks);self.assertNotIn('RetroCreatePlaylist',tasks)
        self.assertTrue(all(n>=10 for n in budgets.values()))
        self.assertEqual(budgets['AudioRecorderRecordAudio'],12)

    def test_failed_task_click_can_enter_stationary_but_not_workflow(self):
        before=self.path/'before.png';after=self.path/'after.png'
        Image.new('RGB',(20,20),'red').save(before);Image.new('RGB',(20,20),'blue').save(after)
        episode={'task_id':'a','goal':'Open settings','success':False,'transitions':[{'action':{'action_type':'click','x':10,'y':10},
                 'step':0,'before':str(before),'after':str(after)}]}
        parser=mock.Mock()
        def crop(before,x,y,target):
            Image.open(before).crop((5,5,15,15)).save(target);return [5,5,15,15],[[5,5,15,15]]
        parser.crop_click.side_effect=crop
        model=mock.Mock();model.call.return_value='{"valid":true,"function":"Open settings","evidence":"settings page opened"}'
        update_memory(SimpleNamespace(cluster_threshold=.75),self.bank,[episode],self.path/'build',model,parser,0)
        self.assertEqual(self.bank.data['procedural'],[]);self.assertEqual(len(self.bank.data['stationary']),1)

    def test_reflect_reaches_next_actor_not_planner_and_score_has_no_done_gate(self):
        args=SimpleNamespace(ir_tasks=[],settle_seconds=0.)
        class Env:
            params={}
            def screenshot(self,path):Image.new('RGB',(20,20)).save(path);return 20,20
            def api(self,method,path,**kw):return {'score':1.} if path=='/task/score' else {}
        model=mock.Mock()
        plan='{"Thought":"Use settings","Future Action Plan":[{"type":"CLICK_ELEMENT","description":"Open settings"}]}'
        model.call.side_effect=['Home screen',plan,'{"ACTION":"click_element","ARGS":{"bbox":[100,100,200,200]}}',
                               'Settings opened','Settings screen',plan,'{"ACTION":"click_element","ARGS":{"bbox":[200,200,300,300]}}','Next settings page']
        folder=self.path/'episode';folder.mkdir()
        result=Episode(args,Env(),model,self.bank).run(folder,'SystemWifiTurnOn','Turn on Wifi',2)
        self.assertTrue(result['success']);self.assertEqual(result['score'],1.)
        self.assertNotIn('Settings opened',model.call.call_args_list[5].args[2])
        self.assertIn('Settings opened',model.call.call_args_list[6].args[2])
        self.assertEqual(model.call.call_args_list[3].args[3][1][1],folder/'step_000/after.png')
