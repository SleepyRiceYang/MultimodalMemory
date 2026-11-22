"""Concurrency isolation, deterministic access merge and recoverable barriers."""
import copy
import json
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from PIL import Image

from gui_baseline.cold_memory import DualMemory,empty_memory
from gui_baseline.cold_parallel import assignments,run_round,merge_task_accesses
from gui_baseline.cold_continual import update_memory,check_stop,ordered_model_calls
from gui_baseline.common import read,write

OPTIONS={'procedural_threshold':.4,'stationary_threshold':.7,
         'procedural_top_n':5,'procedural_top_k':2,'stationary_top_n':20,'stationary_top_k':2}


def encoder(texts):return [[0.,1.] if t.startswith('B') else [1.,0.] for t in texts]


class FakeEnv:
    def __init__(self,args):self.args=args
    def api(self,*args,**kwargs):return {}
    def setup(self,task):return task
    def cleanup(self):pass
    def close(self):pass


class ColdParallelTests(TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.args=SimpleNamespace(**OPTIONS,workers=2,build_workers=2,
            android_urls=['http://a','http://b'],android_containers=[self.root.name+'_a',self.root.name+'_b'],
            round_number=0,round_seed=42,stop_event=threading.Event())
        self.bank=DualMemory(empty_memory(),encoder,OPTIONS)
        self.bank.add_workflow('shared',['perform'],0,'source')

    def test_shards_are_disjoint_and_environments_cannot_be_shared(self):
        jobs=assignments(self.args,['A','B','C'])
        self.assertEqual([j['tasks'] for j in jobs],[['A','C'],['B']])
        self.args.android_urls=['http://a','http://a']
        with self.assertRaisesRegex(ValueError,'distinct'):assignments(self.args,['A','B'])

    def test_simultaneous_tasks_use_same_snapshot_and_merge_canonically(self):
        barrier=threading.Barrier(2);started=[];seen=[];guard=threading.Lock()
        snapshot=copy.deepcopy(self.bank.data);folder=self.root/'round'
        class Episode:
            def __init__(obj,args,env,model,bank):obj.bank=bank;obj.env=env
            def run(obj,target,task,goal,limit):
                with guard:started.append(task);seen.append((task,obj.bank.data['counter'],obj.env.args.android_url))
                barrier.wait(timeout=3)
                for step in range(1 if task=='A' else 2):
                    _,audit=obj.bank.select('procedural','shared')
                    write(target/f'step_{step:03d}/procedural_retrieval.json',audit)
                if task=='A':time.sleep(.03)  # reverse completion order
                return {'task_id':task,'success':True,'transitions':[]}
        episodes,merged=run_round(self.args,folder,['A','B'],{'A':1,'B':2},snapshot,encoder,
            env_factory=FakeEnv,model_factory=lambda a:None,episode_factory=Episode)
        self.assertEqual({x[0] for x in seen},{'A','B'})
        self.assertTrue(all(x[1]==0 for x in seen))
        self.assertEqual(len({x[2] for x in seen}),2)
        self.assertEqual([e['task_id'] for e in episodes],['A','B'])
        self.assertEqual(merged.data['counter'],3)
        self.assertEqual(merged.data['procedural'][0]['access_count'],3)
        self.assertEqual(merged.data['procedural'][0]['last_access'],3)
        self.assertEqual(self.bank.data,snapshot)
        # A complete resume must not execute any task or double-count accesses.
        class NoEpisode:
            def __init__(self,*a):raise AssertionError('Committed task was executed twice')
        again,resumed=run_round(self.args,folder,['A','B'],{'A':1,'B':2},snapshot,encoder,
            env_factory=FakeEnv,model_factory=lambda a:None,episode_factory=NoEpisode)
        self.assertEqual(resumed.data,merged.data)
        self.assertEqual(read(folder/'progress.json')['resumed'],2)

    def test_failed_worker_leaves_checkpoints_but_no_round_commit(self):
        folder=self.root/'failure';snapshot=copy.deepcopy(self.bank.data);fail={'B':True}
        class Episode:
            def __init__(obj,args,env,model,bank):obj.bank=bank
            def run(obj,target,task,goal,limit):
                if task=='B' and fail['B']:
                    deadline=time.monotonic()+3
                    while not (folder/'tasks/A/checkpoint.json').exists():
                        if time.monotonic()>deadline:raise AssertionError('A did not commit')
                        time.sleep(.005)
                    raise RuntimeError('injected connection failure')
                _,audit=obj.bank.select('procedural','shared');write(target/'step_000/procedural_retrieval.json',audit)
                return {'task_id':task,'success':True,'transitions':[]}
        kwargs=dict(env_factory=FakeEnv,model_factory=lambda a:None,episode_factory=Episode)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            run_round(self.args,folder,['A','B'],{'A':1,'B':1},snapshot,encoder,**kwargs)
        self.assertTrue((folder/'tasks/A/checkpoint.json').exists())
        self.assertFalse((folder/'summary.json').exists())
        self.assertEqual(self.bank.data,snapshot)
        fail['B']=False;self.args.stop_event.clear()
        _,merged=run_round(self.args,folder,['A','B'],{'A':1,'B':1},snapshot,encoder,**kwargs)
        self.assertEqual(merged.data['procedural'][0]['access_count'],2)
        self.assertEqual(read(folder/'progress.json')['completed'],2)

    def test_visual_access_replay_tracks_chosen_image_independently(self):
        image=self.root/'view.png';Image.new('RGB',(10,10),'red').save(image)
        self.bank.add_visual('shared',image,0,'source')
        local=DualMemory(self.bank.data,encoder,OPTIONS)
        _,trace=local.select('stationary','shared')
        self.bank.replay_accesses(trace)
        self.assertEqual(self.bank.data,local.data)
        node=self.bank.data['stationary'][0]
        self.assertEqual(node['images'][0]['access_count'],1)
        self.assertEqual(node['access_count'],1)
        self.assertEqual(self.bank.data['counter'],2)

    def test_build_calls_overlap_but_memory_ids_follow_source_order(self):
        episodes=[]
        for name,color in [('A','red'),('B','blue')]:
            before=self.root/(name+'.png');after=self.root/(name+'_after.png')
            Image.new('RGB',(20,20),color).save(before);Image.new('RGB',(20,20),'white').save(after)
            episodes.append({'task_id':name,'goal':name,'success':True,'transitions':[{'step':0,
                'action':{'action_type':'click','x':10,'y':10},'description':'click','screen_description':'screen',
                'effect':'changed','before':str(before),'after':str(after)}]})
        barrier=threading.Barrier(2);active=0;maximum=0;lock=threading.Lock();parser_threads=[]
        class Model:
            def call(obj,path,system,user,images=()):
                nonlocal active,maximum
                with lock:active+=1;maximum=max(maximum,active)
                barrier.wait(timeout=3)
                first='workflow_000' in str(path) or '/A/' in str(path)
                time.sleep(.02 if first else .001)
                with lock:active-=1
                if 'workflow' in str(path):
                    return json.dumps({'workflows':[{'title':'A' if first else 'B','actions':['perform']}]})
                return '{"valid":true,"function":"toggle shared","evidence":"changed"}'
        class Parser:
            def crop_click(obj,before,x,y,target):
                parser_threads.append(threading.get_ident());Image.open(before).crop((5,5,15,15)).save(target)
                return [5,5,15,15],[[5,5,15,15]]
        bank=DualMemory(empty_memory(),encoder,OPTIONS)
        self.args.cluster_threshold=.75
        update_memory(self.args,bank,episodes,self.root/'build',Model(),Parser(),0)
        self.assertEqual(maximum,2)
        self.assertEqual(len(set(parser_threads)),1)
        self.assertEqual([n['title'] for n in bank.data['procedural']],['A','B'])
        self.assertEqual([n['id'] for n in bank.data['procedural']],[0,1])
        self.assertEqual([x['source'] for x in bank.data['stationary'][0]['images']],['A','B'])

    def test_build_queue_is_bounded_and_fully_drains_without_false_stop(self):
        consumed=[]
        def jobs():
            for i in range(10):consumed.append(i);yield i
        results=ordered_model_calls(lambda i:i,jobs(),2,self.args)
        self.assertEqual(next(results),0)
        self.assertEqual(len(consumed),2)
        self.assertEqual(list(results),list(range(1,10)))
        self.assertFalse(self.args.stop_event.is_set())

    def test_build_failure_stops_feeding_new_requests(self):
        consumed=[]
        def jobs():
            for i in range(20):consumed.append(i);yield i
        def request(i):
            if i==0:raise RuntimeError('model endpoint failed')
            return i
        with self.assertRaisesRegex(RuntimeError,'endpoint'):
            list(ordered_model_calls(request,jobs(),2,self.args))
        self.assertEqual(len(consumed),2)
        self.assertTrue(self.args.stop_event.is_set())

    def test_stop_prevents_starting_new_task(self):
        self.args.stop_event.set()
        with self.assertRaises(InterruptedError):check_stop(self.args)
        with self.assertRaises(InterruptedError):
            run_round(self.args,self.root/'stopped',['A','B'],{'A':1,'B':1},self.bank.data,encoder,
                env_factory=FakeEnv,model_factory=lambda a:None)
        self.assertFalse(any((self.root/'stopped').glob('tasks/*/checkpoint.json')))
