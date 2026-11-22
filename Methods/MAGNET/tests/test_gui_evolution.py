"""Protocol regressions: isolation, modality pairing, paper ranking and resume rollback."""
import copy,json,math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase,main,mock
import numpy as np
from gui_baseline.evolution_store import EvolutionBank,empty_bank
from gui_baseline.evolution_build import instruction_clusters,episode_from_result,update_from_episodes
from gui_baseline.evolution import run_phase,freeze_date,PlannerOutputError

OPTIONS={'recall_threshold':.5,'top_n':5,'top_k':2,'max_images':4,'max_text_chars':12000,'planner_max_tokens':128}


def candidate(evidence):
    return {'title':'Create a note','app':'Notes','precondition':'Notes is open','goal':'Create <name>',
            'plan':['Enter <name>','Save'],'postcondition':'Note is listed','source_task_ids':['a'],
            'parameters':[],'evidence':evidence}


class EvolutionTests(TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name)
        self.image=self.root/'evidence.png';self.image.write_bytes(b'evidence')
        self.data=empty_bank('androidworld',['a','b'])
        self.bank=EvolutionBank(self.data,'http://unused',OPTIONS)
        self.encoder=mock.patch('gui_baseline.evolution_store.encode',return_value=[[1.,0.]])
        self.encoder.start();self.addCleanup(self.encoder.stop);self.addCleanup(self.tmp.cleanup)
        self.bank.add(candidate([str(self.image)]*2),['bootstrap/a'],0)

    def test_views_share_nodes_text_and_metadata_updates(self):
        text=EvolutionBank(self.bank.data,'unused',OPTIONS);visual=EvolutionBank(self.bank.data,'unused',OPTIONS)
        a=text.select('create note',visual=False,update=True);b=visual.select('create note',visual=True,update=True)
        self.assertEqual(a['selected'],b['selected']);self.assertEqual(a['workflows'],b['workflows'])
        self.assertEqual(a['images'],[]);self.assertEqual(len(b['images']),2)
        self.assertEqual(text.data,visual.data)

    def test_test_retrieval_is_readonly(self):
        old=copy.deepcopy(self.bank.data)
        self.bank.select('create note',visual=True,update=False)
        self.assertEqual(self.bank.data,old)

    def test_retention_before_creation_and_once_per_event(self):
        other=candidate([str(self.image)]*2);other['title']='New workflow';other['plan']=['Different path']
        self.bank.add(other,['bootstrap/b'],0)
        first=self.bank.data['nodes'][0];first.update(access_count=5,last_access=9)
        self.bank.data['counter']=10
        result=self.bank.select('create',visual=False,update=True)
        self.assertEqual(result['selected'][0]['id'],first['id'])
        self.assertEqual(self.bank.data['counter'],11)
        self.assertEqual([n['last_access'] for n in self.bank.data['nodes']],[11,11])

    def test_duplicates_merge_supports_and_preserve_distinct_paths(self):
        _,added=self.bank.add(candidate([str(self.image)]*2),['round1/a'],1)
        self.assertFalse(added);self.assertEqual(len(self.bank.data['nodes']),1)
        self.assertIn('round1/a',self.bank.data['nodes'][0]['supports'])
        other=candidate([str(self.image)]*2);other['plan']=['Use another visible menu']
        self.assertTrue(self.bank.add(other,['round2/a'],2)[1])

    def test_heldout_insertion_is_rejected(self):
        other=candidate([str(self.image)]*2);other['source_task_ids']=['heldout']
        with self.assertRaisesRegex(ValueError,'held-out'):self.bank.add(other,['test'],1)

    def test_cliques_are_not_transitive_components(self):
        angles=[0,math.pi/6,math.pi/3];vectors=[[math.cos(a),math.sin(a)] for a in angles]
        with mock.patch('gui_baseline.evolution_build.encode',return_value=vectors):
            groups=instruction_clusters([{'goal':str(i)} for i in range(3)],'unused',.8)
        self.assertEqual(groups,[[0,1],[1,2]])

    def test_failed_source_never_enters_abstraction(self):
        self.assertIsNone(episode_from_result(self.root,'androidworld','a','goal',{'score':0},'r1/a'))

    def test_oversized_source_is_audited_without_sending_unbounded_prompt(self):
        options={**OPTIONS,'cluster_threshold':.75,'cluster_batch_size':4,'build_max_tokens':128,'max_build_chars':600}
        def episode(task,goal):
            return {'task_id':task,'episode_id':'round1/'+task,'goal':goal,'score':1,
                    'steps':[{'step':0,'action':'click','before':str(self.image),'after':str(self.image)}]}
        model=mock.Mock();model.generate.return_value={'workflows':[]}
        with mock.patch('gui_baseline.evolution_build.instruction_clusters',return_value=[[0,1]]), \
             mock.patch('gui_baseline.evolution_build.JsonModel',return_value=model):
            result=update_from_episodes(self.bank,[episode('a','x'*2000),episode('b','Open note')],
                                        self.root/'build','unused',options,1)
        self.assertEqual(result['oversized_episodes'],['round1/a'])
        self.assertEqual(model.generate.call_count,1)
        payload=model.generate.call_args.args[2]
        self.assertEqual(payload['successful_trajectories'][0]['task_id'],'b')
        self.assertNotIn('before',payload['successful_trajectories'][0]['steps'][0])

    def test_resume_rolls_back_failed_task_retrievals(self):
        failed=set()
        class Adapter:
            def __init__(self,args,policy):self.policy=policy;self.task=None
            def setup(self,task):self.task=task;return 'Goal '+task
            def run(self,*a,**kw):
                self.policy.bank.select('note',visual=False,update=self.policy.update)
                if self.task=='b' and 'b' not in failed:
                    failed.add('b');raise RuntimeError('injected disconnect')
                return {'score':1.,'steps':1}
            def cleanup(self):pass
            def close(self):pass
        args=SimpleNamespace(embedding_url='unused',options=OPTIONS,model_url='http://unused',benchmark='androidworld',
                             root=self.root,max_steps=1,complexities={'a':1,'b':1})
        with mock.patch('gui_baseline.evolution.EvolutionAndroid',Adapter):
            with self.assertRaisesRegex(RuntimeError,'injected disconnect'):
                run_phase(self.root/'phase',args,['a','b'],self.bank.data,'construction','text',1,True)
            final,rows=run_phase(self.root/'phase',args,['a','b'],self.bank.data,'construction','text',1,True)
        self.assertEqual(final.data['counter'],2);self.assertEqual(len(rows),2)

    def test_planner_failure_is_recorded_and_does_not_pollute_next_task(self):
        executed=[]
        class Adapter:
            def __init__(self,args,policy):self.policy=policy
            def setup(self,task):self.task=task;return 'Goal '+task
            def run(self,*a,**kw):
                executed.append(self.task)
                self.policy.bank.select('note',visual=False,update=self.policy.update)
                if self.task=='a':raise PlannerOutputError('bad planner output')
                return {'score':1.,'steps':1}
            def cleanup(self):pass
            def close(self):pass
        args=SimpleNamespace(embedding_url='unused',options=OPTIONS,model_url='http://unused',benchmark='androidworld',
                             root=self.root,max_steps=1,complexities={'a':1,'b':1})
        with mock.patch('gui_baseline.evolution.EvolutionAndroid',Adapter):
            final,rows=run_phase(self.root/'phase',args,['a','b'],self.bank.data,'construction','text',1,True)
            again,rows2=run_phase(self.root/'phase',args,['a','b'],self.bank.data,'construction','text',1,True)
        self.assertEqual(executed,['a','b'])
        self.assertEqual(rows[0]['failure_type'],'planner_invalid_output')
        self.assertEqual(rows[0]['outcome'],'execution_error')
        self.assertEqual(rows[0]['score'],0)
        self.assertEqual(final.data['counter'],1)
        self.assertEqual(again.data,final.data)
        self.assertEqual(rows2,rows)

    def test_connection_error_is_classified_only_if_pool_is_healthy(self):
        import httpx
        from openai import APIConnectionError
        class Adapter:
            def __init__(self,args,policy):self.policy=policy
            def setup(self,task):self.task=task;return 'Goal '+task
            def run(self,*a,**kw):
                self.policy.bank.select('note',visual=False,update=self.policy.update)
                if self.task=='a':raise APIConnectionError(request=httpx.Request('POST','http://model'))
                return {'score':1.,'steps':1}
            def cleanup(self):pass
            def close(self):pass
        args=SimpleNamespace(embedding_url='unused',options=OPTIONS,model_url='http://pool/v1',benchmark='androidworld',
                             root=self.root,max_steps=1,complexities={'a':1,'b':1})
        response=mock.Mock(ok=True);response.json.return_value={'ready':True}
        session=mock.Mock();session.get.return_value=response
        with mock.patch('gui_baseline.evolution.EvolutionAndroid',Adapter), \
             mock.patch('gui_baseline.evolution.requests.Session',return_value=session):
            final,rows=run_phase(self.root/'healthy',args,['a','b'],self.bank.data,'construction','text',1,True)
            self.assertEqual(rows[0]['failure_type'],'model_connection_error')
            self.assertEqual(final.data['counter'],1)
            response.json.return_value={'ready':False}
            with self.assertRaises(APIConnectionError):
                run_phase(self.root/'unhealthy',args,['a','b'],self.bank.data,'construction','text',1,True)
            error=json.loads((self.root/'unhealthy/a/result.json').read_text())
            self.assertEqual(error['failure_type'],'model_service_unavailable')
            self.assertFalse((self.root/'unhealthy/b/result.json').exists())

    def test_dynamic_date_is_frozen(self):
        a='The current date is Tuesday, October 27, 2026.'
        self.assertEqual(freeze_date(a,'Thursday, October 29, 2026'),'The current date is Thursday, October 29, 2026.')


if __name__=='__main__':main()
