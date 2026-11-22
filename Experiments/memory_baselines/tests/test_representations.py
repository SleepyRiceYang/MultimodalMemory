import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from vlm_memory.memory.bank import MemoryBank
from vlm_memory.memory.representations import build_magnet, crop, entry, maximal_cliques, build_bank, attach_visual


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new('RGB', (100, 100), 'red').save(self.root/'source.png')
        Image.new('RGB', (100, 100), 'blue').save(self.root/'test.png')
        self.episode = dict(id='source',benchmark='vision2code',task='Draw a chart',success=True,
            provenance={'split':'memory'},images=[dict(path='source.png',role='source_observation_0')],
            steps=[{'text':'actual code'}],feedback={'render_success':True})
        self.config = dict(model={},data_mode='unit_test',baselines=['no_memory','raw_multimodal'],benchmarks=['vision2code'])

    def tearDown(self):
        self.temp.cleanup()

    def test_single_success_can_form_visual_memory_but_not_multi_task_workflow(self):
        region=dict(image_index=0,box=[0,0,500,500],object='axis',observation='axis')
        responses=[{'regions':[region]},{'associations':[dict(candidate_id='region_0',function='set axis limits',evidence='observed')]}]
        with patch('vlm_memory.memory.representations.request',side_effect=responses) as call:
            nodes=build_magnet(self.root,self.config,[self.episode])
        self.assertEqual(call.call_count,2)
        self.assertNotIn('actual code',str(call.call_args_list[0]))
        self.assertEqual([n['memory_type'] for n in nodes],['visual_function'])
        with Image.open(self.root/nodes[0]['images'][0]['path']) as image:
            self.assertEqual(image.size,(50,50))
        audit=json.loads((self.root/'vision2code_magnet_maintenance.json').read_text())
        self.assertEqual(audit['clusters'],[])

    def test_failed_and_truncated_sources_do_not_generate_positive_magnet_memories(self):
        for extra in [{'success':False},{'generation_truncated':True}]:
            with patch('vlm_memory.memory.representations.request') as call:
                nodes=build_magnet(self.root,self.config,[{**self.episode,**extra}])
            self.assertEqual(nodes,[])
            call.assert_not_called()

    def test_clustering_requires_pairwise_similarity_not_connected_component(self):
        sim=np.array([[1,.9,.1],[.9,1,.9],[.1,.9,1]])
        self.assertEqual(maximal_cliques(sim,.8),[[0,1],[1,2]])

    def test_workflow_construction_requires_two_distinct_successful_sources(self):
        second={**self.episode,'id':'second'}
        config={**self.config,'retrieval':{'embedding_url':'http://mock'},'memory_construction':{'magnet_cluster_cosine':.75}}
        workflow=dict(title='Plot labeled bars',applicability=['bar charts'],exclusions=[],parameters=['categories','values'],
                      steps=['Read current values','Plot current bars','Set matching labels'],source_ids=['source','second'])
        def response(root,config,name,*args):
            if '_magnet_task_' in name:
                return dict(topic='bar chart',scope='labeled categories',question_anchor='recreate plot')
            if '_magnet_workflows_' in name:
                return {'workflows':[workflow]}
            if '_magnet_regions_' in name:
                return {'regions':[]}
            raise AssertionError(name)
        def endpoint(url,payload):
            return [{'tokens':20}] if url.endswith('/tokens') else {'embeddings':[[1.,0.],[1.,0.]]}
        with patch('vlm_memory.memory.representations.request',side_effect=response),patch('vlm_memory.memory.retrieval.post',side_effect=endpoint):
            nodes=build_magnet(self.root,config,[self.episode,second])
        self.assertEqual(len(nodes),1)
        self.assertEqual(nodes[0]['memory_type'],'workflow')
        self.assertEqual(set(nodes[0]['source_ids']),{'source','second'})

    def test_crop_rejects_zero_area_and_generated_image_as_reference(self):
        with self.assertRaisesRegex(ValueError,'Invalid evidence box'):
            crop(self.root,self.episode,dict(image_index=0,box=[500,0,500,500]),'bad','test')
        generated={**self.episode,'images':[dict(path='source.png',role='render_after_action')]}
        with self.assertRaisesRegex(ValueError,'original observation'):
            crop(self.root,generated,dict(image_index=0,box=[0,0,1000,1000]),'bad','test')

    def test_attachment_cannot_introduce_another_source(self):
        node=entry('k','vision2code',['source'],'fixed knowledge')
        with patch('vlm_memory.memory.representations.request',return_value={'attachments':[
                dict(node_id='k',source_id='unknown',image_index=0,box=[0,0,1000,1000])]}):
            with self.assertRaisesRegex(ValueError,'outside'):
                attach_visual(self.root,self.config,[node],[self.episode])

    def test_paired_selection_reserves_same_image_budget_and_never_changes_text(self):
        plain=[entry('a','vision2code',['source'],'one'),entry('b','vision2code',['source'],'two')]
        visual=copy.deepcopy(plain)
        for e in visual:
            e['images']=[dict(path='source.png',role='evidence')]
        path=self.root/'bank.json'
        path.write_text(json.dumps(dict(sources=[],methods={'ksi_style':plain,'ksi_visual':visual})))
        bank=MemoryBank(path,{'retrieval':{'enabled':True}})
        bank.retriever=SimpleNamespace(rank=lambda *args: ([(1,e) for e in visual],{'retrieval':'dense'}))
        budget=dict(top_k=3,max_images=1,max_text_chars=1000)
        arguments=('vision2code','test','q',self.root/'test.png',budget)
        text=bank.context('ksi_style',*arguments,task={'id':'test'})
        image=bank.context('ksi_visual',*arguments,task={'id':'test'})
        self.assertEqual(text['text'],image['text'])
        self.assertEqual([e['id'] for e in text['selected']],['a'])
        self.assertEqual([e['id'] for e in image['selected']],['a'])
        self.assertEqual(text['images'],[])
        self.assertEqual(len(image['images']),1)

    def test_paired_bank_rejects_changed_claim(self):
        plain=entry('k','vision2code',['source'],'original')
        path=self.root/'bank.json'
        path.write_text(json.dumps(dict(sources=[],methods={'ksi_style':[plain],'ksi_visual':[{**plain,'text':'changed'}]})))
        with self.assertRaisesRegex(ValueError,'content differs'):
            MemoryBank(path)

    def test_no_memory_and_raw_need_no_construction_model(self):
        with patch('vlm_memory.memory.representations.request') as call:
            path=build_bank(self.root,self.config,[self.episode])
        call.assert_not_called()
        data=json.loads(path.read_text())
        self.assertEqual(set(data['methods']),{'raw_multimodal'})
        raw=data['methods']['raw_multimodal'][0]
        self.assertEqual(json.loads(raw['text'])['steps'],self.episode['steps'])
        self.assertEqual(raw['images'],self.episode['images'])

    def test_empty_knowledge_is_valid_not_forced_hallucination(self):
        self.config['baselines']=['ksi_style','ksi_visual']
        with patch('vlm_memory.memory.representations.request',side_effect=[{'claims':[]},{'knowledge':[]}]):
            path=build_bank(self.root,self.config,[self.episode])
        data=json.loads(path.read_text())
        self.assertEqual(data['methods'],{'ksi_style':[],'ksi_visual':[]})

    def test_empty_review_stops_distillation(self):
        self.config['baselines']=['ksi_style','ksi_visual']
        with patch('vlm_memory.memory.representations.request',return_value={'claims':[]}) as call:
            build_bank(self.root,self.config,[self.episode])
        self.assertEqual(call.call_count,1)


if __name__ == '__main__':
    unittest.main()
