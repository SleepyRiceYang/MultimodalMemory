import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from vlm_memory.memory.retrieval import DenseRetriever, validate_ranks, index_text, memory_scope, bounded_rank_schema
from vlm_memory.memory.bank import MemoryBank
from PIL import Image


def assessment(node_id, grade, conflict=False, duplicate=''):
    return dict(id=node_id, grade=grade, scope_conflict=conflict, duplicate_of=duplicate, reason='evidence', condition_evidence='visible')


class DenseRetrievalTests(unittest.TestCase):
    def test_cosine_ranking_skips_llm_and_orders_ties_by_id(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = DenseRetriever.__new__(DenseRetriever)
            engine.root = Path(directory)
            engine.url = 'http://unused'
            engine.options = {'top_n': 20, 'ranking': 'embedding_cosine'}
            engine.encoder = {'model': 'fake'}
            entries = [{'id': key, 'benchmark': benchmark, 'text': key}
                       for key, benchmark in [('b','vision2code'), ('a','vision2code'), ('hidden','mathvision')]]
            engine.bank = SimpleNamespace(data={'methods': {'ksi_style': entries}})
            engine.nodes = [dict(e, fields={}) for e in entries]
            engine.vectors = np.array([[1.,0.], [1.,0.], [1.,0.]])
            engine.describe = Mock(return_value={'text':'visible question'})
            engine.generate = Mock(side_effect=AssertionError('Reranker must never be called'))
            with patch('vlm_memory.memory.retrieval.post', return_value={'embeddings':[[1.,0.]]}) as post:
                ranked, trace = engine.rank('ksi_style', {'id':'t','benchmark':'vision2code'})
            self.assertEqual([entry['id'] for _, entry in ranked], ['a','b'])
            self.assertIsNone(trace['reranker'])
            self.assertEqual(post.call_count, 1)
            engine.generate.assert_not_called()

    def test_bounded_rerank_schema_limits_candidates_and_explanations(self):
        schema = bounded_rank_schema(['a','b'])['properties']['assessments']
        self.assertEqual((schema['minItems'],schema['maxItems']),(2,2))
        props = schema['items']['properties']
        self.assertEqual(props['id']['enum'],['a','b'])
        self.assertEqual(props['duplicate_of']['enum'],['','a','b'])
        self.assertEqual(props['reason']['maxLength'],320)

    def test_scope_preserves_explicit_conditions_without_promoting_source_values(self):
        knowledge = {'claim':'Use current labels', 'conditions':['bars with categories'], 'exclusions':['unlabeled bars']}
        scope = memory_scope({'text':json.dumps(knowledge)})
        self.assertEqual(json.loads(scope),{'conditions':knowledge['conditions'],'exclusions':knowledge['exclusions']})
        scope = memory_scope({'text':json.dumps({'task':'Revenue 2010 to 2020','steps':['plot the chart']})})
        self.assertNotIn('2010',scope)
        self.assertIn('illustrative',scope)

    def test_rank_contract_rejects_missing_unknown_duplicate_ids(self):
        for rows in [[assessment('a', 2)], [assessment('a',2),assessment('x',2)], [assessment('a',2),assessment('a',3)]]:
            with self.assertRaises(ValueError):
                validate_ranks(rows, ['a','b'])
        with self.assertRaises(ValueError):
            validate_ranks([assessment('a',2,duplicate='a')], ['a'])

    def test_semantic_rerank_scope_duplicate_and_benchmark_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = DenseRetriever.__new__(DenseRetriever)
            engine.root = Path(directory)
            engine.url = 'http://unused'
            engine.options = {'top_n':20}
            engine.encoder = {'model':'fake'}
            engine.config = {'model':{'name':'fake'}}
            entries = [{'id':key,'benchmark':'vision2code','text':key} for key in ['a','b','c','d']]
            entries.append({'id':'other','benchmark':'mathvision','text':'hidden'})
            engine.bank = SimpleNamespace(data={'methods':{'ksi_style':entries}})
            engine.nodes = [{'id':e['id'],'benchmark':e['benchmark'],'fields':{'topic':e['text']}} for e in entries]
            engine.vectors = np.array([[1,0],[.7,.7],[.9,.1],[.6,.8],[1,0]])
            engine.describe = Mock(return_value={'text':'current visible task'})
            engine.generate = Mock(return_value={'assessments':[assessment('a',1),assessment('b',3),assessment('c',3,True),assessment('d',2,duplicate='b')]})
            def post(url,payload):
                return {'embeddings':[[1,0]]} if url.endswith('/encode') else [{'text':payload['texts'][0]}]
            with patch('vlm_memory.memory.retrieval.post',side_effect=post):
                ranked,trace = engine.rank('ksi_style',{'id':'t','benchmark':'vision2code','prompt':'question','images':[]})
            self.assertEqual([e['id'] for _,e in ranked], ['b'])
            self.assertEqual(len(trace['candidates']),4)
            self.assertNotIn('hidden',engine.generate.call_args.args[2]['question'])

    def test_dense_budget_skips_whole_multimodal_entry_and_fills_next(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB',(8,8),'red').save(root/'current.png')
            Image.new('RGB',(8,8),'blue').save(root/'old.png')
            entries = [{'id':'big','benchmark':'vision2code','source_ids':['s'],'text':'advice','images':[{'path':'old.png','role':'evidence'}]},
                       {'id':'small','benchmark':'vision2code','source_ids':['s'],'text':'usable','images':[]}]
            (root/'bank.json').write_text(json.dumps({'sources':[{'id':'s','benchmark':'vision2code','image_hashes':[]}],'methods':{'ksi_style':entries}}))
            bank = MemoryBank(root/'bank.json', {'retrieval':{'enabled':True}})
            bank.retriever = SimpleNamespace(rank=Mock(return_value=([(1,entries[0]),(.5,entries[1])],{'retrieval':'dense'})))
            context = bank.context('ksi_style','vision2code','t','q',root/'current.png',{'top_k':1,'max_images':0,'max_text_chars':5000},task={'id':'t'})
            self.assertEqual([e['id'] for e in context['selected']],['small'])
            self.assertEqual(context['images'],[])
            self.assertTrue(context['truncated'])

    def test_long_query_failure_returns_empty_without_reading_answer(self):
        engine = DenseRetriever.__new__(DenseRetriever)
        with tempfile.TemporaryDirectory() as directory:
            engine.root=Path(directory)
            engine.options={}
            engine.generate=Mock(side_effect=ValueError('invalid query output'))
            engine.count=Mock(return_value=300)
            result=engine.describe({'id':'test','benchmark':'mathvision','prompt':'long input','images':[], 'answer':'SECRET'})
            self.assertEqual(result['text'],'')
            self.assertTrue(result['query_rewrite_failed'])
            self.assertNotIn('SECRET',str(engine.generate.call_args))

    def test_frozen_index_and_vectors_reject_changes_without_extra_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = DenseRetriever.__new__(DenseRetriever)
            engine.root = Path(directory)
            engine.frozen_index = {'nodes': ['fixed']}
            engine.vectors = np.array([[1., 0.]], dtype=np.float32)
            (engine.root/'index.json').write_text(json.dumps(engine.frozen_index))
            np.save(engine.root/'vectors.npy', engine.vectors)
            self.assertTrue(engine.verify_frozen())
            np.save(engine.root/'vectors.npy', np.array([[0., 1.]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'vectors changed'):
                engine.verify_frozen()
            np.save(engine.root/'vectors.npy', engine.vectors)
            (engine.root/'index.json').write_text(json.dumps({'nodes':['new']}))
            with self.assertRaisesRegex(ValueError, 'index changed'):
                engine.verify_frozen()

    def test_three_index_fields_keep_scope_explicit(self):
        self.assertEqual(index_text({'topic':'chart','scope':'only logarithmic axes','question_anchor':'plot'}), 'Topic: chart\nScope: only logarithmic axes\nQuestion Anchor: plot')

if __name__ == '__main__':
    unittest.main()
