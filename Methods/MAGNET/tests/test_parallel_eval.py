"""Partition coverage and cached vectors must preserve ordered retrieval updates."""
from unittest import TestCase,mock
from gui_baseline.parallel_eval import shards
from gui_baseline.evolution_store import EvolutionBank,empty_bank

class ParallelTests(TestCase):
    def test_disjoint_complete_shards(self):
        tasks=['a','b','c','d','e'];groups=shards(tasks,2)
        self.assertEqual(groups,[['a','c','e'],['b','d']])
        self.assertEqual(sorted(sum(groups,[])),sorted(tasks))
        with self.assertRaises(ValueError):shards(['a','a'],2)
        with self.assertRaises(ValueError):shards(tasks,0)

    def test_cached_query_still_updates_every_access(self):
        options={'recall_threshold':.5,'top_n':1,'top_k':1,'max_images':2,'max_text_chars':12000}
        data=empty_bank('androidworld',['a'])
        data['nodes']=[dict(id='n',title='Wifi',app='Settings',precondition='Open',goal='Wifi',plan=['Open'],
                            postcondition='Done',source_task_ids=['a'],evidence=[],embedding=[1.,0.],
                            created_order=0,access_count=0,last_access=0)]
        bank=EvolutionBank(data,'unused',options)
        with mock.patch('gui_baseline.evolution_store.encode',return_value=[[1.,0.]]) as encode:
            first=bank.select('Wifi',visual=False,update=True)
            second=bank.select('Wifi',visual=False,update=True)
            self.assertEqual(encode.call_count,1)
        self.assertEqual([r['id'] for r in first['selected']],[r['id'] for r in second['selected']])
        self.assertNotEqual(first['selected'][0]['retention'],second['selected'][0]['retention'])
        self.assertEqual(bank.data['nodes'][0]['access_count'],2)
        self.assertEqual(bank.data['counter'],2)
