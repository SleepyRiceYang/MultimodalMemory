import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from vlm_memory.memory.magnet_v2 import Builder,complete_link,support_error,evidence_anchors,parameterize,inspected_region_supported


class MagnetV2Tests(unittest.TestCase):
    def test_complete_link_does_not_join_a_similarity_chain_or_repeat_sources(self):
        matrix=np.array([[1,.9,.1],[.9,1,.9],[.1,.9,1]])
        groups=complete_link(matrix,['a','b','c'],.8)
        self.assertEqual(groups,[[0,1],[2]])
        self.assertEqual(sorted(i for g in groups for i in g),[0,1,2])

    def test_cluster_size_is_bounded_even_for_a_dense_graph(self):
        groups=complete_link(np.ones((19,19)),[str(i) for i in range(19)],.8,max_size=8)
        self.assertEqual(sum(map(len,groups)),19)
        self.assertTrue(all(len(g)<=8 for g in groups))

    def test_source_ids_are_not_evidence_without_distinct_ordered_operations(self):
        records={s:dict(operations=[{}, {}, {}]) for s in ['a','b']}
        w=dict(source_ids=['a','a'],steps=['Read supported task structure','Transform the observed components','Check the resulting structure'],support=[])
        self.assertIsNotNone(support_error(w,records))
        w['source_ids']=['a','b']
        w['support']=[dict(source_id=s,step_index=i,operation_index=i) for s in records for i in range(3)]
        self.assertIsNone(support_error(w,records))
        w['support'][-1]['operation_index']=0
        self.assertIsNotNone(support_error(w,records))

    def test_evidence_is_a_real_fragment_and_parameterization_preserves_grid_structure(self):
        steps=[dict(text='Set x = 5.\n\nCompute the area exactly.\n')]
        anchors=evidence_anchors(steps)
        self.assertTrue(anchors)
        for a in anchors:self.assertIn(a['quote'],steps[a['step_index']]['text'])
        text=parameterize('A 3x3 grid with year 2010 and value 23.4')
        self.assertIn('3x3',text)
        self.assertNotIn('2010',text)
        self.assertNotIn('23.4',text)

    def test_dedup_checks_all_members_and_does_not_use_transitive_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            builder=Builder.__new__(Builder)
            builder.root=Path(directory)
            candidates=[dict(candidate_id=k,steps=['operation']) for k in ['a','b','c']]
            builder.candidate_descriptions=Mock(return_value={k:dict(problem_description='same task',procedure_description='method') for k in ['a','b','c']})
            builder.encode=Mock(return_value=np.ones((3,2)))
            def call(name,system,payload,*args):
                new=payload['new_candidate']['candidate_id']
                return dict(assessments=[dict(id=c['candidate_id'],relation=('alternative_strategy' if (new,c['candidate_id'])==('c','a') else 'equivalent'),reason='explicit relation') for c in payload['existing']])
            builder.call=Mock(side_effect=call)
            groups,audit=builder.deduplicate(candidates,'test')
            self.assertEqual(groups,[['a','b'],['c']])
            self.assertEqual(len(audit['comparisons']),3)

    def test_numeric_crop_requires_the_actual_readable_digit(self):
        region=dict(object='number 2 in center cell')
        grid=dict(visible_text='',visible_objects='crossing grid lines',clipped_or_ambiguous=False)
        self.assertFalse(inspected_region_supported(region,grid,dict(supported=True)))
        digit=dict(visible_text='2',visible_objects='digit',clipped_or_ambiguous=False)
        self.assertTrue(inspected_region_supported(region,digit,dict(supported=True)))
        digit['clipped_or_ambiguous']=True
        self.assertFalse(inspected_region_supported(region,digit,dict(supported=True)))

    def test_padded_steps_do_not_count_as_valid_common_operations(self):
        records={s:dict(operations=[{}, {}, {}]) for s in ['a','b']}
        w=dict(source_ids=['a','b'],steps=['            ','Set matching tick positions','Plot the resulting objects'],support=[])
        self.assertIn('padding',support_error(w,records))


if __name__=='__main__':unittest.main()
