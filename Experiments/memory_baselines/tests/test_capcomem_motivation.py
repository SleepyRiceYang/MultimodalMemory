"""Protocol regressions: sampling, paired evidence, and outcome isolation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vlm_memory.pipeline import capcomem_motivation as m


class MotivationProtocolTests(unittest.TestCase):
    def test_shortages_keep_actual_categories_and_unique_ids(self):
        rows=[dict(id=str(i),category=('improved' if i<8 else 'harmed' if i<14 else 'unchanged_correct' if i<70 else 'unchanged_wrong')) for i in range(100)]
        selected=m.sample_rows(rows,60)
        self.assertEqual(selected,m.sample_rows(rows,60))
        self.assertEqual(len({r['id'] for r in selected}),60)
        self.assertEqual(sum(r['category']=='improved' for r in selected),8)
        self.assertEqual(sum(r['category']=='harmed' for r in selected),6)
        self.assertTrue(any(r['supplement'] for r in selected))
        originals={r['id']:r['category'] for r in rows}
        self.assertTrue(all(originals[r['id']]==r['category'] for r in selected))

    def test_frozen_spec_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'spec.json';m.freeze(p,dict(a=1))
            with self.assertRaises(ValueError):m.freeze(p,dict(a=2))
            self.assertEqual(m.read(p),dict(a=1))

    def test_self_source_is_allowed_and_forms_share_core(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);node=dict(id='n',core='same core',source_ids=['self'])
            m.write_json(root/'sources.json',dict(nodes=[node]))
            m.write_json(root/'retrieval/androidworld/self.json',dict(selected=['n'],self_hit=True))
            m.write_json(root/'assets'/m.digest('n')[:16]/'asset.json',dict(core_sha256=m.digest(node['core']),frame='before',roi='roi',after='after',
                captions=dict(full_caption='full',roi_caption='crop',pair_caption='transition')))
            ctx=[m.context(root,'androidworld','self',f) for f in ('M1','M2','M3','M4')]
            self.assertEqual(len({c['core_sha256'] for c in ctx}),1)
            self.assertTrue(all(c['self_hit'] for c in ctx))
            self.assertEqual([len(c['images']) for c in ctx],[0,1,1,2])
            self.assertEqual(ctx[1]['images'][0][1],ctx[3]['images'][0][1])
            self.assertEqual(m.context(root,'androidworld','self','M0')['selected'],[])

    def test_selector_cannot_be_created_after_execution(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m.write_json(root/'runs/mpdocvqa/test/x/M0/result.json',dict(score=1))
            with self.assertRaises(ValueError):m.choose(root,'mpdocvqa','test',dict(id='x',prompt='question'),'unused')

    def test_action_schema_rejects_extra_fields_and_nested_actions(self):
        import jsonschema
        schema=m.android_action_schema()
        jsonschema.validate(dict(reason='Open app',action_type='open_app',app_name='Markor'),schema)
        for action in (dict(reason='x',action_type='navigate_home',status='infeasible'),
                       dict(action=dict(action_type='click',x=10,y=20)),
                       dict(reason='x',action_type='click',x=1001,y=20)):
            with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(action,schema)

    def test_fixed_choice_uses_construction_scores_and_cost_tie_break(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for f,score,cost in [('M0',0,0),('M1',1,10),('M2',1,30),('M3',0,20)]:
                m.write_json(root/'runs/mpdocvqa/construction/c'/f/'result.json',dict(status='completed',score=score,cost=dict(usage=dict(total_tokens=cost))))
                m.write_json(root/'runs/mpdocvqa/test/t'/f/'result.json',dict(status='completed',score=1-score))
            with patch.object(m,'tasks',return_value=[('construction',dict(id='c')),('test',dict(id='t'))]):
                spec=m.freeze_fixed(root,'mpdocvqa',1)
                self.assertEqual(spec['form'],'M1')
                self.assertTrue(spec['retrospective_pilot_diagnostic'])
                self.assertEqual(m.freeze_fixed(root,'mpdocvqa',1),spec)

    def test_doc_temporal_form_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);node=dict(id='n',core='core',source_ids=['source'])
            m.write_json(root/'sources.json',dict(nodes=[node]))
            m.write_json(root/'retrieval/mpdocvqa/x.json',dict(selected=['n'],self_hit=False))
            m.write_json(root/'assets'/m.digest('n')[:16]/'asset.json',dict(core_sha256='hash',after=None,captions={}))
            with self.assertRaises(ValueError):m.context(root,'mpdocvqa','x','M4')


if __name__=='__main__':unittest.main()
