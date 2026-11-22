import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from gui_baseline.evidence_eval import EvidenceModel
from gui_baseline.cold_continual import Model

class EvidenceInputTest(unittest.TestCase):
    def test_only_b3_receives_selected_history_and_original_patch_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'stationary_retrieval.json').write_text(json.dumps({'selected':[{'id':4}]}))
            state={'annotation_view_ids':['a'],'available_views':[{'view_id':'a','view_type':'focus_crop','path':'selected.png','source':'task'}]}
            bank={'stationary':[{'id':4,'function':'open menu','evidence_state':state}]}
            calls=[]
            def call(self,folder,system,user,images=(),*args,**kwargs):
                calls.append((Path(folder).name,list(images),user))
                return json.dumps({'view_ids':['stationary_4/a'],'reason':'locate'}) if Path(folder).name.endswith('selection') else '{}'
            for condition,expected in [('B2',[('CURRENT','current.png')]),('B3',[('CURRENT','current.png'),('HISTORICAL focus_crop','selected.png')])]:
                model=object.__new__(EvidenceModel);model.args=SimpleNamespace(condition=condition);model.memory=bank
                with patch.object(Model,'call',call):
                    model.call(root/'actor','system','request',[('CURRENT','current.png'),('OLD DEFAULT','wrong.png')])
                self.assertEqual(calls[-1][1],expected)
                self.assertNotIn('wrong.png',str(calls[-1][1]))
            self.assertEqual(calls[1][2],calls[3][2])
if __name__=='__main__':unittest.main()
