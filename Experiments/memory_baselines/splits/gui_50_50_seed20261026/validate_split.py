"""Independent integrity checks for the generated GUI 50/50 manifests."""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
read=lambda name:json.loads((HERE/name).read_text())
audit=read('audit.json');aw=read('androidworld.json');os=read('osworld.json')
for name,manifest in [('androidworld',aw),('osworld',os)]:
    if name=='androidworld':
        left=set(manifest['memory_construction']);right=set(manifest['test'])
    else:
        left={x for v in manifest['memory_construction'].values() for x in v}
        right={x for v in manifest['test'].values() for x in v}
    assert len(left)==len(right)==50 and not left&right,(name,len(left),len(right))
    assert not (left|right)&set(manifest['pilot_excluded'])
    if name=='androidworld':
        assert not (left|right)&set(manifest['additional_omitted'])
        items={x['task']:x for x in audit[name]['selected_items']}
        assert set(items)==left|right
        assert all((x['role']=='memory_construction')==(x['task'] in left) for x in items.values())
        for side in (left,right):
            assert Counter(items[t]['difficulty_tier'] for t in side)=={'easy':36,'medium':9,'hard':5}
        assert audit[name]['max_app_gap']<=1
        families={}
        for t,x in items.items():
            families.setdefault(x['family'],set()).add('memory' if t in left else 'test')
        assert all(len(v)==1 for v in families.values())
    else:
        assert not (left|right)&set(manifest['near_duplicate_excluded'])
        assert read('osworld_memory.json')==manifest['memory_construction']
        assert read('osworld_test.json')==manifest['test']
        records=audit[name]['selected_items'];assert {x['id'] for x in records}==left|right
        assert all((x['role']=='memory_construction')==(x['id'] in left) for x in records)
        assert audit[name]['domain_counts']['memory_construction']==audit[name]['domain_counts']['test']
        assert audit[name]['joint_counts']['memory_construction']==audit[name]['joint_counts']['test']
        assert audit[name]['download_setup_counts']['memory_construction']==audit[name]['download_setup_counts']['test']==43
        def assets(role):
            return {v for x in records if x['role']==role for v in x['download_assets']
                    if not v.split('|')[-1].endswith('thunderbird-profile.tar.gz')}
        assert not assets('memory_construction') & assets('test')
        available=json.loads((ROOT/'Benchmark/OSWorld/evaluation_examples/test_all.json').read_text())
        assert all(t in available[d] for d,ids in manifest['memory_construction'].items() for t in ids)
        assert all(t in available[d] for d,ids in manifest['test'].items() for t in ids)
    print(name,'50 construction + 50 test; disjoint; pilot excluded; strata checked')
