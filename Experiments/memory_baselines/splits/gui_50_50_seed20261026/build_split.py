"""Deterministic 50/50 GUI task split; reads benchmark metadata, never task outcomes."""
from __future__ import annotations
import json,math,random,re
from collections import Counter,defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SEED=20261026
AW_PILOT={'SystemBrightnessMax','SystemBluetoothTurnOn','ClockTimerEntry','MarkorCreateNote','ContactsAddContact',
          'SystemBrightnessMin','SystemBluetoothTurnOff','ClockStopWatchRunning','MarkorCreateFolder','ContactsNewContactDraft'}
OS_PILOT={'28cc3b7e-b194-4bc9-8353-d04c0f4d56d2','3ce045a0-877b-42aa-8d2c-b4a863336ab8',
          'b6781586-6346-41cd-935a-a6b1487918fc','e0df059f-28a6-4169-924f-b9623e7184cc',
          'f9be0997-4b7c-45c5-b05c-4612b44a6118','bedcedc4-4d72-425e-ad62-21960b11fe0d',
          'a4d98375-215b-4a4d-aee9-3d4370fccc41','ec4e3f68-9ea4-4c18-a5c9-69f89d1178b3',
          '13584542-872b-42d8-b299-866967b5c3ef','5ced85fc-fa1a-4217-95fd-0fb530545ce2'}

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def aw_level(value):
    return 'easy' if value<=2 else 'medium' if value<=4 else 'hard'

def aw_family(task):
    base=re.sub(r'Verify$','',task)
    base=re.sub(r'\d+$','',base)
    for prefix in ('SimpleCalendarAddOneEvent','SimpleSmsReply','RecipeDeleteDuplicateRecipes',
                   'RecipeDeleteMultipleRecipes','ExpenseDeleteDuplicates','ExpenseDeleteMultiple'):
        if base.startswith(prefix):return prefix
    return base

def android():
    path=ROOT/'Docs/实验方案/assets/20261026-gui-protocol/androidworld_registry_stats.json'
    data=json.loads(path.read_text());assert data['total']==116
    items=data['items'];known={x['task'] for x in items}
    excluded=AW_PILOT|{t+'Verify' for t in AW_PILOT if t+'Verify' in known}
    eligible=[x for x in items if x['task'] not in excluded]
    assert len(eligible)==102
    # Prespecified duplicate-numbered variants omitted to make three tier totals even.
    omitted={'ExpenseDeleteMultiple2','RecipeAddMultipleRecipesFromMarkor2'}
    chosen=[x for x in eligible if x['task'] not in omitted]
    assert len(chosen)==100 and Counter(aw_level(x['complexity']) for x in chosen)=={'easy':72,'medium':18,'hard':10}
    groups=defaultdict(list)
    for row in chosen:groups[aw_family(row['task'])].append(row)
    names=sorted(groups);n=len(names)
    apps=sorted({a for x in chosen for a in (x['apps'] or ['information_query'])})
    # z_g=1 construction, 0 test. d_app measures absolute app-count gap.
    size=n+len(apps);objective=np.zeros(size);rng=random.Random(SEED)
    objective[:n]=[rng.random()*1e-5 for _ in names]
    objective[n:]=1.0
    lower=np.zeros(size);upper=np.array([1]*n+[100]*len(apps),dtype=float)
    rows=[];lb=[];ub=[]
    def constraint(coeff,left,right):rows.append(coeff);lb.append(left);ub.append(right)
    for level,target in [('easy',36),('medium',9),('hard',5)]:
        coeff=np.zeros(size)
        for i,name in enumerate(names):coeff[i]=sum(aw_level(x['complexity'])==level for x in groups[name])
        constraint(coeff,target,target)
    coeff=np.zeros(size)
    for i,name in enumerate(names):coeff[i]=len(groups[name])
    constraint(coeff,50,50)
    for j,app in enumerate(apps):
        coeff=np.zeros(size)
        for i,name in enumerate(names):coeff[i]=sum(app in (x['apps'] or ['information_query']) for x in groups[name])*2
        total=sum(app in (x['apps'] or ['information_query']) for x in chosen)
        plus=coeff.copy();plus[n+j]=-1;constraint(plus,-np.inf,total)
        minus=-coeff;minus[n+j]=-1;constraint(minus,-np.inf,-total)
    # A previously seen stopwatch variant is kept in construction if selected.
    if 'ClockStopWatchPaused' in names:
        coeff=np.zeros(size);coeff[names.index('ClockStopWatchPaused')]=1;constraint(coeff,1,1)
    matrix=np.vstack(rows)
    result=milp(c=objective,integrality=np.array([1]*n+[0]*len(apps)),bounds=Bounds(lower,upper),
                constraints=LinearConstraint(matrix,lb,ub),options={'time_limit':60,'mip_rel_gap':0.001})
    if result.x is None or result.status not in (0,1):raise RuntimeError('Android stratification infeasible: '+str(result.message))
    construction=sorted(x['task'] for i,name in enumerate(names) if result.x[i]>.5 for x in groups[name])
    test=sorted(x['task'] for i,name in enumerate(names) if result.x[i]<=.5 for x in groups[name])
    assert len(construction)==len(test)==50 and not set(construction)&set(test)
    byname={x['task']:x for x in chosen}
    for level,target in [('easy',36),('medium',9),('hard',5)]:
        assert sum(aw_level(byname[t]['complexity'])==level for t in construction)==target
        assert sum(aw_level(byname[t]['complexity'])==level for t in test)==target
    manifest={'benchmark':'androidworld','protocol':'task_type_disjoint_frozen_test','split_seed':SEED,
              'suite_seed':SEED,'task_index':0,'memory_construction':construction,'test':test,
              'difficulty':'native task complexity; project tiers easy<=2, medium<=4, hard>4',
              'pilot_excluded':sorted(excluded),'additional_omitted':sorted(omitted)}
    write(OUT/'androidworld.json',manifest)
    summary={'eligible_before_two_omissions':len(eligible),'selected':100,'groups':len(groups),
             'tier_counts':{role:dict(Counter(aw_level(byname[t]['complexity']) for t in manifest[role])) for role in ('memory_construction','test')},
             'app_counts':{role:dict(sorted(Counter(a for t in manifest[role] for a in (byname[t]['apps'] or ['information_query'])).items())) for role in ('memory_construction','test')},
             'max_app_gap':max(abs(sum(app in (byname[t]['apps'] or ['information_query']) for t in construction)-
                                sum(app in (byname[t]['apps'] or ['information_query']) for t in test)) for app in apps),
             'selected_items':[{**x,'difficulty_tier':aw_level(x['complexity']),'family':aw_family(x['task']),
                                'role':'memory_construction' if x['task'] in construction else 'test'} for x in chosen]}
    return summary

def proxy(row):
    words=len(re.findall(r"[A-Za-z0-9_']+",row['instruction']))
    return words+8*max(0,len(row['related_apps'])-1)+2*min(len(row.get('config') or []),6)

def osworld():
    root=ROOT/'Benchmark/OSWorld/evaluation_examples'
    all_ids=json.loads((root/'test_all.json').read_text())
    infeasible={task for ids in json.loads((root/'test_infeasible.json').read_text()).values() for task in ids}
    records=[]
    for domain,ids in all_ids.items():
        for task in ids:
            path=root/'examples'/domain/(task+'.json');row=json.loads(path.read_text())
            if row['id']!=task:raise ValueError('OSWorld ID mismatch: '+task)
            if task in infeasible or task in OS_PILOT or row.get('proxy'):continue
            records.append({'id':task,'domain':domain,'instruction':row['instruction'],
                            'related_apps':row['related_apps'],'config_count':len(row.get('config') or []),
                            'config_types':[c.get('type') for c in (row.get('config') or []) if isinstance(c,dict)],
                            'download_assets':[str(f.get('url'))+'|'+str(f.get('path'))
                                for c in (row.get('config') or []) if isinstance(c,dict) and c.get('type')=='download'
                                for f in c.get('parameters',{}).get('files',[])],
                            'proxy_score':proxy(row)})
    assert len(records)==277
    # Exclude both sides of very close same-domain instruction pairs before sampling.
    near_duplicate=set()
    for i,left in enumerate(records):
        for right in records[i+1:]:
            if left['domain']!=right['domain']:continue
            if SequenceMatcher(None,left['instruction'].lower(),right['instruction'].lower()).ratio()>.90:
                near_duplicate.update((left['id'],right['id']))
    records=[r for r in records if r['id'] not in near_duplicate]
    assert len(records)==275
    bydomain=defaultdict(list)
    for row in records:bydomain[row['domain']].append(row)
    cells=defaultdict(list)
    for domain,values in bydomain.items():
        ordered=sorted(values,key=lambda x:(x['proxy_score'],x['id']))
        for rank,row in enumerate(ordered):
            tier=('easy','medium','hard')[min(2,rank*3//len(ordered))]
            row['difficulty_tier']=tier;cells[(domain,tier)].append(row)
    # First reserve proportional domain quotas, then allocate each domain across
    # its own proxy tertiles. Each cell contributes equally to the two arms.
    slots={k:0 for k in cells};capacity={k:len(v)//2 for k,v in cells.items()}
    domain_cap={d:sum(capacity[k] for k in cells if k[0]==d) for d in bydomain}
    domain_target={d:50*len(v)/len(records) for d,v in bydomain.items()}
    quotas={d:min(domain_cap[d],max(1,math.floor(domain_target[d]))) for d in bydomain}
    while sum(quotas.values())<50:
        available=[d for d in bydomain if quotas[d]<domain_cap[d]]
        if not available:raise RuntimeError('OSWorld domain capacity below 50 pairs')
        d=max(available,key=lambda d:(domain_target[d]-quotas[d],len(bydomain[d]),d))
        quotas[d]+=1
    while sum(quotas.values())>50:
        available=[d for d in bydomain if quotas[d]>1]
        d=min(available,key=lambda d:(domain_target[d]-quotas[d],len(bydomain[d]),d))
        quotas[d]-=1
    for domain in sorted(bydomain):
        domain_cells=[k for k in cells if k[0]==domain]
        targets={k:quotas[domain]*len(cells[k])/len(bydomain[domain]) for k in domain_cells}
        for k in domain_cells:slots[k]=min(capacity[k],math.floor(targets[k]))
        while sum(slots[k] for k in domain_cells)<quotas[domain]:
            available=[k for k in domain_cells if slots[k]<capacity[k]]
            if not available:raise RuntimeError('OSWorld tier capacity below domain quota')
            k=max(available,key=lambda k:(targets[k]-slots[k],len(cells[k]),k))
            slots[k]+=1
    assert sum(slots.values())==50 and all(quotas[d]>=1 for d in quotas)
    memory=[];test=[]
    for key in sorted(cells):
        values=sorted(cells[key],key=lambda x:(x['proxy_score'],x['id']))
        rng=random.Random(f'{SEED}:{key[0]}:{key[1]}')
        rng.shuffle(values);take=values[:2*slots[key]]
        # Neighboring proxy scores go to opposite arms; orientation uses fixed RNG.
        take.sort(key=lambda x:(x['proxy_score'],x['id']))
        for i in range(0,len(take),2):
            pair=take[i:i+2]
            if rng.getrandbits(1):pair.reverse()
            memory.append(pair[0]);test.append(pair[1])
    # Pair orientation may be flipped without changing domain/tier quotas. Balance
    # setup-download dependence to avoid comparing different environment burdens.
    has_download=lambda row:'download' in row['config_types']
    gap=sum(has_download(x) for x in memory)-sum(has_download(x) for x in test)
    for i in range(len(memory)):
        if abs(gap)<=1:break
        new_gap=gap-2*(int(has_download(memory[i]))-int(has_download(test[i])))
        if abs(new_gap)<abs(gap):
            memory[i],test[i]=test[i],memory[i];gap=new_gap
    assert abs(gap)<=1
    # Prevent the same task-specific downloaded document/code file occurring
    # across construction/test. Thunderbird's common profile is base environment.
    def task_assets(row):
        return {a for a in row['download_assets'] if not a.split('|')[-1].endswith('thunderbird-profile.tar.gz')}
    for _ in range(50):
        source_assets=set().union(*(task_assets(x) for x in memory))
        conflicts=[i for i,row in enumerate(test) if task_assets(row)&source_assets]
        if not conflicts:break
        i=conflicts[0];old=test[i];used={x['id'] for x in memory+test}
        candidates=[x for x in cells[(old['domain'],old['difficulty_tier'])]
                    if x['id'] not in used and ('download' in x['config_types'])==('download' in old['config_types'])
                    and not task_assets(x)&source_assets]
        if not candidates:raise RuntimeError('Cannot isolate OSWorld task-specific asset for '+old['id'])
        test[i]=min(candidates,key=lambda x:(abs(x['proxy_score']-old['proxy_score']),x['id']))
    else:raise RuntimeError('OSWorld asset repair did not converge')
    assert abs(gap)<=1
    assert len(memory)==len(test)==50 and not {x['id'] for x in memory}&{x['id'] for x in test}
    assert not set().union(*(task_assets(x) for x in memory)) & set().union(*(task_assets(x) for x in test))
    def native(items):
        result=defaultdict(list)
        for row in items:result[row['domain']].append(row['id'])
        return {k:sorted(v) for k,v in sorted(result.items())}
    construction=native(memory);heldout=native(test)
    assert {k:len(v) for k,v in construction.items()}=={k:len(v) for k,v in heldout.items()}
    for key in cells:
        assert sum((r['domain'],r['difficulty_tier'])==key for r in memory)==slots[key]
        assert sum((r['domain'],r['difficulty_tier'])==key for r in test)==slots[key]
    manifest={'benchmark':'osworld','protocol':'task_id_disjoint_frozen_test','split_seed':SEED,
              'eligible_rule':'non-infeasible, proxy=false, not pilot, no near-duplicate pair',
              'difficulty':'project structural-complexity proxy; within-domain tertiles; NOT official difficulty',
              'proxy_formula':'instruction token count + 8*(related_apps-1) + 2*min(config_count,6)',
              'memory_construction':construction,'test':heldout,
              'pilot_excluded':sorted(OS_PILOT),'near_duplicate_excluded':sorted(near_duplicate),
              'shared_base_asset_exception':['thunderbird-profile.tar.gz']}
    write(OUT/'osworld.json',manifest)
    write(OUT/'osworld_memory.json',construction);write(OUT/'osworld_test.json',heldout)
    summary={'offline_noninfeasible_after_pilot':277,'eligible_after_near_duplicate':275,
             'selected':100,'domain_counts':{role:{k:len(v) for k,v in manifest[role].items()} for role in ('memory_construction','test')},
             'tier_counts':{role:dict(Counter(x['difficulty_tier'] for x in items)) for role,items in [('memory_construction',memory),('test',test)]},
             'joint_counts':{role:{f'{d}/{tier}':sum(x['domain']==d and x['difficulty_tier']==tier for x in items) for d,tier in sorted(cells)} for role,items in [('memory_construction',memory),('test',test)]},
             'selected_items':[{**x,'role':'memory_construction'} for x in memory]+[{**x,'role':'test'} for x in test],
             'download_setup_counts':{role:sum('download' in x['config_types'] for x in items) for role,items in [('memory_construction',memory),('test',test)]}}
    return summary

if __name__=='__main__':
    audit={'split_seed':SEED,'androidworld':android(),'osworld':osworld()}
    write(OUT/'audit.json',audit)
    print('Android',audit['androidworld']['tier_counts'],'max_app_gap',audit['androidworld']['max_app_gap'])
    print('OSWorld',audit['osworld']['tier_counts'],'domains',audit['osworld']['domain_counts'])
