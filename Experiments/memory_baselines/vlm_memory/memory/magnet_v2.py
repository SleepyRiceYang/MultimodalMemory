"""Evidence-bound MAGNET adaptation with disjoint clustering and local deduplication."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np

from ..config import write_json
from ..prompts import magnet_v2 as prompts
from .representations import request, obj, array, STRING, BOX, assets, source_images, crop, entry, eligible
from .retrieval import post

VERSION = 'described_complete_link_v2'


def text_normalized(text):
    return ' '.join(text.split())


def parameterize(text):
    """Keep grid dimensions, mask incidental numeric instances before embedding."""
    dimensions={}
    def dimension(match):
        key=f'GRIDDIMENSION{chr(65+len(dimensions))}'
        dimensions[key]=match.group(0)
        return key
    text=re.sub(r'\b\d+\s*[x×]\s*\d+(?:\s*[x×]\s*\d+)?\b',dimension,text)
    text=re.sub(r'(?<![A-Za-z_])\d+(?:[.,]\d+)*(?:st|nd|rd|th)?', '[VALUE]',text)
    for key,value in dimensions.items():
        text=text.replace(key,value)
    return text


def evidence_anchors(steps):
    anchors=[]
    for step_index,step in enumerate(steps):
        # Store exact source substrings; the model selects IDs instead of retyping evidence.
        for match in re.finditer(r'[^\n]+(?:\n(?!\n)[^\n]+)*',step.get('text','')):
            text=match.group(0)
            for start in range(0,len(text),600):
                quote=text[start:start+600]
                if len(quote.strip())>=8:
                    anchors.append(dict(evidence_id=f'e{len(anchors)}',step_index=step_index,quote=quote))
    return anchors


def inspected_region_supported(region,inspection,assessment):
    if inspection is None or inspection['clipped_or_ambiguous'] or assessment is None or not assessment['supported']:
        return False
    text=inspection['visible_text'].strip()
    if not text and not inspection['visible_objects'].strip():
        return False
    digit=re.search(r'\b(?:number|digit)\s+(\d+)\b',region['object'],re.I)
    if digit and not re.search(r'(?<!\d)'+re.escape(digit.group(1))+r'(?!\d)',text):
        return False
    if re.match(r'^(?:date|text|label)\b',region['object'],re.I) and not text:
        return False
    return True


def complete_link(similarities, ids, threshold, max_size=8):
    """Disjoint, deterministic complete-link merges; never a single-link chain."""
    groups = [[i] for i in sorted(range(len(ids)), key=lambda i: ids[i])]
    while True:
        candidates = []
        for a, left in enumerate(groups):
            for b in range(a+1, len(groups)):
                right = groups[b]
                if len(left)+len(right) > max_size:
                    continue
                score = min(float(similarities[i,j]) for i in left for j in right)
                if score >= threshold:
                    candidates.append((-score, tuple(sorted(ids[i] for i in left+right)), a, b))
        if not candidates:
            return sorted(groups, key=lambda g: tuple(sorted(ids[i] for i in g)))
        _, _, a, b = min(candidates)
        groups[a] = sorted(groups[a]+groups[b], key=lambda i: ids[i])
        groups.pop(b)


def support_error(workflow, records):
    """Return an admission error; IDs alone never count as step evidence."""
    ids = workflow['source_ids']
    if len(set(ids)) < 2 or len(set(ids)) != len(ids) or not set(ids) <= set(records):
        return 'requires at least two distinct legal source IDs'
    if not 3 <= len(workflow['steps']) <= 5:
        return 'requires 3..5 meaningful workflow steps'
    if any(len(step.strip())<12 or step.strip().lower() in ('observe the problem','analyze the problem','return the answer') for step in workflow['steps']):
        return 'empty or generic padding is not a workflow operation'
    if any(not re.search(r'[A-Za-z\u4e00-\u9fff]',step) for step in workflow['steps']):
        return 'punctuation or numeric padding is not a workflow operation'
    rows = workflow['support']
    keys = [(r['step_index'], r['source_id']) for r in rows]
    if len(keys) != len(set(keys)):
        return 'repeated step/source evidence'
    for r in rows:
        if r['source_id'] not in ids or not 0 <= r['step_index'] < len(workflow['steps']):
            return 'unknown source or workflow step'
        if not 0 <= r['operation_index'] < len(records[r['source_id']]['operations']):
            return 'unknown source operation'
    for step_index in range(len(workflow['steps'])):
        if {r['source_id'] for r in rows if r['step_index'] == step_index} != set(ids):
            return 'every claimed source must support every common step'
    # The matched operations must form an actual ordered subsequence in each source.
    for source_id in ids:
        indices = [next(r['operation_index'] for r in rows if r['source_id'] == source_id and r['step_index'] == i)
                   for i in range(len(workflow['steps']))]
        if indices != sorted(set(indices)):
            return 'common operations are repeated or out of execution order'
    return None


class Builder:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self.options = config.get('memory_construction', {})
        self.url = config['retrieval']['embedding_url'].rstrip('/')

    def call(self, name, system, payload, images, schema, cap=1536):
        model = {**self.config['model'], 'construction_max_tokens': cap,
                 'check_context_budget': True, 'max_input_tokens': 16384,
                 'context_reserve_tokens': 256, 'min_output_tokens': 128}
        result = request(self.root, {**self.config, 'model':model}, 'v2_'+name, system, payload, images, schema)
        return result

    def count(self, texts):
        return post(self.url+'/tokens', {'texts':texts})

    def encode(self, texts):
        if not texts:
            return np.empty((0,1024),dtype=np.float32)
        return np.asarray(post(self.url+'/encode', {'texts':texts})['embeddings'],dtype=np.float32)

    def describe(self, episode):
        benchmark, source_id = episode['benchmark'], episode['id']
        schema = obj(dict(family=dict(enum=prompts.FAMILIES[benchmark]), problem_structure=STRING,
            target=STRING, critical_constraints=array(STRING,5), visual_requirements=array(STRING,5), normalized_description=STRING))
        public = dict(benchmark=benchmark, question=episode['task'], family_vocabulary=prompts.FAMILIES[benchmark])
        originals=assets(self.root,[episode],True)
        cached=self.root/'construction_calls'/('v2_problem_'+source_id+'.json')
        if len(originals)<=2 or cached.exists():
            problem=self.call('problem_'+source_id,prompts.PROBLEM,public,originals,schema)
        else:
            batches=[]
            for start in range(0,len(originals),2):
                batch=self.call(f'problem_{source_id}_pages_{start}',prompts.PROBLEM,
                    dict(**public,visible_page_indices=list(range(start,min(start+2,len(originals)))),total_pages=len(originals)),originals[start:start+2],schema)
                if batch is not None:
                    batches.append(batch)
            problem=self.call('problem_pages_summary_'+source_id,prompts.PROBLEM+' Combine the supplied fixed page-batch descriptions; do not invent missing evidence.',
                dict(**public,page_batches=batches),[],schema)
        if problem is None:
            return None
        raw_problem=copy.deepcopy(problem)
        if benchmark=='mpdocvqa' and re.search(r'\b(?:what(?:\s+is|\s+was)?\s+(?:the\s+)?year|which\s+year)\b',episode['task'],re.I):
            # An explicit public target cannot be replaced by the model's broader date class.
            problem['family']='year'
        if benchmark=='mpdocvqa':
            # Remove observed source answers from the grouping key, never read evaluation gold.
            for step in episode['steps']:
                try:
                    answer=json.loads(step.get('text','')).get('answer','')
                except (ValueError,AttributeError):
                    continue
                if isinstance(answer,str) and len(answer.strip())>=2:
                    problem['normalized_description']=re.sub(re.escape(answer.strip()),'[FIELD_VALUE]',problem['normalized_description'],flags=re.I)
        problem['normalized_description']=parameterize(problem['normalized_description'])
        if self.count([problem['normalized_description']])[0]['tokens'] > 160:
            compressed = self.call('problem_'+source_id+'_compressed',prompts.COMPRESS,problem,[],obj(dict(normalized_description=STRING)),512)
            if compressed is None:
                return None
            problem['normalized_description']=parameterize(compressed['normalized_description'])
            raw_problem['normalized_description']=compressed['normalized_description']
        if self.count([problem['normalized_description']])[0]['tokens'] > 160:
            write_json(self.root/'magnet_v2'/'rejected_descriptions'/(source_id+'.json'),dict(reason='description exceeds 160 encoder tokens',problem=problem))
            return None
        operations = obj(dict(operation=STRING,step_index=dict(enum=list(range(len(episode['steps'])))),quote=dict(type='string',maxLength=500)))
        if benchmark=='mpdocvqa':
            # These traces record reader outputs, not the unobserved search/OCR process.
            actual=[]
            for i,step in enumerate(episode['steps']):
                if step.get('kind') in ('page_reading','final_answer'):
                    actual.append(dict(operation=('Read supplied pages and record field evidence' if step['kind']=='page_reading' else 'Return the field from the recorded evidence'),
                        step_index=i,quote=step.get('text','')[:500]))
            procedure=dict(procedure_description='Read supplied document pages, record the requested field and page evidence, then return the supported field. No search, OCR or regex process is recorded.',operations=actual[:5])
        else:
            anchors=evidence_anchors(episode['steps'])
            if not anchors:
                procedure=dict(procedure_description='No reusable recorded operation.',operations=[])
            else:
                selected=self.call('anchored_procedure_'+source_id,prompts.ANCHORED_PROCEDURE,
                    dict(fragments=anchors,problem=raw_problem),[],obj(dict(procedure_description=STRING,
                    operations=array(obj(dict(operation=STRING,evidence_id=dict(enum=[a['evidence_id'] for a in anchors]))),5))),1536)
                if selected is None:
                    return None
                by_id={a['evidence_id']:a for a in anchors}
                seen=set()
                chosen=[]
                for row in selected['operations']:
                    if row['evidence_id'] in by_id and row['evidence_id'] not in seen:
                        seen.add(row['evidence_id'])
                        chosen.append(dict(**by_id[row['evidence_id']],operation=parameterize(row['operation'])))
                chosen.sort(key=lambda row:int(row['evidence_id'][1:]))
                procedure=dict(procedure_description=selected['procedure_description'],operations=chosen)
        if procedure is None:
            return None
        valid, rejected = [], []
        for operation in procedure['operations']:
            i=operation['step_index']
            quote=text_normalized(operation['quote'])
            if i not in range(len(episode['steps'])) or len(quote)<8 or quote not in text_normalized(episode['steps'][i].get('text','')):
                rejected.append(dict(operation=operation,reason='quote not a real nontrivial source substring'))
            else:
                valid.append(operation)
        record=dict(id=source_id,benchmark=benchmark,problem=problem,procedure_description=parameterize(procedure['procedure_description']),
                    operations=valid,rejected_operations=rejected,source_success=episode['success'])
        write_json(self.root/'magnet_v2'/'descriptions'/(source_id+'.json'),record)
        return record

    def describe_sources(self, episodes):
        records=[]
        for episode in episodes:
            if eligible(episode):
                record=self.describe(episode)
                if record is not None:
                    records.append(record)
        problems=[r['problem']['normalized_description'] for r in records]
        procedures=[r['procedure_description'] for r in records]
        if any(r['tokens']>512 for r in self.count(procedures)):
            raise ValueError('Procedure summary exceeds encoder window; explicit compression required')
        problem_vectors,procedure_vectors=self.encode(problems),self.encode(procedures)
        similarities=np.minimum(problem_vectors@problem_vectors.T,procedure_vectors@procedure_vectors.T)
        for i,r in enumerate(records):
            for j,s in enumerate(records):
                if not r['operations'] or not s['operations'] or (r['problem']['family']!=s['problem']['family']):
                    similarities[i,j]=-1
        self.root.joinpath('magnet_v2').mkdir(parents=True,exist_ok=True)
        np.save(self.root/'magnet_v2'/'similarities.npy',similarities)
        write_json(self.root/'magnet_v2'/'source_descriptions.json',records)
        return records,similarities

    def workflows(self, records, similarities, threshold):
        ids=[r['id'] for r in records]
        clusters=complete_link(similarities,ids,threshold)
        admitted, rejected=[],[]
        for index,group in enumerate(clusters):
            if len(group)<2:
                continue
            sources={records[i]['id']:records[i] for i in group}
            if any(len(r['operations'])<3 for r in sources.values()):
                rejected.append(dict(cluster=list(sources),reason='insufficient recorded operations for a three-step common workflow'))
                continue
            fields=dict(title=STRING,problem_description=STRING,procedure_description=STRING,
                applicability=array(STRING,4),exclusions=array(STRING,4),parameters=array(STRING,5),steps=array({**STRING,'minLength':12},5,3),
                support=array(obj(dict(step_index=dict(type='integer',minimum=0,maximum=4),
                    operations_by_source=obj({key:dict(enum=list(range(len(record['operations'])))) for key,record in sources.items()}))),5,3))
            data=self.call('evidence_workflows_'+str(index),prompts.ALIGNED_WORKFLOWS,dict(sources=list(sources.values())),[],obj(dict(workflows=array(obj(fields),3))),3072)
            if data is None:
                continue
            for candidate_index,w in enumerate(data['workflows']):
                w['source_ids']=list(sources)
                w['support']=[dict(step_index=row['step_index'],source_id=key,operation_index=operation_index)
                    for row in w['support'] for key,operation_index in row['operations_by_source'].items()]
                error=support_error(w,sources)
                verification=[]
                if error is None:
                    items=[]
                    for row_index,row in enumerate(w['support']):
                        source=sources[row['source_id']]
                        operation=source['operations'][row['operation_index']]
                        items.append(dict(id=row_index,step=w['steps'][row['step_index']],problem=source['problem']['normalized_description'],operation=operation))
                    checks=[]
                    for start in range(0,len(items),12):
                        batch=items[start:start+12]
                        name=f'verify_evidence_{index}_{candidate_index}'+(f'_batch_{start}' if len(items)>12 else '')
                        response=self.call(name,prompts.VERIFY_BATCH,dict(items=batch),[],
                            obj(dict(checks=array(obj(dict(id=dict(enum=[r['id'] for r in batch]),supported=dict(type='boolean'),
                                reason=dict(type='string',maxLength=160))),len(batch),len(batch)))),3072)
                        if response is not None:
                            checks.extend(response['checks'])
                    checked=dict(checks=checks)
                    if checked is None or sorted(r['id'] for r in checked['checks'])!=list(range(len(items))):
                        error='missing or repeated source/step verification'
                    else:
                        assessments={r['id']:r for r in checked['checks']}
                        verification=[dict(**row,assessment=assessments[i]) for i,row in enumerate(w['support'])]
                        if not all(r['supported'] for r in checked['checks']):
                            error='common step lacks semantic support'
                candidate=dict(**w,candidate_id=f'workflow_{index}_{candidate_index}')
                if error is not None:
                    rejected.append(dict(candidate=candidate,reason=error,verification=verification))
                else:
                    candidate['step_verification']=verification
                    admitted.append(candidate)
        write_json(self.root/'magnet_v2'/'workflow_admission.json',dict(clusters=[[ids[i] for i in group] for group in clusters],
            memberships=sum(map(len,clusters)),unique_sources=len(ids),candidates=admitted,rejected=rejected,threshold=threshold))
        return admitted

    def candidate_descriptions(self, candidates, tag):
        result={}
        row=obj(dict(id=dict(type='string'),problem_description=STRING,procedure_description=STRING))
        # Descriptions are bounded before any global embedding or local comparison.
        for start in range(0,len(candidates),4):
            batch=candidates[start:start+4]
            data=self.call(f'describe_{tag}_{start}',prompts.CANDIDATE_DESCRIPTIONS,
                dict(candidates=[{k:v for k,v in c.items() if k not in ('step_verification','support','image','crop_inspection')} for c in batch]),[],
                obj(dict(descriptions=array(row,len(batch),len(batch)))),2048)
            if data is None:
                raise ValueError('Cannot establish bounded candidate descriptions')
            keys=[r['id'] for r in data['descriptions']]
            if len(keys)!=len(set(keys)) or set(keys)!={c['candidate_id'] for c in batch}:
                raise ValueError('Candidate description identities differ')
            for r in data['descriptions']:
                text=r['problem_description']+'\n'+r['procedure_description']
                if self.count([text])[0]['tokens']>384:
                    raise ValueError('Candidate description exceeds 384 tokens; do not silently truncate')
                result[r['id']]=r
        return result

    def deduplicate(self,candidates,tag):
        if not candidates:
            return [],dict(groups=[],comparisons=[])
        ids=[c['candidate_id'] for c in candidates]
        if len(set(ids))!=len(ids):
            raise ValueError('Duplicate candidate identity')
        descriptions=self.candidate_descriptions(candidates,tag)
        vectors=self.encode([descriptions[i]['problem_description']+'\n'+descriptions[i]['procedure_description'] for i in ids])
        mapped={c['candidate_id']:c for c in candidates}
        groups,comparisons,pair_relations=[],[],{}
        for i,candidate in enumerate(candidates):
            # Fixed source order; recall only existing representatives, never the whole bank.
            nearest=sorted(range(len(groups)),key=lambda g:(-float(vectors[i]@vectors[ids.index(groups[g][0])]),groups[g][0]))[:5]
            checks=[member for g in nearest for member in groups[g]]
            # Validate all members before merging: representative matches do not imply transitivity.
            for start in range(0,len(checks),5):
                batch=checks[start:start+5]
                assessment=obj(dict(id=dict(enum=batch),relation=dict(enum=['equivalent','alternative_strategy','shared_subroutine','conflict','unrelated','uncertain']),reason=dict(type='string',maxLength=240)))
                def view(value):
                    return {k:v for k,v in value.items() if k not in ('step_verification','support','image')}
                data=self.call(f'dedup_{tag}_{i}_{start}',prompts.RELATIONS,
                    dict(new_candidate=view(candidate),existing=[view(mapped[key]) for key in batch]),[],
                    obj(dict(assessments=array(assessment,len(batch),len(batch)))),1536)
                if data is None:
                    data=dict(assessments=[dict(id=key,relation='uncertain',reason='invalid model response') for key in batch])
                actual=[r['id'] for r in data['assessments']]
                if len(actual)!=len(set(actual)) or set(actual)!=set(batch):
                    raise ValueError('Local relation comparison omitted or invented IDs')
                for row in data['assessments']:
                    pair_relations[(ids[i],row['id'])]=row['relation']
                    comparisons.append(dict(new_id=ids[i],**row))
            equal=[g for g in nearest if all(pair_relations.get((ids[i],member))=='equivalent' for member in groups[g])]
            if equal:
                groups[equal[0]].append(ids[i])
            else:
                groups.append([ids[i]])
            # Incremental checkpoint enables inspection while a 118-candidate replay runs.
            write_json(self.root/'magnet_v2'/(tag+'_dedup.json'),dict(processed=i+1,total=len(candidates),groups=groups,comparisons=comparisons,
                recall_limit=5,max_comparison_batch=5,descriptions=descriptions))
        return groups,dict(groups=groups,comparisons=comparisons,descriptions=descriptions)

    def visual_candidates(self,episodes):
        from ..prompts import magnet_construction
        candidates=[]
        for i,e in enumerate(episodes):
            if not eligible(e) or not source_images(e):
                continue
            record=self.describe(e)
            if record is None or not record['operations']:
                continue
            originals=source_images(e)
            if e['benchmark']=='mpdocvqa':
                pages=[]
                for step in e['steps']:
                    try:
                        answer=json.loads(step.get('text',''))
                    except ValueError:
                        continue
                    page=answer.get('answer_page_idx') if isinstance(answer,dict) else None
                    if type(page) is int and page not in pages:
                        pages.append(page)
                originals=[pair for pair in originals if pair[0] in pages][:2] if pages else originals[:2]
            else:
                originals=originals[:2]
            if not originals:
                continue
            reference_assets=[(f"Source {e['id']} original image {j}",self.root/a['path']) for j,a in originals]
            schema=obj(dict(regions=array(obj(dict(image_index=dict(enum=[j for j,_ in originals]),box=BOX,object=STRING,observation=STRING)),2)))
            data=self.call('grounded_regions_'+e['id'],magnet_construction.REGION_SYSTEM,
                dict(source_id=e['id'],question=e['task'],supplied_image_indices=[j for j,_ in originals],page_policy='historical recorded page evidence; no evaluation gold'),reference_assets,schema)
            if data is None:
                continue
            regions={};images=[];inspections={}
            for j,r in enumerate(data['regions']):
                try:
                    asset=crop(self.root,e,r,f'v2_region_{i}_{j}','function_visual_patch')
                except ValueError as exc:
                    write_json(self.root/'magnet_v2'/'rejected_regions'/f'{e["id"]}_{j}.json',dict(region=r,reason=str(exc)))
                    continue
                inspected=self.call('inspect_crop_'+e['id']+f'_{j}',prompts.INSPECT_CROP,{},
                    [('Unlabeled crop',self.root/asset['path'])],obj(dict(visible_text=dict(type='string',maxLength=240),
                        visible_objects=dict(type='string',maxLength=240),clipped_or_ambiguous=dict(type='boolean'))),384)
                check=None
                if inspected is not None and not inspected['clipped_or_ambiguous']:
                    check=self.call('match_crop_'+e['id']+f'_{j}',prompts.MATCH_CROP,
                        dict(proposed_component=dict(object=r['object'],observation=r['observation']),blind_reading=inspected),[],
                        obj(dict(supported=dict(type='boolean'),reason=dict(type='string',maxLength=200))),384)
                if not inspected_region_supported(r,inspected,check):
                    write_json(self.root/'magnet_v2'/'rejected_crops'/f'{e["id"]}_{j}.json',
                        dict(region=r,blind_reading=inspected,assessment=check,reason='crop is not clear supporting visual evidence'))
                    continue
                regions[f'region_{j}']=r
                inspections[f'region_{j}']=dict(blind_reading=inspected,assessment=check)
                images.append((f'Original region_{j}',self.root/asset['path']))
            if not regions:
                continue
            schema=obj(dict(associations=array(obj(dict(candidate_id=dict(enum=list(regions)),function=STRING,evidence=STRING,
                operation_index=dict(enum=list(range(len(record['operations'])))))),2)))
            rendered=[(f"Source {e['id']} generated output",self.root/a['path']) for a in e['images'] if a['role'].startswith('render')][:1]
            linked=self.call('inspected_associations_'+e['id'],magnet_construction.ASSOCIATION_SYSTEM+' Also cite operation_index from supplied source_operations for each association.',
                dict(episode={key:e[key] for key in ('id','task','feedback')},regions=regions,source_operations=record['operations']),reference_assets+rendered+images,schema)
            if linked is None:
                continue
            seen=set()
            for link in linked['associations']:
                key=link['candidate_id']
                if key not in regions or key in seen:
                    continue
                seen.add(key)
                r=regions[key]
                operation=record['operations'][link['operation_index']]
                checked=self.call('verify_inspected_visual_'+e['id']+'_'+key,prompts.VERIFY_STEP,
                    dict(step=link['function'],region=r,operation=operation),[],
                    obj(dict(supported=dict(type='boolean'),reason=dict(type='string',maxLength=240))),384)
                if checked is None or not checked['supported']:
                    write_json(self.root/'magnet_v2'/'rejected_associations'/f'{e["id"]}_{key}.json',dict(association=link,assessment=checked))
                    continue
                asset=crop(self.root,e,r,f'v2_patch_{len(candidates)}','function_visual_patch')
                candidates.append(dict(candidate_id=f'visual_{len(candidates)}',object=r['object'],function=link['function'],
                    observation=r['observation'],evidence=link['evidence'],source_ids=[e['id']],image=asset,
                    source_operation=operation,verification=checked,crop_inspection=inspections[key]))
        return candidates


def build(root,config,episodes):
    builder=Builder(root,config)
    records,similarities=builder.describe_sources(episodes)
    threshold=config.get('memory_construction',{}).get('magnet_v2_threshold')
    if threshold is None:
        raise ValueError('Calibrate magnet_v2_threshold on construction-only pairs before building')
    candidates=builder.workflows(records,similarities,threshold)
    groups,audit=builder.deduplicate(candidates,'workflows')
    mapped={c['candidate_id']:c for c in candidates}
    nodes=[]
    benchmark=episodes[0]['benchmark']
    for group in groups:
        variants=[mapped[key] for key in group]
        canonical=variants[0]
        text=json.dumps({k:v for k,v in canonical.items() if k not in ('step_verification','support')},ensure_ascii=False)
        nodes.append(entry(f'{benchmark}/magnet_v2/{canonical["candidate_id"]}',benchmark,
            sorted({s for c in variants for s in c['source_ids']}),text,memory_type='workflow',equivalent_variants=variants))
    visuals=builder.visual_candidates(episodes)
    visual_groups,_=builder.deduplicate(visuals,'visuals')
    mapped={c['candidate_id']:c for c in visuals}
    for i,group in enumerate(visual_groups):
        variants=[mapped[key] for key in group]
        images=[v['image'] for v in variants]
        # All variants stay in provenance; the common serving budget selects representatives.
        text=json.dumps(dict(associations=[{k:v[k] for k in ('object','function','observation','evidence','source_ids')} for v in variants[:2]]),ensure_ascii=False)
        nodes.append(entry(f'{benchmark}/magnet_v2/visual_{i}',benchmark,sorted({s for v in variants for s in v['source_ids']}),
            text,images[:2],memory_type='visual_function',visual_variants=images))
    write_json(Path(root)/'magnet_v2'/'build_summary.json',dict(protocol=VERSION,sources=len(records),workflow_candidates=len(candidates),
        workflow_nodes=len(groups),visual_candidates=len(visuals),visual_nodes=len(visual_groups),node_count=len(nodes)))
    return nodes
