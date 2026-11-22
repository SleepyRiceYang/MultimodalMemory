"""Offline five-baseline construction. KSI arms differ only in visual payload."""
from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import write_json
from ..models.local import LocalModel, image_hash, parse_object
from ..prompts import ksi_style, ksi_visual, magnet_construction, retrieval
from .bank import inside

PROTOCOL = 'paired_ksi_magnet_v1'
BASELINES = ('no_memory', 'raw_multimodal', 'ksi_style', 'ksi_visual', 'magnet_adapted')
STRING = {'type': 'string', 'maxLength': 700}
BOX = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000}, 'minItems': 4, 'maxItems': 4}


def obj(fields):
    return dict(type='object', properties=fields, required=list(fields), additionalProperties=False)


def array(item, maximum=10, minimum=0):
    return dict(type='array', items=item, maxItems=maximum, minItems=minimum)


def validate_response(value, schema, path='$'):
    """Validate the schema properties that commonly make a parsed response unusable."""
    expected = schema.get('type')
    if expected == 'object':
        if not isinstance(value, dict):
            raise ValueError(f'{path}: expected object')
        missing = set(schema.get('required', ())) - set(value)
        if missing:
            raise ValueError(f'{path}: missing required field(s): {", ".join(sorted(missing))}')
        properties = schema.get('properties', {})
        for key, child in properties.items():
            if key in value:
                validate_response(value[key], child, f'{path}.{key}')
    elif expected == 'array':
        if not isinstance(value, list):
            raise ValueError(f'{path}: expected array')
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', float('inf')):
            raise ValueError(f'{path}: array length {len(value)} violates item bounds')
        for i, child_value in enumerate(value):
            validate_response(child_value, schema.get('items', {}), f'{path}[{i}]')
    elif expected == 'string' and not isinstance(value, str):
        raise ValueError(f'{path}: expected string')
    elif expected == 'integer' and (type(value) is not int):
        raise ValueError(f'{path}: expected integer')
    elif expected == 'number' and (type(value) not in (int, float)):
        raise ValueError(f'{path}: expected number')
    elif expected == 'boolean' and not isinstance(value, bool):
        raise ValueError(f'{path}: expected boolean')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: value is outside the allowed enum')
    if 'const' in schema and value != schema['const']:
        raise ValueError(f'{path}: value does not match the required constant')


def parse_and_validate(response, schema):
    if response.get('finish_reason') == 'length':
        raise ValueError('response was truncated at the token limit')
    parsed = parse_object(response.get('text', ''))
    validate_response(parsed, schema)
    return parsed


def request(root, config, name, system, payload, images, schema):
    path = root / 'construction_calls' / (name + '.json')
    spec = dict(system=system, payload=payload, images=[[label, str(p)] for label, p in images],
                schema=schema, model=config['model'])
    saved = path.with_suffix('.request.json')
    if saved.exists():
        if json.loads(saved.read_text()) != spec:
            raise ValueError('Preserved construction request changed: ' + name)
    else:
        write_json(saved, spec)
    if path.exists():
        response = json.loads(path.read_text())
        try:
            return parse_and_validate(response, schema)
        except (ValueError, TypeError) as exc:
            # Keep the original response for diagnosis and retry from a new attempt.
            first_attempt = path.with_name(path.stem + '.attempt_1.json')
            if not first_attempt.exists():
                shutil.move(str(path), str(first_attempt))
            else:
                path.unlink()
            write_json(path.with_name(path.stem + '.attempt_1.error.json'),
                       {'error': str(exc), 'source': first_attempt.name})

    retries = max(0, int(config.get('memory_construction', {}).get('parse_retries', 3)))
    max_attempts = retries + 1
    # attempt_1 may hold a malformed response preserved from an earlier run.
    for attempt in range(1, max_attempts + 1):
        attempt_path = path.with_name(path.stem + f'.attempt_{attempt}.json')
        if attempt_path.exists():
            response = json.loads(attempt_path.read_text())
        else:
            print(f'CONSTRUCT {name} attempt={attempt}/{max_attempts}', flush=True)
            response = LocalModel(config['model']).generate(system, json.dumps(payload, ensure_ascii=False), images,
                max_tokens=config['model']['construction_max_tokens'], json_schema=schema,
                progress_path=path.with_name(path.stem + f'.attempt_{attempt}.partial.json'))
            write_json(attempt_path, response)
        try:
            parsed = parse_and_validate(response, schema)
        except (ValueError, TypeError) as exc:
            write_json(path.with_name(path.stem + f'.attempt_{attempt}.error.json'),
                       {'error': str(exc), 'finish_reason': response.get('finish_reason')})
            continue
        write_json(path, response)
        return parsed
    write_json(path.with_suffix('.failed.json'), {
        'status': 'skipped_after_parse_retries', 'attempts': max_attempts,
        'errors': [json.loads(path.with_name(path.stem + f'.attempt_{i}.error.json').read_text())
                   for i in range(1, max_attempts + 1)
                   if path.with_name(path.stem + f'.attempt_{i}.error.json').exists()]})
    print(f'CONSTRUCT SKIP {name}: invalid response after {max_attempts} attempts', flush=True)
    return None


def source_images(episode):
    return [(i, a) for i, a in enumerate(episode['images']) if not a['role'].startswith('render')]


def assets(root, episodes, original_only=False):
    return [(f"Source {e['id']} image {i} role {a['role']}", inside(root, a['path']))
            for e in episodes for i, a in (source_images(e) if original_only else enumerate(e['images']))]


def entry(node_id, benchmark, source_ids, text, images=None, **extra):
    return dict(id=node_id, benchmark=benchmark, source_ids=source_ids, text=text,
                index_text=text, images=images or [], **extra)


def crop(root, episode, item, name, role):
    image_index = item['image_index']
    originals = dict(source_images(episode))
    if image_index not in originals:
        raise ValueError('Visual evidence must refer to an original observation')
    if any(type(v) is not int for v in item['box']):
        raise ValueError('Evidence coordinates must use the integer 0..1000 grid')
    box = [v/1000 for v in item['box']]
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError('Invalid evidence box')
    target = root / 'patches' / (name + '.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(inside(root, originals[image_index]['path'])) as image:
        pixels = (int(box[0]*image.width), int(box[1]*image.height), int(box[2]*image.width), int(box[3]*image.height))
        if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
            raise ValueError('Empty evidence region')
        image.crop(pixels).save(target)
    return dict(path=str(target.relative_to(root)), role=role, source_id=episode['id'],
                source_image_index=image_index, box=box, observation=item['observation'])


def attach_visual(root, config, nodes, episodes):
    by_id = {e['id']: e for e in episodes}
    paired = copy.deepcopy(nodes)
    for i, node in enumerate(paired):
        group = [by_id[s] for s in node['source_ids']]
        item = obj(dict(node_id=dict(const=node['id']), source_id=dict(enum=node['source_ids']),
            image_index=dict(type='integer', minimum=0), box=BOX, observation=STRING, relevance=STRING))
        data = request(root, config, f"{node['benchmark']}_ksi_visual_{i}", ksi_visual.ATTACH_SYSTEM,
            dict(knowledge=node, sources=[dict(id=e['id'], observations=[dict(index=j, role=a['role']) for j, a in source_images(e)]) for e in group]),
            assets(root, group, True), obj(dict(attachments=array(item, 2))))
        if data is None:
            continue
        for j, attachment in enumerate(data['attachments']):
            if attachment['node_id'] != node['id'] or attachment['source_id'] not in node['source_ids']:
                raise ValueError('Visual evidence cites a source outside the knowledge node')
            node['images'].append(crop(root, by_id[attachment['source_id']], attachment,
                f"{node['benchmark']}_ksi_{i}_{j}", 'ksi_visual_evidence'))
    return paired


def maximal_cliques(similarities, threshold, max_clusters=256):
    neighbors = {i: {j for j in range(len(similarities)) if i != j and similarities[i, j] >= threshold}
                 for i in range(len(similarities))}
    result = []
    def visit(found, possible, excluded):
        if not possible and not excluded:
            if len(found) >= 2:
                result.append(sorted(found))
                if len(result) > max_clusters:
                    raise ValueError('Too many MAGNET clusters; configure clustering before rebuilding')
            return
        pivot = max(possible | excluded, key=lambda i: len(possible & neighbors[i]), default=None)
        for v in sorted(possible - (neighbors[pivot] if pivot is not None else set())):
            visit(found | {v}, possible & neighbors[v], excluded & neighbors[v])
            possible.remove(v)
            excluded.add(v)
    visit(set(), set(neighbors), set())
    return sorted(result)


def eligible(episode):
    return (episode.get('success') is True and not episode.get('generation_truncated', False)
            and not episode.get('feedback', {}).get('generation_truncated', False))


def group_candidates(root, config, name, candidates):
    ids = [c['candidate_id'] for c in candidates]
    if len(ids) < 2:
        return [[i] for i in ids]
    result = request(root, config, name, magnet_construction.GROUP_SYSTEM,
        dict(candidates=candidates), [], obj(dict(groups=array(array(dict(enum=ids), len(ids), 1), len(ids), 1))))
    if result is None:
        return [[i] for i in ids]
    flat = [i for group in result['groups'] for i in group]
    if len(flat) != len(set(flat)) or set(flat) != set(ids):
        raise ValueError('Grouping lost, duplicated or invented candidates')
    return result['groups']


def build_magnet(root, config, episodes):
    if config.get('memory_construction', {}).get('magnet_protocol') == 'described_complete_link_v2':
        from .magnet_v2 import build
        return build(root, config, episodes)
    from .retrieval import post, index_text
    benchmark = episodes[0]['benchmark']
    admitted = [e for e in episodes if eligible(e)]
    audit = dict(admitted=[e['id'] for e in admitted], rejected=[dict(id=e['id'], reason='failed, unknown or truncated source') for e in episodes if not eligible(e)], clusters=[])
    nodes, workflows = [], []
    if len(admitted) >= 2:
        descriptions, described = [], []
        for i, e in enumerate(admitted):
            fields = request(root, config, f'{benchmark}_magnet_task_{i}', retrieval.QUERY_SYSTEM,
                dict(benchmark=benchmark, question=e['task']), assets(root, [e], True), retrieval.FIELDS)
            if fields is None:
                audit.setdefault('description_failures', []).append(e['id'])
                continue
            description = index_text(fields)
            url = config['retrieval']['embedding_url'].rstrip('/')
            if post(url + '/tokens', dict(texts=[description]))[0]['tokens'] > 384:
                fields = request(root, config, f'{benchmark}_magnet_task_{i}_compressed', retrieval.QUERY_SYSTEM + retrieval.COMPRESS_SUFFIX,
                    dict(description=fields), [], retrieval.FIELDS)
                if fields is None:
                    audit.setdefault('description_failures', []).append(e['id'])
                    continue
                description = index_text(fields)
            if post(url + '/tokens', dict(texts=[description]))[0]['tokens'] > 384:
                raise ValueError('MAGNET source description exceeds encoder budget')
            descriptions.append(description)
            described.append(e)
        if descriptions:
            vectors = np.asarray(post(url + '/encode', dict(texts=descriptions))['embeddings'])
            cliques = maximal_cliques(vectors @ vectors.T, config['memory_construction'].get('magnet_cluster_cosine', 0.75))
        else:
            cliques = []
        audit['clusters'] = [[described[i]['id'] for i in clique] for clique in cliques]
        seen = set()
        for i, clique in enumerate(cliques):
            # Bounded batches avoid putting an entire benchmark's images in one request.
            for start in range(0, len(clique), 8):
                group = [described[j] for j in clique[start:start+8]]
                if len(group) < 2:
                    continue
                ids = [e['id'] for e in group]
                workflow = obj(dict(title=STRING, applicability=array(STRING), exclusions=array(STRING), parameters=array(STRING),
                    steps=array(STRING, 8, 3), source_ids=array(dict(enum=ids), len(ids), 2)))
                result = request(root, config, f'{benchmark}_magnet_workflows_{i}_{start}', magnet_construction.WORKFLOW_SYSTEM,
                    dict(episodes=group), assets(root, group), obj(dict(workflows=array(workflow, 6))))
                if result is None:
                    continue
                for w in result['workflows']:
                    if len(set(w['source_ids'])) < 2 or not set(w['source_ids']) <= set(ids):
                        raise ValueError('Workflow lacks two distinct supporting tasks in its cluster')
                    key = re.sub(r'\W+', ' ', json.dumps([w['title'], w['steps']], ensure_ascii=False).lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    workflows.append({**w, 'candidate_id': f'workflow_{len(workflows)}'})
        mapped = {w['candidate_id']: w for w in workflows}
        groups = group_candidates(root, config, benchmark+'_magnet_workflow_groups', workflows)
        audit['workflow_groups'] = groups
        for group in groups:
            w = mapped[group[0]]
            sources = sorted({s for key in group for s in mapped[key]['source_ids']})
            nodes.append(entry(f'{benchmark}/magnet/{w["candidate_id"]}', benchmark, sources,
                json.dumps({k:v for k,v in w.items() if k != 'candidate_id'}, ensure_ascii=False), memory_type='workflow'))
    patches = []
    for i, e in enumerate(admitted):
        original = source_images(e)
        if not original:
            continue
        schema = obj(dict(regions=array(obj(dict(image_index=dict(enum=[j for j, _ in original]),
            box=BOX, object=STRING, observation=STRING)), 2)))
        observed = request(root, config, f'{benchmark}_magnet_regions_{i}', magnet_construction.REGION_SYSTEM,
            dict(source_id=e['id'], question=e['task']), assets(root, [e], True), schema)
        if observed is None:
            audit.setdefault('region_failures', []).append(e['id'])
            continue
        regions = {f'region_{j}': {**r, 'source_id':e['id']} for j,r in enumerate(observed['regions'])}
        if not regions:
            continue
        # Validate and show actual crops before associating them with historical code.
        crops = [crop(root,e,r,f'{benchmark}_region_{i}_{key}','candidate_region') for key,r in regions.items()]
        association_schema = obj(dict(associations=array(obj(dict(candidate_id=dict(enum=list(regions)),function=STRING,evidence=STRING)),2)))
        linked = request(root, config, f'{benchmark}_magnet_associations_{i}', magnet_construction.ASSOCIATION_SYSTEM,
            dict(episode=e, regions=regions), assets(root,[e]) + [(f'Original region {key}',root/a['path']) for key,a in zip(regions,crops)], association_schema)
        if linked is None:
            audit.setdefault('association_failures', []).append(e['id'])
            continue
        used = set()
        for link in linked['associations']:
            key=link['candidate_id']
            if key not in regions or key in used:
                raise ValueError('Unknown or repeated visual association')
            used.add(key)
            patches.append({**regions[key], 'function':link['function'], 'evidence':link['evidence'], 'candidate_id':f'patch_{len(patches)}'})
    groups = group_candidates(root, config, benchmark+'_magnet_visual_groups', patches)
    mapped = {p['candidate_id']: p for p in patches}
    by_id = {e['id']: e for e in admitted}
    for group in groups:
        values = [mapped[i] for i in group]
        # Keep two representatives; all omitted variants and evidence remain in maintenance audit.
        images = [crop(root, by_id[p['source_id']], p, benchmark+'_magnet_'+p['candidate_id'], 'function_visual_patch') for p in values[:2]]
        text = json.dumps(dict(associations=[{k:p[k] for k in ('object', 'function', 'observation', 'evidence')} for p in values[:2]]), ensure_ascii=False)
        nodes.append(entry(f'{benchmark}/magnet/visual_{len(nodes)}', benchmark,
            sorted({p['source_id'] for p in values}), text, images, memory_type='visual_function'))
    audit.update(workflow_candidates=workflows, visual_candidates=patches, visual_groups=groups, node_count=len(nodes),
                 single_source_workflow_policy='no shared workflow without two distinct successful tasks')
    write_json(root / (benchmark+'_magnet_maintenance.json'), audit)
    return nodes


def build_knowledge(root, config, episodes):
    from .build import construction_schema
    knowledge = []
    for i, e in enumerate(episodes):
        review_schema = construction_schema('paired_ksi_forum', [e])
        review_schema['properties']['claims']['minItems'] = 0
        review_schema = obj(dict(observed_support={'type':'string','maxLength':1400}, claims=review_schema['properties']['claims']))
        discussion = request(root, config, f'{e["benchmark"]}_ksi_forum_{i}', ksi_style.PAIRED_REVIEW_SYSTEM,
            dict(episodes=[e]), assets(root, [e]), review_schema)
        if discussion is None or not discussion['claims']:
            continue  # Distillation cannot invent knowledge when review found no supported claim.
        schema = construction_schema('paired_ksi_distill', [e])
        schema['properties']['knowledge']['minItems'] = 0
        schema['properties']['knowledge']['maxItems'] = 2
        result = request(root, config, f'{e["benchmark"]}_ksi_distill_{i}', ksi_style.PAIRED_DISTILL_SYSTEM,
            dict(episodes=[e], review=discussion), assets(root, [e]), schema)
        if result is None:
            continue
        knowledge.extend(result['knowledge'])
    level = 0
    while len(episodes) > 1 and len(knowledge) > 1:
        merged = []
        for start in range(0, len(knowledge), 8):
            items = knowledge[start:start+8]
            ids = sorted({s for k in items for s in k['evidence_ids']})
            schema = construction_schema('cross_ksi_distill', [dict(id=s) for s in ids])
            schema['properties']['knowledge']['minItems'] = 0
            result = request(root, config, f'{episodes[0]["benchmark"]}_ksi_merge_{level}_{start}',
                ksi_style.PAIRED_DISTILL_SYSTEM + ksi_style.CROSS_TASK_SUFFIX.format(task_count=len(ids)),
                dict(knowledge_from_tasks=items), [], schema)
            if result is None:
                continue
            merged.extend(result['knowledge'])
        knowledge = merged
        if len(knowledge) <= 6:
            break
        level += 1
    allowed = {e['id'] for e in episodes}
    accepted, rejected = [], []
    for item in knowledge:
        if not item['evidence_ids'] or not set(item['evidence_ids']) <= allowed:
            raise ValueError('Unknown KSI source')
        # A concrete source pixel coordinate cannot be a reusable plotting instruction.
        # This narrow gate does not reject mathematical coordinates or normalized axis conventions.
        positions = re.findall(r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)', item['claim'])
        if episodes[0]['benchmark'] == 'vision2code' and any(abs(float(x)) > 1 or abs(float(y)) > 1 for x,y in positions):
            rejected.append(dict(knowledge=item, reason='literal source coordinates in reusable claim'))
        else:
            accepted.append(item)
    write_json(root/(episodes[0]['benchmark']+'_ksi_admission.json'),dict(accepted=len(accepted),rejected=rejected))
    return [entry(f'{episodes[0]["benchmark"]}/ksi_style/{i}', episodes[0]['benchmark'], item['evidence_ids'],
                  json.dumps(item, ensure_ascii=False)) for i, item in enumerate(accepted)]


def build_bank(root, config, episodes):
    path = root / 'bank.json'
    if path.exists():
        raise FileExistsError(path)
    if not episodes or len({e['id'] for e in episodes}) != len(episodes):
        raise ValueError('Need unique construction episodes')
    if not set(config['baselines']) <= set(BASELINES):
        raise ValueError('This protocol supports the five current baselines only')
    methods = {m: [] for m in config['baselines'] if m != 'no_memory'}
    for benchmark in config['benchmarks']:
        group = [e for e in episodes if e['benchmark'] == benchmark]
        if not group:
            raise ValueError('Missing benchmark construction episodes')
        if 'raw_multimodal' in methods:
            methods['raw_multimodal'] += [entry(f'{benchmark}/raw_multimodal/{i}', benchmark, [e['id']],
                json.dumps({k:e[k] for k in ('task', 'steps', 'feedback')}, ensure_ascii=False), copy.deepcopy(e['images'])) for i,e in enumerate(group)]
        if 'ksi_style' in methods or 'ksi_visual' in methods:
            nodes = build_knowledge(root, config, group)
            if 'ksi_style' in methods:
                methods['ksi_style'] += nodes
            if 'ksi_visual' in methods:
                methods['ksi_visual'] += attach_visual(root, config, nodes, group)
        if 'magnet_adapted' in methods:
            methods['magnet_adapted'] += build_magnet(root, config, group)
    paths = {a['path'] for e in episodes for a in e['images']}
    paths.update(a['path'] for nodes in methods.values() for n in nodes for a in n['images'])
    write_json(path, dict(schema_version=2, protocol=PROTOCOL, mode=config['data_mode'], frozen=True,
        sources=[dict(id=e['id'], benchmark=e['benchmark'], provenance=e['provenance'], image_hashes=[image_hash(inside(root,a['path'])) for a in e['images']]) for e in episodes],
        methods=methods, asset_hashes={p:image_hash(inside(root,p)) for p in sorted(paths)}))
    return path
