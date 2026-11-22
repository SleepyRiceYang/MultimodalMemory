"""Frozen, benchmark-isolated dense recall and shared listwise reranking."""
from __future__ import annotations
import json
import copy
import urllib.request
from pathlib import Path
import numpy as np
from ..config import write_json
from ..models.local import LocalModel, parse_object
from ..prompts import retrieval as prompts


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def index_text(fields):
    return '\n'.join(f'{label}: {fields[key]}' for label, key in [('Topic','topic'), ('Scope','scope'), ('Question Anchor','question_anchor')])


def memory_scope(entry):
    """Keep declared applicability separate from the historical question's instance values."""
    try:
        data = json.loads(entry['text'])
    except (ValueError, TypeError):
        data = {}
    if isinstance(data, dict) and 'conditions' in data:
        return json.dumps({'conditions': data['conditions'], 'exclusions': data.get('exclusions', [])}, ensure_ascii=False)
    if isinstance(data, dict) and 'applicability' in data:
        return json.dumps({'conditions': data['applicability'], 'exclusions': data.get('exclusions', [])}, ensure_ascii=False)
    return prompts.EXAMPLE_SCOPE


def validate_ranks(rows, ids):
    actual = [r['id'] for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != set(ids):
        raise ValueError('Reranker must assess each candidate exactly once')
    for row in rows:
        if row['grade'] not in (0,1,2,3) or row['duplicate_of'] not in ['', *ids] or row['duplicate_of'] == row['id']:
            raise ValueError('Invalid rerank grade or duplicate reference')
    return rows


def bounded_rank_schema(ids):
    schema = copy.deepcopy(prompts.RANK_SCHEMA)
    rows = schema['properties']['assessments']
    rows.update(minItems=len(ids), maxItems=len(ids))
    props = rows['items']['properties']
    props['id'] = {'enum': ids}
    props['duplicate_of'] = {'enum': ['', *ids]}
    for key in ('reason', 'condition_evidence'):
        props[key]['maxLength'] = 320
    return schema


class DenseRetriever:
    def __init__(self, bank, config):
        self.bank, self.config = bank, config
        self.options = config['retrieval']
        self.root = bank.root / 'retrieval'
        self.url = self.options['embedding_url'].rstrip('/')
        self.model = LocalModel({**config['model'], 'temperature': 0})
        self.manifest = json.loads((bank.root / 'dataset_manifest.json').read_text())
        with urllib.request.urlopen(self.url + '/health', timeout=30) as response:
            self.encoder = json.load(response)
        expected = self.options.get('embedding_model')
        if expected and expected != self.encoder['model']:
            raise ValueError('Unexpected embedding model')
        self.nodes = self.prepare()
        self.frozen_index = json.loads((self.root / 'index.json').read_text())

    def verify_frozen(self):
        if json.loads((self.root / 'index.json').read_text()) != self.frozen_index:
            raise ValueError('Frozen retrieval index changed')
        if not np.array_equal(np.load(self.root / 'vectors.npy', allow_pickle=False), self.vectors):
            raise ValueError('Frozen embedding vectors changed')
        return True

    def count(self, text, query=False):
        return post(self.url + '/tokens', {'texts': [text], 'query': query})[0]['tokens']

    def generate(self, name, system, payload, images, schema, max_tokens=2048):
        path = self.root / 'calls' / (name + '.json')
        prompt = json.dumps(payload, ensure_ascii=False)
        request = {'system': system, 'prompt': prompt, 'images': [[label,str(path)] for label,path in images], 'schema': schema, 'max_tokens': max_tokens, 'model': self.config['model']}
        request_path = path.with_suffix('.request.json')
        if path.exists():
            if json.loads(request_path.read_text()) != request:
                raise ValueError(f'Cached retrieval request changed: {name}')
            response = json.loads(path.read_text())
        else:
            write_json(request_path, request)
            print(f'RETRIEVAL {name}', flush=True)
            response = self.model.generate(system, prompt, images, max_tokens=max_tokens, json_schema=schema, progress_path=path.with_suffix('.partial.json'))
            write_json(path, response)
        if response['finish_reason'] == 'length':
            raise ValueError(f'Truncated retrieval generation: {name}')
        return parse_object(response['text'])

    def describe(self, task, source=False):
        name = ('source_queries/' if source else 'queries/') + task['id']
        # Fixed first pages only; never gold answer pages. Original solve still sees all pages.
        paths = task['images'][:self.options.get('query_max_images', 2)]
        images = [(f'Current image/page {i}', Path(p)) for i,p in enumerate(paths)]
        payload = {'benchmark': task['benchmark'], 'question': task['prompt'], 'visible_pages': list(range(len(paths))), 'total_pages': len(task['images'])}
        try:
            fields = self.generate(name, prompts.QUERY_SYSTEM, payload, images, prompts.FIELDS)
            text = index_text(fields)
            if self.count(text, True) > 192:
                fields = self.generate(name + '_compressed', prompts.QUERY_SYSTEM + prompts.COMPRESS_SUFFIX, {'description': fields}, [], prompts.FIELDS)
                text = index_text(fields)
            if self.count(text, True) > 192:
                raise ValueError('Query still exceeds 192 tokens')
            return {'fields': fields, 'text': text, 'fallback': False, 'visible_pages': payload['visible_pages']}
        except Exception as exc:
            raw = task['prompt']
            fallback = self.count(raw, True) <= 192
            result = {'fields': None, 'text': raw if fallback else '', 'fallback': True, 'error': str(exc), 'query_rewrite_failed': True}
            write_json(self.root / (name + '_failure.json'), result)
            return result

    def prepare(self):
        entries = [e for group in self.bank.data['methods'].values() for e in group]
        guarded_scope = self.options.get('scope_policy') == 'memory_conditions_v1'
        index_system = prompts.INDEX_SYSTEM + (prompts.SCOPE_BOUNDARY_SUFFIX if guarded_scope else '')
        signature = {'options': self.options, 'encoder': self.encoder, 'entries': entries, 'model': self.config['model'], 'prompts': {'query': prompts.QUERY_SYSTEM, 'index': index_system, 'compress': prompts.COMPRESS_SUFFIX}}
        if guarded_scope:
            signature['prompts']['example_scope'] = prompts.EXAMPLE_SCOPE
        path = self.root / 'index.json'
        if path.exists():
            frozen = json.loads(path.read_text())
            if frozen['signature'] != signature:
                raise ValueError('Frozen retrieval index configuration/content changed; use a new run')
            self.vectors = np.load(self.root / 'vectors.npy', allow_pickle=False)
            if self.vectors.shape != (len(frozen['nodes']), 1024):
                raise ValueError('Invalid frozen embedding matrix')
            return frozen['nodes']
        sources = {t['id']: t for t in self.manifest['memory']}
        descriptions = {key: self.describe(task, True) for key,task in sources.items()}
        nodes = []
        # Paired KSI arms share IDs/text/index; encode each semantic node exactly once.
        unique = {}
        for entry in entries:
            if entry['id'] in unique and any(unique[entry['id']][k] != entry[k] for k in ('text', 'source_ids', 'benchmark')):
                raise ValueError('Conflicting retrieval node identity')
            unique.setdefault(entry['id'], entry)
        for i, entry in enumerate(unique.values()):
            payload = {'memory': entry['text'], 'source_questions': [descriptions[s] for s in entry['source_ids']]}
            fields = self.generate(f'index/node_{i}', index_system, payload, [], prompts.FIELDS)
            if guarded_scope:
                fields['scope'] = memory_scope(entry)
            text = index_text(fields)
            if self.count(text) > 384:
                fields = self.generate(f'index/node_{i}_compressed', index_system + prompts.COMPRESS_SUFFIX, {'description': fields}, [], prompts.FIELDS)
                if guarded_scope:
                    fields['scope'] = memory_scope(entry)
                text = index_text(fields)
            if self.count(text) > 384:
                raise ValueError('Index still exceeds 384 tokens')
            nodes.append({'id': entry['id'], 'benchmark': entry['benchmark'], 'fields': fields, 'text': text})
        vectors = post(self.url + '/encode', {'texts': [n['text'] for n in nodes]})['embeddings'] if nodes else []
        self.vectors = np.asarray(vectors, dtype=np.float32).reshape(len(nodes), 1024)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / 'vectors.pending.npy'
        np.save(temporary, self.vectors)
        temporary.replace(self.root / 'vectors.npy')
        write_json(path, {'signature': signature, 'nodes': nodes})
        return nodes

    def rank(self, method, task):
        query = self.describe(task)
        if not query['text']:
            return [], {'query': query, 'candidates': [], 'status': 'query_rewrite_failed'}
        vector = np.asarray(post(self.url + '/encode', {'texts': [query['text']], 'query': True})['embeddings'][0], dtype=np.float32)
        entry_map = {e['id']: e for e in self.bank.data['methods'][method] if e['benchmark'] == task['benchmark']}
        candidates = [(float(self.vectors[i] @ vector), node) for i,node in enumerate(self.nodes) if node['id'] in entry_map and node['benchmark'] == task['benchmark']]
        candidates.sort(key=lambda pair: (-pair[0],pair[1]['id']))
        candidates = candidates[:self.options.get('top_n',20)]
        if not candidates:
            return [], {'query': query, 'candidates': []}
        if self.options.get('ranking') == 'embedding_cosine':
            # Budgeting selects up to top_k whole entries in this deterministic order.
            trace = {'index': 'topic_scope_question_anchor',
                     'retrieval': 'normalized_embedding_cosine', 'ranking': 'embedding_cosine',
                     'reranker': None, 'encoder': self.encoder, 'query': query,
                     'candidates': [{'id': node['id'], 'cosine': score} for score, node in candidates]}
            write_json(self.root / 'traces' / method / (task['id'] + '.json'), trace)
            return [(score, entry_map[node['id']]) for score, node in candidates], trace
        payloads = []
        for score,node in candidates:
            excerpt = post(self.url + '/tokens', {'texts': [entry_map[node['id']]['text']], 'limit': 128})[0]
            payloads.append({'id': node['id'], **node['fields'], 'memory_excerpt': excerpt['text']})
        images = [(f'Current image/page {i}',Path(p)) for i,p in enumerate(task['images'][:self.options.get('query_max_images',2)])]
        ranking_method = 'ksi_style' if method == 'ksi_visual' and 'ksi_style' in self.bank.data['methods'] else method
        rerank_system = prompts.RERANK_SYSTEM + (prompts.SCOPE_BOUNDARY_SUFFIX if self.options.get('scope_policy') == 'memory_conditions_v1' else '')
        schema = bounded_rank_schema([n['id'] for _, n in candidates]) if self.options.get('bounded_rerank') else prompts.RANK_SCHEMA
        result = self.generate(f'rerank/{ranking_method}/{task["id"]}', rerank_system, {'question': task['prompt'], 'query': query, 'candidates': payloads}, images, schema, 4096)
        rows = validate_ranks(result['assessments'], list(entry_map[n['id']]['id'] for _,n in candidates))
        ranks = {r['id']: r for r in rows}
        ranked = [(score, entry_map[node['id']]) for score,node in candidates if ranks[node['id']]['grade'] >= 2 and not ranks[node['id']]['scope_conflict']]
        # Retain one representative of true duplicate groups; choose grade, cosine, ID.
        ranked.sort(key=lambda pair: (-ranks[pair[1]['id']]['grade'], -pair[0], pair[1]['id']))
        accepted, groups = [], []
        for score,entry in ranked:
            node_id = entry['id']
            duplicate = ranks[node_id]['duplicate_of']
            if any(node_id in group or (duplicate and duplicate in group) for group in groups):
                continue
            groups.append({node_id, duplicate} - {''})
            accepted.append((score,entry))
        trace = {'index': 'topic_scope_question_anchor', 'retrieval': 'normalized_embedding_cosine', 'encoder': self.encoder, 'query': query, 'candidates': [{'id': n['id'], 'cosine': s, **ranks[n['id']]} for s,n in candidates], 'reranker': self.config['model']['name']}
        write_json(self.root / 'traces' / method / (task['id'] + '.json'), trace)
        return accepted, trace
