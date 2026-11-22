"""Task-only MiniCPM cosine recall with a versioned, process-safe JSON/NumPy index."""
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import requests

POLICY = 'task_only_minicpm_cosine_top10_llm_up_to_three_or_null_v3'


def service_info(args):
    response = requests.get(args['embedding_url'].rstrip('/') + '/health', timeout=30)
    response.raise_for_status()
    info = response.json()
    if not (info.get('ready') and info.get('model') == 'openbmb/MiniCPM-Embedding'
            and info.get('dimension') == 2304 and info.get('normalized') is True
            and info.get('max_tokens') == 512 and info.get('query_prefix') == 'Query: '
            and info.get('pooling') == 'model_scaled_hidden_states_then_masked_mean'):
        raise ValueError('Embedding service does not match the configured MiniCPM protocol')
    return info


def encode(texts, args, query=False):
    response = requests.post(args['embedding_url'].rstrip('/') + '/encode',
                             json={'texts': texts, 'query': query}, timeout=args['embedding_timeout'])
    response.raise_for_status()
    data = response.json()
    values = np.asarray(data['embeddings'], dtype=np.float32)
    if values.shape != (len(texts), 2304) or not np.isfinite(values).all():
        raise ValueError('Invalid embedding shape or values')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError('Embedding service returned a zero vector')
    return values / norms, data


def index_metadata(metadata):
    return [{'skill': Path(m.directory).name, 'name': m.name, 'description': m.description,
             'text': m.name + '\n' + m.description} for m in sorted(metadata, key=lambda m: str(m.directory))]


def load_index(metadata, args):
    info = service_info(args)
    entries = index_metadata(metadata)
    if not entries or len({e['skill'] for e in entries}) != len(entries):
        raise ValueError('Expected non-empty, unambiguous skill IDs')
    identity = {'entries': entries, 'model': {k: info[k] for k in
                ('model', 'revision', 'dimension', 'normalized', 'max_tokens', 'pooling', 'query_prefix')},
                'document_format': 'skill_name newline description; no query prefix'}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(args['embedding_index_dir']) / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'index.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        meta_path, vector_path = folder / 'metadata.json', folder / 'vectors.npy'
        if not meta_path.exists() or not vector_path.exists():
            vectors, details = encode([e['text'] for e in entries], args)
            temporary = folder / 'vectors.tmp.npy'
            np.save(temporary, vectors, allow_pickle=False)
            temporary.replace(vector_path)
            temporary = folder / 'metadata.tmp.json'
            temporary.write_text(json.dumps({'fingerprint': fingerprint, **identity,
                'original_token_counts': details['original_token_counts'],
                'truncated': details['truncated']}, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(meta_path)
        saved = json.loads(meta_path.read_text())
        vectors = np.load(vector_path, allow_pickle=False)
        if saved['fingerprint'] != fingerprint or saved['entries'] != entries:
            raise ValueError('Skill index metadata mismatch')
        if vectors.shape != (len(entries), 2304) or not np.isfinite(vectors).all():
            raise ValueError('Skill vector cache corrupted')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError('Cached skill embeddings are not normalized')
    return entries, vectors, {'fingerprint': fingerprint, 'path': str(folder), 'embedding': identity['model']}


def cosine_top_k(entries, vectors, query, top_k=10):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    scores = vectors @ query
    order = sorted(range(len(entries)), key=lambda i: (-float(scores[i]), entries[i]['skill']))[:top_k]
    return [{**{k: entries[i][k] for k in ('skill', 'name', 'description')},
             'cosine_similarity': float(scores[i])} for i in order]


def shortlist(instruction, metadata, args):
    entries, vectors, index = load_index(metadata, args)
    query, detail = encode([instruction], args, query=True)
    return cosine_top_k(entries, vectors, query[0], args['retrieval_top_k']), {
        **index, 'library_size': len(entries), 'query_original_tokens': detail['original_token_counts'][0],
        'query_truncated': detail['truncated'][0], 'recall_threshold': None}


def parse_decision(raw, candidates, max_skills=3):
    import re
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    decision = json.loads(cleaned)
    if not isinstance(decision, dict) or 'skills' not in decision:
        raise ValueError('Expected an object with skills: a list or null')
    skills = decision['skills']
    if skills is None:
        skills = []
    if (not isinstance(skills, list) or len(skills) > max_skills
            or not all(isinstance(s, str) for s in skills)):
        raise ValueError('Expected up to three exact skill names or null')
    if len(set(skills)) != len(skills) or any(s not in {c['skill'] for c in candidates} for s in skills):
        raise ValueError('LLM selected duplicate skills or a skill outside the dense candidate set')
    if not isinstance(decision.get('reason'), str):
        raise ValueError('Expected a textual selection reason')
    return skills, decision
