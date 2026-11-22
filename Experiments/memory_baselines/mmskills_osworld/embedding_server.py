"""Pinned MiniCPM-Embedding service using the model card's weighted-mean implementation."""
import json
import os
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = Path(os.getenv('EMBEDDING_MODEL', '/data/pretrained_models/MiniCPM-Embedding'))
QUERY_PREFIX = 'Query: '
MAX_LENGTH = 512
DIMENSION = 2304
BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '2'))
app = FastAPI()
lock = threading.Lock()
model = tokenizer = None
manifest = None


@app.on_event('startup')
def load_model():
    global model, tokenizer, manifest
    manifest = json.loads((MODEL_PATH / 'download_manifest.json').read_text())
    if manifest['repo'] != 'openbmb/MiniCPM-Embedding':
        raise ValueError('Unexpected embedding model')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(str(MODEL_PATH), local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.float16).to('cuda:0').eval()
    torch.set_num_threads(4)


class EncodeRequest(BaseModel):
    texts: list[str]
    query: bool = False


@app.get('/health')
def health():
    return {'ready': model is not None, 'model': 'openbmb/MiniCPM-Embedding',
            'model_path': str(MODEL_PATH), 'revision': manifest['revision'] if manifest else None,
            'dimension': DIMENSION, 'normalized': True, 'max_tokens': MAX_LENGTH,
            'pooling': 'model_scaled_hidden_states_then_masked_mean', 'query_prefix': QUERY_PREFIX,
            'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'batch_size': BATCH_SIZE}


@app.post('/encode')
def encode(body: EncodeRequest):
    if model is None:
        raise HTTPException(503, 'Model not loaded')
    if not body.texts or len(body.texts) > 512 or any(not t.strip() for t in body.texts):
        raise HTTPException(422, 'Expected 1-512 non-empty texts')
    vectors, counts = [], []
    with lock, torch.inference_mode():
        texts = [(QUERY_PREFIX if body.query else '') + text for text in body.texts]
        counts = [len(tokenizer.encode(text)) for text in texts]
        for start in range(0, len(texts), BATCH_SIZE):
            batch = tokenizer(texts[start:start + BATCH_SIZE], max_length=MAX_LENGTH,
                padding=True, truncation=True, return_tensors='pt', return_attention_mask=True).to('cuda:0')
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask']
            # MiniCPM scales hidden states inside forward; do not apply positional weights again.
            reps = (hidden * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            reps = torch.nn.functional.normalize(reps, p=2, dim=1)
            if reps.shape[1] != DIMENSION or not torch.isfinite(reps).all():
                raise HTTPException(500, 'Invalid embedding output')
            vectors.extend(reps.cpu().tolist())
    return {'embeddings': vectors, 'original_token_counts': counts,
            'truncated': [n > MAX_LENGTH for n in counts], 'max_tokens': MAX_LENGTH}
