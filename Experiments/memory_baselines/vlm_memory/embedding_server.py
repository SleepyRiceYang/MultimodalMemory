"""Local BGE CLS/normalized embedding service; run in the torch environment."""
import os
import threading
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

MODEL = os.getenv('EMBEDDING_MODEL', '/data/pretrained_models/BAAI-bge-large-en-v1.5')
PREFIX = 'Represent this sentence for searching relevant passages: '
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModel.from_pretrained(MODEL, local_files_only=True).to('cuda:0').eval()
lock = threading.Lock()
app = FastAPI()

class Request(BaseModel):
    texts: list[str]
    query: bool = False
    limit: int | None = None

@app.get('/health')
def health():
    return {'model': MODEL, 'gpu': os.getenv('CUDA_VISIBLE_DEVICES'), 'pooling': 'cls',
            'normalized': True, 'max_tokens': 512, 'query_prefix': PREFIX}

@app.post('/tokens')
def tokens(body: Request):
    with lock:
        return tokenize_request(body)

def tokenize_request(body: Request):
    out = []
    for text in body.texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        snippet = tokenizer.decode(ids[:body.limit]) if body.limit is not None else text
        full = (PREFIX if body.query else '') + snippet
        out.append({'text': snippet, 'tokens': len(tokenizer.encode(full)), 'original_tokens': len(tokenizer.encode((PREFIX if body.query else '') + text))})
    return out

@app.post('/encode')
def encode(body: Request):
    # Fast tokenizers change padding/truncation state during batch calls. Every
    # tokenizer access must share this lock with /tokens and model inference.
    with lock:
        return encode_request(body)

def encode_request(body: Request):
    texts = [(PREFIX if body.query else '') + t for t in body.texts]
    counts = [len(tokenizer.encode(t)) for t in texts]
    if any(n > 512 for n in counts):
        raise HTTPException(422, 'Input exceeds 512 tokens; explicit compression is required')
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = tokenizer(texts[start:start+16], padding=True, return_tensors='pt').to('cuda:0')
            value = model(**batch).last_hidden_state[:, 0]
            vectors.extend(torch.nn.functional.normalize(value, p=2, dim=1).cpu().tolist())
    return {'embeddings': vectors, 'token_counts': counts}
