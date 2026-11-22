"""Download pinned MiniCPM files through configured proxies, with resumable ranges and checksums."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

import requests

REPO = 'openbmb/MiniCPM-Embedding'
REVISION = 'dc0f82b4466b254dddc25787bf7b1cbc28f755b0'
DEST = Path('/data/pretrained_models/MiniCPM-Embedding')
CHUNK = 32 * 1024 * 1024


def get(url, **kwargs):
    last = None
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(20, 120), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last = exc
            time.sleep(min(2 ** attempt, 10))
    raise last


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if not any(os.environ.get(k) for k in ('HTTPS_PROXY', 'https_proxy', 'ALL_PROXY', 'all_proxy')):
        raise SystemExit('Set HTTPS_PROXY to the working proxy before downloading')
    info = get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true').json()
    DEST.mkdir(parents=True, exist_ok=True)
    tasks, files = [], []
    for entry in info['siblings']:
        name, size = entry['rfilename'], entry['size']
        target = DEST / name
        expected = entry.get('lfs', {}).get('sha256')
        if target.is_file() and target.stat().st_size == size and (not expected or checksum(target) == expected):
            print('REUSE', name, flush=True)
            files.append(entry)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        parts = DEST / '.download_parts' / name
        parts.mkdir(parents=True, exist_ok=True)
        count = (size + CHUNK - 1) // CHUNK
        for index in range(count):
            start, end = index * CHUNK, min(size, (index + 1) * CHUNK) - 1
            part = parts / f'{index:05d}'
            if not part.is_file() or part.stat().st_size != end - start + 1:
                tasks.append((name, start, end, part, size))
        files.append(entry)
    def transfer(task):
        name, start, end, part, size = task
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        expected_length = end - start + 1
        expected_range = f'bytes {start}-{end}/{size}'
        last_error = None
        # Some proxy cache entries ignore Range or return a short body. Validate
        # headers before buffering and retry with a fresh URL instead of aborting
        # the entire resumable download.
        for attempt in range(10):
            params = {
                'download': 'true',
                'range_id': f'{start}-{end}-{attempt}',
                'nonce': time.time_ns(),
            }
            headers = {
                'Range': f'bytes={start}-{end}',
                'Accept-Encoding': 'identity',
            }
            try:
                with requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=(20, 120),
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    content_range = response.headers.get('Content-Range')
                    if size > CHUNK and (
                        response.status_code != 206 or content_range != expected_range
                    ):
                        raise ValueError(
                            f'Invalid range response: {name} {start}-{end}: '
                            f'status={response.status_code}, Content-Range={content_range!r}'
                        )
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) != expected_length:
                        raise ValueError(
                            f'Invalid Content-Length: {name} {start}-{end}: {content_length}'
                        )
                    body = bytearray()
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if not block:
                            continue
                        if len(body) + len(block) > expected_length:
                            raise ValueError(
                                f'Range body too large: {name} {start}-{end}'
                            )
                        body.extend(block)
                    if len(body) != expected_length:
                        raise ValueError(
                            f'Invalid download size: {name} {start}-{end}: '
                            f'{len(body)} != {expected_length}'
                        )
                temporary = part.with_suffix('.tmp')
                temporary.write_bytes(body)
                temporary.replace(part)
                return len(body)
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < 4:
                    print(
                        f'RETRY {name} {start}-{end} ({attempt + 1}/5): {exc}',
                        flush=True,
                    )
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f'Failed range after 10 attempts: {name} {start}-{end}: {last_error}')
    done, started = 0, time.monotonic()
    # Keep proxy load modest; the configured tunnel intermittently times out
    # when too many large range requests are active at once.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(transfer, task) for task in tasks]):
            done += future.result()
            print(f'DOWNLOAD {done / 1e6:.1f} MB, {done / 1e6 / (time.monotonic() - started):.1f} MB/s', flush=True)
    manifest = {'repo': REPO, 'revision': REVISION, 'files': []}
    for entry in files:
        name, size = entry['rfilename'], entry['size']
        target = DEST / name
        parts = DEST / '.download_parts' / name
        if parts.exists():
            temp = target.with_suffix(target.suffix + '.incomplete')
            with temp.open('wb') as output:
                for index in range((size + CHUNK - 1) // CHUNK):
                    output.write((parts / f'{index:05d}').read_bytes())
            temp.replace(target)
        actual = checksum(target)
        expected = entry.get('lfs', {}).get('sha256')
        if target.stat().st_size != size or (expected and actual != expected):
            raise ValueError(f'Checksum verification failed: {name}')
        manifest['files'].append({'name': name, 'bytes': size, 'sha256': actual})
        if parts.exists():
            for part in parts.iterdir():
                part.unlink()
            parts.rmdir()
        print('VERIFIED', name, flush=True)
    (DEST / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('COMPLETED', DEST, flush=True)


if __name__ == '__main__':
    main()
