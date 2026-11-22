"""Generate source-grounded capability sidecars using local vLLM, without tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SCHEMA = '''Return JSON only, with keys supports, requires, preconditions, unknown_scope,
and state_evidence. supports and requires are lists of objects with capability,
effect, scope, source_id. Use these functional categories:
object_localization, stage_recognition, structural_editing, semantic_conversion,
settings_persistence, cross_app_transfer, result_verification, recovery.
Use [] when unsupported. requires means a prerequisite, NOT support provided.
preconditions and unknown_scope are lists of strings.
state_evidence is a list of objects with state_id, capability, question,
source_id. Select an existing evidence ID; no invented steps,
coordinates or application behavior. Infer only from the supplied skill text
and existing state-card text. Do not infer proficiency or task assignments.
Use at most THREE primary supports. Do not fill every category.
structural_editing means changing document/table object structure, NOT editing settings.
semantic_conversion means numerical/unit/format meaning conversion, NOT verifying text.
object_localization support requires actual control-location clues, not merely needing clicks.
Every source_id must be copied from the supplied evidence table.
State evidence must cite a field from that same state_id.'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('skill_dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8170/v1')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sources = {name: (a.skill_dir / name).read_text() for name in
               ('SKILL.md', 'runtime_state_cards.json')}
    evidence = {}
    for index, line in enumerate(sources['SKILL.md'].splitlines(), 1):
        if line.strip():
            evidence[f'md:{index}'] = dict(source_file='SKILL.md', quote=line)
    cards = json.loads(sources['runtime_state_cards.json'])['states']
    for state in cards:
        for field in ('when_to_use', 'when_not_to_use', 'visible_cues', 'verification_cue'):
            value = state[field]
            values = value if isinstance(value, list) else [value]
            for index, snippet in enumerate(values):
                evidence[f"state:{state['state_id']}:{field}:{index}"] = dict(
                    source_file='runtime_state_cards.json', state_id=state['state_id'], quote=snippet)
    payload = dict(model='qwen3-vl-8b-instruct', temperature=0, max_tokens=8192,
                   response_format={'type': 'json_object'}, messages=[
                       {'role': 'system', 'content': SCHEMA},
                       {'role': 'user', 'content': json.dumps(evidence)}])
    request = urllib.request.Request(a.base_url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.with_suffix('.annotation_request_response.json').write_text(
        json.dumps({'request': payload, 'response': result}, ensure_ascii=False, indent=2) + '\n')
    labels = json.loads(result['choices'][0]['message']['content'])
    for key in ('supports', 'requires', 'preconditions', 'unknown_scope', 'state_evidence'):
        if not isinstance(labels[key], list):
            raise ValueError(f'Expected list: {key}')
    states = {s['state_id'] for s in json.loads(sources['runtime_state_cards.json'])['states']}
    for item in labels['supports'] + labels['requires'] + labels['state_evidence']:
        source = evidence[item['source_id']]
        item['source_file'], item['quote'] = source['source_file'], source['quote']
    for item in labels['state_evidence']:
        if item['state_id'] not in states:
            raise ValueError(f'Unknown state: {item}')
        if evidence[item['source_id']]['state_id'] != item['state_id']:
            raise ValueError(f'State evidence mismatch: {item}')
    record = dict(skill_id=a.skill_dir.name, labels=labels,
                  source_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
                  annotation_prompt=SCHEMA, request=payload, response=result,
                  provenance='local_vllm_skill_text_only_no_task_input')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
