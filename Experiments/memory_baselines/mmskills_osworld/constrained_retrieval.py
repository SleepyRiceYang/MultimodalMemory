"""Frozen-pool retrieval comparison; no task mappings or evaluator access."""
import json
from pathlib import Path

from dense_retrieval import parse_decision

CATALOG = Path(__file__).resolve().parents[3] / 'Experiments/memory_baselines/splits/capability_constraints_cases_v1/operation_catalog.json'

BASELINE_PROMPT = (
    'Select reusable desktop skills BEFORE task execution using only the task instruction and '
    'the candidate skill descriptions. Select zero to three directly helpful skills, ordered by relevance. '
    'Do not select unrelated skills to fill a quota. Match the specific task constraints rather than '
    'only the application or topic. Return ONLY a JSON object with "skills": '
    '[exact skill directory names] or null, and "reason": a brief explanation.')

CAPABILITY_PROMPT = '''Select zero to three reusable skills from the supplied pool.
Use only task instruction and candidate names/descriptions. Do not infer an app
from dataset identity. Parse functional requirements as effect, object, scope,
preconditions without inventing application or paragraph assumptions. Quote the
task text supporting each requirement. For EACH candidate assess provided support, explicit conflicts
and unknown scope. Distinguish requiring a function from providing it.
Unknown is not evidence of support or of incompatibility. Reject explicitly
conflicting candidates; prefer supported effects to word overlap. Allow null.
Return JSON only: {"requirements":[{"effect":"...","object":"...",
"scope":"...","preconditions":"unknown unless stated","task_quote":"exact task substring"}],
"assessments":{"exact skill name":{"effect_match":"supported or unknown or conflict",
"supported_effects":["source-grounded effects"],"conflicts":["explicit mismatches"],
"unknowns":["unestablished properties"],"source_quote":"exact descriptor substring"}},
"skills":["up to three names"] or null,
"reason":"why selected supports meet requirements and what remains unknown"}.
Do not invent procedures, coordinates or missing application context.
effect_match=supported requires the requested operation itself, not a related
operation. Tab stop positions, alignment, line indentation and character format
are distinct effects; do not infer one from another. If the exact operation is
unestablished, mark unknown and do not select based on that unsupported effect.'''


def selection_schema(candidates, mode, instruction='', operations=()):
    properties = dict(skills={'anyOf':[{'type':'null'}, {'type':'array', 'maxItems':3,
        'items':{'type':'string','enum':[c['skill'] for c in candidates]}}]}, reason={'type':'string'})
    if mode == 'cap':
        requirement = {k:{'type':'string'} for k in ('effect','object','scope','preconditions','task_quote')}
        requirement['task_quote'] = {'type':'string','enum':[instruction]}
        requirement['effect'] = {'type':'string','enum':list(operations)}
        properties['requirements'] = {'type':'array','items':{'type':'object','properties':requirement,
            'required':list(requirement),'additionalProperties':False}}
        assessment = {k:{'type':'array','items':{'type':'string'}}
                      for k in ('supported_effects','conflicts','unknowns')}
        assessment.update(effect_match={'type':'string','enum':['supported','unknown','conflict']},
                          source_quote={'type':'string'})
        item = {'type':'object','properties':assessment,'required':list(assessment),'additionalProperties':False}
        items = {}
        for c in candidates:
            fields = dict(assessment, source_quote={'type':'string','enum':[c['description']]})
            items[c['skill']] = dict(item, properties=fields)
        properties['assessments'] = {'type':'object','properties':items,
            'required':[c['skill'] for c in candidates],'additionalProperties':False}
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def retrieve_frozen(instruction, args, transport):
    candidates = args['frozen_candidates']
    mode = args['retrieval_mode']
    catalog = json.loads(Path(args.get('operation_catalog_path', CATALOG)).read_text())
    vocabulary = sorted({op for row in catalog['skills'].values() for op in row['supported_operations']} |
                        {row['operation'] for row in catalog['ontology_extensions']} | {'unknown_operation'})
    visible = candidates if mode == 'baseline' else [
        {**c, 'supported_operations':catalog['skills'][c['skill']]['supported_operations']} for c in candidates]
    prompt = BASELINE_PROMPT if mode == 'baseline' else CAPABILITY_PROMPT
    if mode == 'cap':
        prompt += ('\nFor every requirement, effect IS the single canonical operation from the vocabulary. '
                   'Use unknown_operation when none fits. Operation support is limited to the '
                   'source-grounded supported_operations catalog; do not add inferred operations. '
                   'Select only candidates supporting at least one requested operation. '
                   'The task_quote is the complete original instruction (a source reference, not proof of interpretation).')
    response = transport.complete(model=args['model'], temperature=0, top_p=.9,
        response_format={'type':'json_schema','json_schema':{'name':'skill_selection',
                         'schema':selection_schema(candidates, mode, instruction, vocabulary),'strict':True}},
        messages=[{'role':'system','content':prompt},
                  {'role':'user','content':json.dumps(dict(instruction=instruction, candidates=visible,
                       **({'operation_vocabulary':vocabulary} if mode == 'cap' else {})))}])
    raw = response.choices[0].message.content
    names, decision = parse_decision(raw, candidates)
    if mode == 'cap':
        assessments = decision['assessments']
        expected = {c['skill'] for c in candidates}
        if set(assessments) != expected:
            raise ValueError('Capability assessment must cover each candidate exactly once')
        if any(assessments[n]['conflicts'] or not assessments[n]['supported_effects'] or
               assessments[n]['effect_match'] != 'supported' for n in names):
            raise ValueError('Selected skill violates explicit compatibility/support constraints')
        if not isinstance(decision['requirements'], list):
            raise ValueError('Expected functional requirement records')
        if any(not r['task_quote'] or r['task_quote'] not in instruction for r in decision['requirements']):
            raise ValueError('Requirement quote not grounded in task')
        descriptions = {c['skill']:c['description'] for c in candidates}
        if any(not assessments[n]['source_quote'] or
               assessments[n]['source_quote'] not in descriptions[n] for n in names):
            raise ValueError('Selected support quote not grounded in descriptor')
        requested = {r['effect'] for r in decision['requirements']}
        blocked = [n for n in names if not requested.intersection(catalog['skills'][n]['supported_operations'])]
        decision['proposed_skills'] = list(names)
        decision['operation_constraint_rejections'] = blocked
        names = [n for n in names if n not in blocked]
        decision['skills'] = list(names)
    return names, dict(policy='frozen_minicpm_top10_' + mode, candidates=candidates,
        decision=decision, max_selected_skills=3, mapping_used=False,
        visible_information=('task and original descriptors' if mode == 'baseline' else
                             'task, original descriptors and source-grounded operation labels'),
        recall_changed=False)
