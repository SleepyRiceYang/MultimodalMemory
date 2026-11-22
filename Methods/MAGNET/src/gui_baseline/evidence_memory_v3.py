"""Build multiple source-grounded procedural state cards without evaluating new tasks."""
import argparse
import copy
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from .cold_continual import Model, parse_json
from .cold_vision import MiniCPMEncoder
from .common import read, write

PLAN = '''Divide this recorded successful workflow into reusable subtask segments and identify
its KEY DECISION STATES, not one card per action. Use only supplied observations and effects.
A subtask is a short coherent goal spanning one or more actions, not the entire user task.
Return JSON {"states":[{"state_name":"...","stage":"entry|operation|verification",
"start_step":0,"end_step":2,"anchor_step":0,"precondition":"...","goal":"..."}]}.
Use 1 to 6 cards as supported; do not force a fixed count. Step numbers must exist and obey
start_step <= anchor_step <= end_step. Segments can overlap for different decision needs.
The anchor supplies the before/after screenshots for visual annotation. Preserve action direction.
Parameterize literal names, dates and numbers. Do not treat attempted actions as proven success.
Choose checkpoints for navigation, meaningful operations, branching or result verification.
Do not invent alternate paths or states not present in the trajectory.'''

ANNOTATE = '''Ground ONE proposed decision state in the supplied historical BEFORE and AFTER
screenshots and the recorded segment. Neither screenshot depicts the live evaluation task.
Return ONLY JSON with keys state_name, stage, precondition, goal, when_to_use, when_not_to_use,
visible_cues (list of strings), verification_cue, evidence_purpose
(locate_control|recognize_state|verify_result), visual_risk, focus_bbox, focus_label,
include_before, include_after. stage must be entry|operation|verification.
Use unknown for unsupported conditions. Do not preserve example names, dates, values or coordinates
as reusable facts. Describe starting UI separately from the intended subtask state transformation.
For verification stage, describe the visible result and check the LIVE task's parameters.
Always retain a full_frame of BEFORE for global context. focus_bbox is either [] if a crop adds
no grounded evidence, or [left,top,right,bottom] in normalized 0..1000 coordinates on BEFORE,
tightly enclosing an identifiable control or relevant local work region with enough context.
Do not fabricate a crop around a click coordinate. include_before/include_after are booleans:
include temporal views only when these REAL frames clarify a useful transition or result.
A historical AFTER never proves current task completion. A failed/no-change local operation
must not be described as verified merely because the whole trajectory ultimately succeeded.'''

BINDINGS = '''Identify literal TASK INSTANCE VALUES in the source instruction or recorded history that must be
parameterized in reusable memory. Return JSON {"bindings":[{"literal":"exact substring of task",
"placeholder":"[ParameterName]"}]}. Include complete values and component first/last names when
the supplied cards mention them separately. Include instance dates, times, durations, numbers,
filenames, titles, message contents and similar task parameters. Preserve application names,
UI labels, operation direction, requested feature, and generic functional knowledge; those are
not instance values. An empty list is valid for parameter-free tasks. Do not invent literals.
Each literal must occur verbatim in the task or supplied historical records. Placeholders contain only ASCII letters/digits/
underscores, start with a letter, and are enclosed in square brackets.'''


def build(args):
    original = read(args.memory)
    bank = copy.deepcopy(original)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'version': 'multistate_v3', 'source': str(Path(args.memory).resolve()),
                'source_sha256': hashlib.sha256(Path(args.memory).read_bytes()).hexdigest(),
                'model': args.model, 'model_url': args.model_url, 'plan_prompt': PLAN,
                'annotation_prompt': ANNOTATE, 'embedding_model': args.embedding_model,
                'workers': args.workers, 'max_states_per_source': 6}
    path = out / 'construction_protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Construction configuration changed; use a new output directory')
    write(path, protocol)
    provenance = Path(original['source_memory']).parent.parent

    def annotate_node(node):
        model = Model(SimpleNamespace(model_url=args.model_url, model=args.model,
                                     temperature=0., max_tokens=3000, max_model_attempts=3))
        annotation_model = Model(SimpleNamespace(model_url=args.model_url, model=args.model,
                                                temperature=0., max_tokens=3000, max_model_attempts=1))
        annotation_properties={k:{'type':'string'} for k in ['state_name','precondition','goal',
            'when_to_use','when_not_to_use','verification_cue','visual_risk','focus_label']}
        annotation_properties.update(stage={'type':'string','enum':['entry','operation','verification']},
            evidence_purpose={'type':'string','enum':['locate_control','recognize_state','verify_result']},
            visible_cues={'type':'array','items':{'type':'string'}},
            focus_bbox={'type':'array','maxItems':4,'items':{'type':'number','minimum':0,'maximum':1000}},
            include_before={'type':'boolean'},include_after={'type':'boolean'})
        annotation_schema={'type':'object','properties':annotation_properties,
                           'required':list(annotation_properties),'additionalProperties':False}
        def validate_annotation(value):
            if set(value)!=set(annotation_properties):
                raise ValueError('Invalid state annotation fields')
            b=value['focus_bbox']
            if b and (len(b)!=4 or any(type(v) not in (int,float) or not 0<=v<=1000 for v in b)
                      or b[0]>=b[2] or b[1]>=b[3]):
                raise ValueError('Invalid normalized crop bbox')
        states = []
        for task in node['source']:
            # One actual successful trajectory per source task, in earliest-round order.
            episodes = [read(p)['episode'] for r in range(3)
                        if (p := provenance / f'round_{r:02d}/tasks' / task / 'checkpoint.json').exists()]
            episode = next(e for e in episodes if e['success'] and e['transitions'])
            transitions = episode['transitions']
            by_step = {t['step']: t for t in transitions}
            audit = out / 'annotations' / str(node['id']) / task
            summary = [{k: t[k] for k in ['step', 'description', 'screen_description', 'effect']}
                       for t in transitions]
            properties={k:{'type':'string'} for k in ['state_name','precondition','goal']}
            properties['stage']={'type':'string','enum':['entry','operation','verification']}
            properties.update({k:{'type':'integer','enum':list(by_step)}
                               for k in ['start_step','end_step','anchor_step']})
            plan_schema={'type':'object','properties':{'states':{'type':'array','minItems':1,'maxItems':6,
                'items':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}},
                'required':['states'],'additionalProperties':False}
            def validate_plan(value):
                if not 1<=len(value['states'])<=6:
                    raise ValueError('Expected 1..6 decision states')
                for item in value['states']:
                    a,b,c=(item[k] for k in ['start_step','end_step','anchor_step'])
                    if not all(s in by_step for s in [a,b,c]) or not a<=c<=b:
                        raise ValueError('Invalid state trajectory segment')
            plan = parse_json(model.call(audit / 'plan', PLAN,
                json.dumps({'workflow': {'title': node['title'], 'actions': node['actions']},
                            'trajectory': summary}, ensure_ascii=False),
                validator=validate_plan,response_schema=plan_schema))
            if not 1 <= len(plan['states']) <= 6:
                raise ValueError('Expected 1..6 decision states')
            for number, proposal in enumerate(plan['states']):
                start, end, anchor = (proposal[k] for k in ['start_step', 'end_step', 'anchor_step'])
                if not all(s in by_step for s in [start, end, anchor]) or not start <= anchor <= end:
                    raise ValueError('State segment references invalid trajectory steps')
                t = by_step[anchor]
                segment = [x for x in transitions if start <= x['step'] <= end]
                annotation_system=ANNOTATE
                review_path=audit/f'state_{number:02d}'/'review.json'
                for attempt in range(3):
                    annotation_folder=audit/f'state_{number:02d}'
                    if review_path.exists():
                        metadata=read(review_path)['metadata']
                        validate_annotation(metadata)
                        break
                    if attempt:
                        annotation_folder=annotation_folder/f'repair_{attempt}'
                    try:
                        metadata = parse_json(annotation_model.call(annotation_folder, annotation_system,
                            json.dumps({'proposal': proposal, 'segment':[
                                {k:x[k] for k in ['step','description','screen_description','effect']} for x in segment]}, ensure_ascii=False),
                            [('HISTORICAL BEFORE', t['before']), ('HISTORICAL AFTER', t['after'])],
                            validator=validate_annotation,response_schema=annotation_schema))
                        break
                    except ValueError as exc:
                        if attempt==2:
                            raise
                        annotation_system=ANNOTATE+'\nThe preceding candidate was rejected: '+str(exc)+'''
Re-read the screenshots. A bbox is an AREA: right > left and bottom > top, never a point
duplicated twice. If no credible area can be located, return focus_bbox=[]. Do not invent
a larger box merely to pass validation. Respect every required schema field.'''
                required = {'state_name', 'stage', 'precondition', 'goal', 'when_to_use',
                            'when_not_to_use', 'visible_cues', 'verification_cue',
                            'evidence_purpose', 'visual_risk', 'focus_bbox', 'focus_label',
                            'include_before', 'include_after'}
                if set(metadata) != required or metadata['stage'] not in ['entry','operation','verification']:
                    raise ValueError('Invalid state annotation fields/stage')
                if metadata['evidence_purpose'] not in ['locate_control','recognize_state','verify_result']:
                    raise ValueError('Invalid evidence purpose')
                state_id = f'procedural_{node["id"]}/{task}/state_{number:02d}'
                source = {'task': task, 'step': anchor, 'before': t['before'], 'after': t['after']}
                views = [{'view_id': state_id + '/full_frame', 'view_type': 'full_frame',
                          'path': t['before'], 'use_for': 'global_context', 'source': source,
                          'frame_side': 'before'}]
                bbox = metadata.pop('focus_bbox')
                label = metadata.pop('focus_label')
                if bbox:
                    if (len(bbox) != 4 or any(type(v) not in (int,float) or not 0 <= v <= 1000 for v in bbox)
                            or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
                        raise ValueError('Invalid normalized crop bbox')
                    image = Image.open(t['before'])
                    pixels = [round(v * (image.width if i % 2 == 0 else image.height) / 1000)
                              for i, v in enumerate(bbox)]
                    if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
                        raise ValueError('Empty grounded crop')
                    crop = audit / f'state_{number:02d}' / 'focus_crop.png'
                    image.crop(pixels).save(crop)
                    views.append({'view_id': state_id + '/focus_crop', 'view_type': 'focus_crop',
                                  'path': str(crop), 'use_for': label, 'bbox_1000': bbox, 'source': source})
                for side in ['before','after']:
                    include = metadata.pop('include_' + side)
                    if type(include) is not bool:
                        raise ValueError('Temporal view flag must be boolean')
                    if include:
                        views.append({'view_id': state_id + '/' + side, 'view_type': side,
                                      'path': t[side], 'use_for': 'recognize_state' if side=='before' else 'expected_result',
                                      'source': source})
                states.append({**metadata, 'state_id': state_id, 'source': source,
                               'annotation_request':str(annotation_folder/'request.json'),
                               'annotation_review':str(review_path) if review_path.exists() else None,
                               'segment_steps': [start,end],
                               'procedure': [metadata['goal']], 'available_views': views})
        node.pop('evidence_state')
        node['states'] = states
        print('BUILT', node['id'], len(states), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(annotate_node, bank['procedural']))
    # Explicit instance binding audit: annotation instructions alone did not reliably remove values.
    by_task={}
    for node in bank['procedural']:
        for state in node['states']:
            by_task.setdefault(state['source']['task'],[]).append(state)
    binding_schema={'type':'object','properties':{'bindings':{'type':'array','items':{
        'type':'object','properties':{'literal':{'type':'string'},'placeholder':{'type':'string'}},
        'required':['literal','placeholder'],'additionalProperties':False}}},'required':['bindings'],'additionalProperties':False}
    def parameterize(item):
        task,states=item
        goal=read(Path(states[0]['source']['before']).parent.parent/'task.json')['goal']
        history=read(Path(states[0]['source']['before']).parent.parent/'checkpoint.json')['episode']['transitions']
        recorded=json.dumps(history,ensure_ascii=False)
        model=Model(SimpleNamespace(model_url=args.model_url,model=args.model,temperature=0.,
                                   max_tokens=1600,max_model_attempts=3))
        def validate_bindings(value):
            for b in value['bindings']:
                if not b['literal'].strip() or (b['literal'] not in goal and b['literal'] not in recorded) or not re.fullmatch(r'\[[A-Za-z][A-Za-z0-9_]*\]',b['placeholder']):
                    raise ValueError('Instance binding is not grounded in source task')
        review_path=out/'bindings'/task/'review.json'
        if review_path.exists():
            value=read(review_path)
            validate_bindings(value)
            bindings=value['bindings']
        else:
            bindings=parse_json(model.call(out/'bindings'/task,BINDINGS,
            json.dumps({'task':goal,'recorded_history':history,'cards':[{k:s[k] for k in ['state_name','precondition','goal','visible_cues','verification_cue','procedure']}
                                           for s in states]},ensure_ascii=False),
            validator=validate_bindings,response_schema=binding_schema))['bindings']
        bindings=sorted(bindings,key=lambda b:-len(b['literal']))
        for state in states:
            for field in ['state_name','precondition','goal','when_to_use','when_not_to_use',
                          'visible_cues','verification_cue','visual_risk','procedure']:
                values=state[field] if isinstance(state[field],list) else [state[field]]
                for b in bindings:
                    values=[re.sub(r'(?<!\w)'+re.escape(b['literal'])+r'(?!\w)',b['placeholder'],v) for v in values]
                state[field]=values if isinstance(state[field],list) else values[0]
            for view in state['available_views']:
                for b in bindings:
                    view['use_for']=re.sub(r'(?<!\w)'+re.escape(b['literal'])+r'(?!\w)',b['placeholder'],view['use_for'])
        print('PARAMETERIZED',task,len(bindings),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(parameterize,by_task.items()))
    # Stationary follows the original MAGNET bank and execution path, without state-card extensions.
    bank['stationary'] = copy.deepcopy(read(original['source_memory'])['stationary'])
    bank['stationary_policy'] = 'original_MAGNET_unchanged'
    write(out/'stationary_protocol.json',{'policy':bank['stationary_policy'],
          'source':original['source_memory'],
          'source_sha256':hashlib.sha256(Path(original['source_memory']).read_bytes()).hexdigest(),
          'functions':len(bank['stationary']),'images':sum(len(n['images']) for n in bank['stationary']),
          'evidence_gate_scope':'procedural_planner_only'})
    write(out / 'memory_unindexed.json', bank)
    encoder = MiniCPMEncoder(args.embedding_model, args.device)
    for node in bank['procedural']:
        for state in node['states']:
            state['precondition_embedding'], state['goal_embedding'] = encoder(
                [state['precondition'], state['goal']])
    bank['evidence_protocol'] = 'multistate_subtask_v3'
    write(out/'parameterization_protocol.json',{'prompt':BINDINGS,'schema':binding_schema,
          'scope':'Procedural reusable text only; raw audit, source IDs and historical pixels retain provenance'})
    write(out / 'memory.json', bank)
    write(out / 'summary.json', {
        'procedural_nodes': len(bank['procedural']),
        'procedural_states': sum(len(n['states']) for n in bank['procedural']),
        'procedural_views': sum(len(s['available_views']) for n in bank['procedural'] for s in n['states']),
        'states_per_node': {str(n['id']):len(n['states']) for n in bank['procedural']},
        'view_types': {k:sum(v['view_type']==k for n in bank['procedural'] for s in n['states']
                             for v in s['available_views']) for k in ['full_frame','focus_crop','before','after']}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model-url', default='http://127.0.0.1:8177/v1')
    p.add_argument('--model', default='qwen3-vl-8b-instruct')
    p.add_argument('--embedding-model', default='/data/pretrained_models/MiniCPM-Embedding')
    p.add_argument('--device', default='cpu')
    p.add_argument('--workers', type=int, default=2)
    build(p.parse_args())


if __name__ == '__main__':
    main()
