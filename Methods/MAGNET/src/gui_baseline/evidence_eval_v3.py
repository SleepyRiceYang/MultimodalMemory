"""Versioned multi-state evidence pilot with current-subtask procedural retrieval."""
import fcntl
import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from .cold_continual import ROOT, Episode, Model, argument_parser, parse_json, task_selection
from .cold_parallel import assignments, memory_digest, preflight, run_round
from .cold_vision import MiniCPMEncoder
from .common import read, write
from .evidence_prompts_v2 import USE_V2

SUBTASK = '''Identify the CURRENT short coherent subtask BEFORE consulting historical memory.
Use the live screenshot, full user constraints, recent attempted actions, last reflection and
previous subtask. Output JSON with precondition, goal, completion_cue, mode, reason.
mode is execute|recover|verify. precondition describes the observed current app/surface/state;
goal describes the next meaningful state transformation, not the whole task or a pixel click.
Do not restart a completed entry stage. One subtask may span multiple actions. Retain it while
it remains applicable; advance only on visible evidence of its completion. On no-progress or
contradiction, revise the precondition/goal and use recover. Use verify if the live task may be
done; never declare completion here or infer success from an attempted action alone.
Preserve operation direction and current task parameters. This is an index query, not a GUI command.'''

APPLICABLE = '''Check each candidate historical subtask against the LIVE screenshot and current
subtask. Return JSON {"decisions":[{"state_id":"exact candidate ID","applicable":true,
"reason":"..."}]} with exactly one decision per candidate. Reject mismatched application,
operation direction, already-passed entry stage, unsupported preconditions and historical
instance values presented as current facts. Unknown applicability is not positive evidence.
A recover query does not make a normal success example a recovery example. All false is valid.
Do not generate actions. Evaluate conditions only; no historical images are supplied.'''

SELECT = '''Select minimal historical state-view evidence for the CURRENT decision, not actions.
Input: Planner role, current_request, current_subtask, and candidate memory states/views.
Use current screenshot as authority. State conditions apply only to this source-grounded card.
Output JSON with current_stage, uncertainty, visual_reference_needed (boolean),
why_not_text_only, requests, view_ids. requests contains memory_id, state_id, views (view types),
evidence_goal, reason. Use exact IDs/types from the manifest. At most 3 states and 6 total views.
If current screenshot/text suffices or nothing applies: visual_reference_needed=false,
requests=[], view_ids=[]. If true, require nonempty requests and view_ids. view_ids must equal
those resolved from requests in order. No duplicate states, views or identical image aliases.
Goals: locate_control -> exactly one full_frame OR focus_crop; recognize_before -> before,
optionally full_frame; verify_after -> after, optionally full_frame; compare_transition ->
minimal before and/or after (optionally context), never merely full_frame+focus_crop.
A historical AFTER describes an expected result, not current success. Never copy old values
or coordinates. Reject stage/direction/surface mismatches; do not load images just because
work is difficult. Preserve the current subtask direction and task parameters.
Example: current menu is open and only the small target is ambiguous -> one matching focus_crop.
Example: current text and UI already specify a formula -> false, requests=[], view_ids=[].
Example: compare unapplied/applied state cues -> coherent before+after if actually available.
Example: current app is open, reference only shows launcher -> reject it.'''


def selection_schema(manifest):
    states=[s for m in manifest for s in m['states']]
    request={'type':'object','properties':{
        'memory_id':{'type':'integer','enum':[m['id'] for m in manifest]},
        'state_id':{'type':'string','enum':[s['state_id'] for s in states]},
        'views':{'type':'array','minItems':1,'items':{'type':'string','enum':
                  sorted({v['view_type'] for s in states for v in s['available_views']})}},
        'evidence_goal':{'type':'string','enum':['locate_control','recognize_before','verify_after','compare_transition']},
        'reason':{'type':'string'}},
        'required':['memory_id','state_id','views','evidence_goal','reason'],'additionalProperties':False}
    properties={k:{'type':'string'} for k in ['current_stage','uncertainty','why_not_text_only']}
    properties.update(visual_reference_needed={'type':'boolean'},
        requests={'type':'array','maxItems':3,'items':request},
        view_ids={'type':'array','maxItems':6,'items':{'type':'string','enum':
                   [v['view_id'] for s in states for v in s['available_views']]}})
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def resolve_selection(value, manifest):
    """Validate the external model decision and resolve exact state/view references."""
    states = {(m['id'], s['state_id']): s for m in manifest for s in m['states']}
    if type(value['visual_reference_needed']) is not bool:
        raise ValueError('visual_reference_needed must be boolean')
    requests = value['requests']
    if len(requests) > 3 or len({(r['memory_id'],r['state_id']) for r in requests}) != len(requests):
        raise ValueError('Expected at most three distinct state requests')
    resolved = []
    cards = []
    for request in requests:
        key = (request['memory_id'], request['state_id'])
        if key not in states:
            raise ValueError('Unknown memory/state pair')
        state = states[key]
        types = request['views']
        available = {v['view_type']:v for v in state['available_views']}
        if not types or len(types) != len(set(types)) or any(t not in available for t in types):
            raise ValueError('Invalid view types for state')
        goal = request['evidence_goal']
        chosen = set(types)
        if goal == 'locate_control':
            valid = len(types)==1 and chosen <= {'full_frame','focus_crop'}
        elif goal == 'recognize_before':
            valid = 'before' in chosen and chosen <= {'before','full_frame'}
        elif goal == 'verify_after':
            valid = 'after' in chosen and chosen <= {'after','full_frame'}
        elif goal == 'compare_transition':
            valid = bool(chosen & {'before','after'})
        else:
            valid = False
        if not valid:
            raise ValueError('View combination does not support evidence goal')
        resolved.extend(available[t] for t in types)
        cards.append({'memory_id':request['memory_id'], 'state':state,
                      'evidence_goal':goal, 'reason':request['reason']})
    ids = [v['view_id'] for v in resolved]
    if len(ids)>6 or len({v['path'] for v in resolved})!=len(resolved) or value['view_ids']!=ids:
        raise ValueError('view_ids must match unique requested images in order, at most six')
    if value['visual_reference_needed'] != bool(ids):
        raise ValueError('Visual gate and requested views disagree')
    return resolved, cards


class SubtaskEpisode(Episode):
    def retrieve_procedural(self,folder,goal,description,history,previous,before):
        prior = folder.with_name(f'step_{int(folder.name[-3:])-1:03d}') / 'subtask.json'
        context = {'task':goal, 'screen_description':description, 'recent_actions':history[-6:],
                   'last_reflection':previous, 'previous_subtask':read(prior) if prior.exists() else None}
        def validate_subtask(value):
            if set(value)!={'precondition','goal','completion_cue','mode','reason'} or value['mode'] not in ['execute','recover','verify']:
                raise ValueError('Invalid subtask query')
            if any(not isinstance(value[k],str) or not value[k].strip() for k in value):
                raise ValueError('Subtask fields must be nonempty strings')
        subtask_properties={k:{'type':'string'} for k in ['precondition','goal','completion_cue','reason']}
        subtask_properties['mode']={'type':'string','enum':['execute','recover','verify']}
        subtask_schema={'type':'object','properties':subtask_properties,'required':list(subtask_properties),'additionalProperties':False}
        subtask = parse_json(self.model.call(folder/'subtask',SUBTASK,json.dumps(context,ensure_ascii=False),
                              [('CURRENT SCREEN',before)],validator=validate_subtask,response_schema=subtask_schema))
        write(folder/'subtask.json',subtask)
        pre, target = self.bank.encoder([subtask['precondition'],subtask['goal']])
        candidates = []
        for node in self.bank.data['procedural']:
            for state in node['states']:
                a = float(np.dot(pre,state['precondition_embedding']))
                b = float(np.dot(target,state['goal_embedding']))
                # Independent positive thresholds avoid multiplying two negative similarities.
                if a >= self.args.subtask_pre_threshold and b >= self.args.subtask_goal_threshold:
                    candidates.append({'memory_id':node['id'], 'state_id':state['state_id'],
                        'pre_similarity':a, 'goal_similarity':b, 'score':a*b,
                        'state':{k:v for k,v in state.items() if not k.endswith('_embedding')}})
        candidates.sort(key=lambda c:(-c['score'],c['memory_id'],c['state_id']))
        candidates = candidates[:self.args.subtask_top_k]
        decisions = []
        if candidates:
            ids = {c['state_id'] for c in candidates}
            def validate_applicability(value):
                ds = value['decisions']
                if len(ds)!=len(ids) or {d['state_id'] for d in ds}!=ids or any(type(d['applicable']) is not bool for d in ds):
                    raise ValueError('Applicability must cover each candidate exactly once')
            applicable_schema={'type':'object','properties':{'decisions':{
                'type':'array','minItems':len(ids),'maxItems':len(ids),'items':{'type':'object','properties':{
                    'state_id':{'type':'string','enum':sorted(ids)},'applicable':{'type':'boolean'},'reason':{'type':'string'}},
                    'required':['state_id','applicable','reason'],'additionalProperties':False}}},
                    'required':['decisions'],'additionalProperties':False}
            decisions = parse_json(self.model.call(folder/'subtask_applicability',APPLICABLE,
                json.dumps({'subtask':subtask,'candidates':candidates},ensure_ascii=False),
                [('CURRENT SCREEN',before)],validator=validate_applicability,response_schema=applicable_schema))['decisions']
        accepted = {d['state_id'] for d in decisions if d['applicable']}
        selected = []; workflows = []
        counter = self.bank.data['counter']
        for node in self.bank.data['procedural']:
            matches = [c for c in candidates if c['memory_id']==node['id'] and c['state_id'] in accepted]
            if not matches:
                continue
            selected.append({'id':node['id'], 'state_ids':[c['state_id'] for c in matches],
                             'score':max(c['score'] for c in matches)})
            workflows.append({'id':node['id'],'title':node['title'],
                              'actions':[{'subtask':c['state']['goal'],'steps':c['state']['procedure']} for c in matches]})
            self.bank.data['counter']+=1
            node['access_count']+=1
            node['last_access']=self.bank.data['counter']
        return workflows, {'kind':'procedural','query':subtask,'counter_before':counter,
                           'candidates':candidates,'applicability':decisions,'selected':selected,
                           'counter_after':self.bank.data['counter']}


class MultiStateModel(Model):
    def __init__(self,args):
        super().__init__(args)
        self.memory=read(args.evidence_memory)

    def planner_states(self, node, entry):
        return [s for s in node['states'] if s['state_id'] in entry['state_ids']]

    def select_evidence(self, folder, selection_user, image, manifest):
        def validate(value):
            resolve_selection(value,manifest)
        if manifest:
            decision=parse_json(super().call(folder,SELECT,
                json.dumps(selection_user,ensure_ascii=False),[image],validator=validate,
                response_schema=selection_schema(manifest)))
        else:
            decision={'current_stage':'No applicable retrieved state','uncertainty':'Use live UI and task text',
                      'visual_reference_needed':False,'why_not_text_only':'No applicable reference',
                      'requests':[],'view_ids':[]}
        return decision

    def call(self,folder,system,user,images=(),prefix=None,json_output=True,validator=None,response_schema=None):
        folder=Path(folder)
        if folder.name!='planner':
            return super().call(folder,system,user,images,prefix,json_output,validator,response_schema)
        kind='procedural'
        trace_path=folder.parent/(kind+'_retrieval.json')
        trace=read(trace_path)
        entries=trace['selected']
        manifest=[]
        for entry in entries:
            node=next(n for n in self.memory[kind] if n['id']==entry['id'])
            states=self.planner_states(node,entry)
            manifest.append({'id':node['id'],'knowledge':node['title'],
                             'states':[{k:v for k,v in s.items() if not k.endswith('_embedding')} for s in states]})
        selection_user={'role':folder.name,'current_request':user,'current_subtask':trace['query'],'memory':manifest}
        decision=self.select_evidence(folder.parent/(folder.name+'_evidence_selection'),
                                      selection_user, images[0], manifest)
        selected,cards=resolve_selection(decision,manifest)
        supplied=[('HISTORICAL '+v['view_type'],v['path']) for v in selected] if self.args.condition=='B3' else []
        user+=USE_V2+json.dumps({'evidence_selection':decision,
                                'current_subtask':trace['query'],
                                'selected_state_cards':cards},ensure_ascii=False)
        write(folder.parent/(folder.name+'_evidence.json'),{'condition':self.args.condition,
              'decision':decision,'state_metadata':manifest,'selected_state_cards':cards,'loaded_images':[str(p) for _,p in supplied],
              'selected_images':[v['path'] for v in selected]})
        return super().call(folder,system,user,[images[0]]+supplied,prefix,json_output,validator,response_schema)


def main():
    p=argument_parser()
    p.set_defaults(model_url='http://127.0.0.1:8177/v1',rounds=0)
    p.add_argument('--evidence-memory',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--condition',choices=['B2','B3'],required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--build-workers',type=int,default=2)
    p.add_argument('--android-urls',nargs='+',default=[f'http://127.0.0.1:{15006+i}' for i in range(4)])
    p.add_argument('--android-containers',nargs='+',default=['magnet_cold_aw']+[f'magnet_cold_aw_{i}' for i in range(1,4)])
    p.add_argument('--subtask-pre-threshold',type=float,default=.35)
    p.add_argument('--subtask-goal-threshold',type=float,default=.4)
    p.add_argument('--subtask-top-k',type=int,default=5)
    args=p.parse_args()
    snapshot=read(args.evidence_memory)
    if snapshot['evidence_protocol']!='multistate_subtask_v3':
        raise ValueError('Expected version 3 indexed multi-state memory')
    if snapshot['stationary_policy']!='original_MAGNET_unchanged' or snapshot['stationary']!=read(snapshot['source_memory'])['stationary']:
        raise ValueError('Stationary must be the unchanged original MAGNET snapshot')
    if args.subtask_top_k<1 or not 0<args.subtask_pre_threshold<=1 or not 0<args.subtask_goal_threshold<=1:
        p.error('Positive thresholds <=1 and top_k >=1 are required')
    tasks,budgets=task_selection(args)
    jobs=assignments(args,tasks)
    if args.stage=='plan':
        print(json.dumps({'protocol':'multistate_subtask_v3','tasks':tasks,'budgets':budgets,
                          'workers':jobs,'output':args.output,'memory_sha256':memory_digest(snapshot)},ensure_ascii=False,indent=2))
        return
    if args.stage=='preflight':
        raise SystemExit(0 if preflight(args,jobs) else 1)
    if args.stage!='run':
        p.error('Use plan, preflight or run; reports use evidence_report_v3')
    if not preflight(args,jobs):
        raise SystemExit('Preflight failed')
    args.task_index=0;args.max_steps=max(budgets.values());args.round_seed=args.seed;args.round_number=0
    import re
    proto=(ROOT/'Benchmark/AndroidWorld/android_world/task_evals/information_retrieval/proto/tasks.textproto').read_text()
    args.ir_tasks=re.findall(r'^  name: "([^"]+)"',proto,re.M)
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(tasks=tasks,budgets=budgets,memory_sha256=memory_digest(snapshot),
                  selection_prompt=SELECT,use_prompt=USE_V2,subtask_prompt=SUBTASK,applicability_prompt=APPLICABLE,
                  protocol='multistate_subtask_v3',stationary_policy='original_MAGNET_unchanged',
                  source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ['evidence_eval_v3.py','cold_continual.py','cold_parallel.py','cold_memory.py','evidence_prompts_v2.py']})
    if (target/'config.json').exists() and read(target/'config.json')!=config:
        raise ValueError('Changed configuration; use a new run')
    write(target/'config.json',config)
    args.stop_event=threading.Event()
    with (ROOT/'Experiments/memory_baselines/artifacts/locks/androidworld_http.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        encoder=MiniCPMEncoder(args.embedding_model,args.device)
        episodes,_=run_round(args,target,tasks,budgets,snapshot,encoder,
                             model_factory=MultiStateModel,episode_factory=SubtaskEpisode)
    write(target/'summary.json',{'condition':args.condition,'episodes':episodes,
                                'successes':sum(e['success'] for e in episodes),'n':len(episodes),'memory_frozen':True})


if __name__=='__main__':
    main()
