"""RQ1 factorial evaluation using the original stationary path in all conditions."""
import fcntl
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .cold_continual import ROOT, Episode, Model, argument_parser, parse_json, task_selection
from .cold_parallel import assignments, memory_digest, preflight, run_round
from .cold_vision import MiniCPMEncoder
from .common import read, write
from .evidence_eval_v3 import SUBTASK, MultiStateModel
from .evidence_prompts_v2 import USE_V2
from .evidence_eval_v3 import resolve_selection

LABEL = """Index the supplied reusable workflow for retrieval by local subtask goal.
Return contiguous groups of original steps with one concise, parameterized goal per group.
Use the numbered indices provided. Both start and end are inclusive. Fragments may overlap. Prefer coherent
subgoals (navigation, operation, verification); one step is allowed. Add a precondition only
when the workflow explicitly documents an entry constraint, otherwise use an empty string.
Preserve app, operation direction and target. JSON: {"labels":[{"goal":"...","start":0,"end":1,"precondition":""}]}."""
GOAL = SUBTASK + "\nPhrase goal as a reusable index label. Omit instance names, dates and literal values while preserving operation direction."
CHECK = """Check only whether an explicit fragment entry condition is contradicted by the
live screenshot or user constraints. Unknown is not a contradiction. Return rejected_ids
as the list of contradicted workflow IDs. Do not reject based on guesses."""
VISUAL = """Select the smallest set of historical views that resolves the current decision's
specific visual uncertainty. Use the live screenshot as authority. An empty list means no
historical images are needed. Return current_stage, uncertainty, why_not_text_only and choice_ids.
Each choice is one exact state and evidence bundle from choices. Return only choice_id values
from choices in choice_ids, never memory_id or state_id. Prefer one bundle. Consider
the state conditions, cues, view roles and current task direction. Never copy historical values
or coordinates, or treat a historical result as proof of live success."""
TASKS = 'ClockStopWatchRunning,ContactsAddContact,SimpleCalendarAddRepeatingEvent,SystemBluetoothTurnOffVerify,AudioRecorderRecordAudio,CameraTakePhoto,MarkorEditNote,SystemCopyToClipboard'


class SubtaskFlowEpisode(Episode):
    def retrieve_procedural(self, folder, goal, description, history, previous, before):
        context = dict(task=goal, screen_description=description, recent_actions=history[-6:],
                       last_reflection=previous)
        fields = {k: {'type':'string'} for k in ['goal','precondition','completion_cue','mode','reason']}
        schema = {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
        query = parse_json(self.model.call(folder/'subtask', GOAL, json.dumps(context),
                           [('CURRENT SCREEN', before)], response_schema=schema))
        write(folder/'subtask.json',query)
        vector = self.bank.encoder([query['goal']])[0]
        ranked = []
        for node in self.bank.data['procedural']:
            score, index = max((float(np.dot(vector, label['embedding'])), -j)
                               for j,label in enumerate(node['subtasks']))
            if score >= self.args.procedural_threshold:
                ranked.append((score, node, node['subtasks'][-index]))
        ranked.sort(key=lambda row: (-row[0], row[1]['id']))
        ranked = ranked[:self.args.procedural_top_k]
        explicit = {n['id']:label['precondition'] for _,n,label in ranked if label['precondition']}
        rejected = []
        if explicit:
            schema = {'type':'object','properties':{'rejected_ids':{'type':'array',
                       'items':{'type':'integer','enum':list(explicit)}}},
                       'required':['rejected_ids'],'additionalProperties':False}
            rejected = parse_json(self.model.call(folder/'subtask_applicability', CHECK,
                json.dumps(dict(query=query, task=goal, conditions=explicit)),
                [('CURRENT SCREEN',before)],response_schema=schema))['rejected_ids']
        selected, workflows = [], []
        counter = self.bank.data['counter']
        for score,node,label in ranked:
            if node['id'] in rejected:
                continue
            selected.append(dict(id=node['id'], score=score, subtask=label['goal'],
                                 span=[label['start'],label['end']]))
            workflows.append(dict(id=node['id'], title=node['title'],
                                  actions=node['actions'][label['start']:label['end']+1]))
            self.bank.data['counter'] += 1
            node['access_count'] += 1
            node['last_access'] = self.bank.data['counter']
        return workflows, dict(kind='procedural', query=query, selected=selected,
            candidates=[dict(id=n['id'], score=s, subtask=l['goal']) for s,n,l in ranked],
            rejected_ids=rejected, counter_before=counter, counter_after=self.bank.data['counter'])


class VisualModel(MultiStateModel):
    def planner_states(self, node, entry):
        return node['states']

    def select_evidence(self, folder, selection_user, image, manifest):
        if not manifest:
            return super().select_evidence(folder, selection_user, image, manifest)
        choices = {}
        for node in manifest:
            for state in node['states']:
                types = {view['view_type']: view for view in state['available_views']}
                bundles = [
                    ('locate_control', ['focus_crop']), ('locate_control', ['full_frame']),
                    ('recognize_before', ['before']), ('verify_after', ['after']),
                    ('compare_transition', ['before','after'])]
                for goal, views in bundles:
                    if all(t in types for t in views) and len({types[t]['path'] for t in views}) == len(views):
                        choices[str(len(choices))] = (node['id'], state['state_id'], goal, views, types)
        props = {k:{'type':'string'} for k in ['current_stage','uncertainty','why_not_text_only']}
        props['choice_ids'] = {'type':'array','maxItems':3,'items':{'type':'string','enum':list(choices)}}
        schema = {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
        prompt_memory = [{
            'knowledge': node['knowledge'],
            'states': [{
                **{key: state[key] for key in ('state_name','stage','precondition','goal',
                    'when_to_use','when_not_to_use','visible_cues','verification_cue',
                    'evidence_purpose','visual_risk','procedure') if key in state},
                'source': {key: state['source'][key] for key in ('task','step') if key in state['source']},
                'available_views': [{key: view[key] for key in ('view_type','use_for','frame_side') if key in view}
                                    for view in state['available_views']]
            } for state in node['states']]
        } for node in selection_user['memory']]
        payload = {**selection_user, 'memory':prompt_memory, 'choices':[
            dict(choice_id=key, evidence_goal=g, views=v)
            for key,(n,s,g,v,_) in choices.items()]}
        def validate(answer):
            if any(key not in choices for key in answer['choice_ids']):
                raise ValueError('Evidence selection contains an unknown choice ID')
        answer = parse_json(Model.call(self, folder, VISUAL, json.dumps(payload,ensure_ascii=False),
                                      [image], validator=validate, response_schema=schema))
        requests, ids, paths, states = [], [], set(), set()
        for key in answer.pop('choice_ids'):
            node, state, goal, views, types = choices[key]
            bundle_paths = {types[t]['path'] for t in views}
            if state in states or paths.intersection(bundle_paths):
                continue
            requests.append(dict(memory_id=node,state_id=state,evidence_goal=goal,views=views,
                                 reason=answer['why_not_text_only']))
            ids.extend(types[t]['view_id'] for t in views)
            paths.update(bundle_paths)
            states.add(state)
        decision = dict(**answer, visual_reference_needed=bool(ids), requests=requests, view_ids=ids)
        resolve_selection(decision,manifest)
        return decision


def main():
    p = argument_parser()
    p.set_defaults(model_url='http://127.0.0.1:8177/v1', device='cpu')
    p.add_argument('--evidence-memory', type=Path, required=True)
    p.add_argument('--reuse-indexed-memory', action='store_true', help='Reuse procedural subtasks and embeddings already stored in evidence-memory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--build-workers', type=int, default=4)
    p.add_argument('--android-urls', nargs='+', default=[f'http://127.0.0.1:{15006+i}' for i in range(4)])
    p.add_argument('--android-containers', nargs='+', default=['magnet_cold_aw']+[f'magnet_cold_aw_{i}' for i in range(1,4)])
    p.add_argument('--all-easy', action='store_true', help='Use all 61 AndroidWorld Easy tasks')
    p.add_argument('--rollouts', type=int, default=1, help='Independent repetitions per condition and task')
    args = p.parse_args()
    if args.rollouts < 1:
        p.error('--rollouts must be positive')
    if args.all_easy and args.tasks is not None:
        p.error('--all-easy cannot be combined with --tasks')
    if not args.all_easy and args.tasks is None:
        args.tasks = TASKS
    tasks,budgets = task_selection(args)
    if not preflight(args, assignments(args,tasks)):
        raise SystemExit('Preflight failed')
    source = read(args.evidence_memory)
    original = read(source['source_memory'])
    assert source['stationary'] == original['stationary']
    assert [(n['id'],n['title'],n['actions']) for n in source['procedural']] == [(n['id'],n['title'],n['actions']) for n in original['procedural']]
    args.task_index=0
    args.max_steps=max(budgets.values())
    args.round_seed=args.seed
    args.round_number=3
    proto=(ROOT/'Benchmark/AndroidWorld/android_world/task_evals/information_retrieval/proto/tasks.textproto').read_text()
    args.ir_tasks=re.findall(r'^  name: "([^"]+)"',proto,re.M)
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(tasks=tasks,budgets=budgets,source_memory_digest=memory_digest(source),
                  prompts=dict(index=LABEL,subtask=GOAL,entry_check=CHECK,selection=VISUAL,use=USE_V2),
                  source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ['evidence_factorial.py','evidence_eval_v3.py','cold_continual.py','cold_parallel.py','cold_memory.py']})
    args.stop_event=threading.Event()
    with (ROOT/'Experiments/memory_baselines/artifacts/locks/androidworld_http.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'config.json').exists() and read(out/'config.json')!=config:
            raise ValueError('Changed configuration; use a new output directory')
        write(out/'config.json',config)
        encoder=MiniCPMEncoder(args.embedding_model,args.device)
        def index(node):
            model=Model(SimpleNamespace(**{**vars(args),'temperature':0.}))
            def validate(value):
                for label in value['labels']:
                    if not 0 <= label['start'] <= label['end'] < len(node['actions']) or not label['goal'].strip():
                        raise ValueError('Invalid workflow fragment')
            props={k:{'type':'string'} for k in ['goal','precondition']}
            props.update({k:{'type':'integer','minimum':0,'maximum':len(node['actions'])-1} for k in ['start','end']})
            schema={'type':'object','properties':{'labels':{'type':'array','minItems':1,'maxItems':len(node['actions']),
                    'items':{'type':'object','properties':props,'required':list(props),'additionalProperties':False}}},
                    'required':['labels'],'additionalProperties':False}
            return parse_json(model.call(out/'index'/str(node['id']),LABEL,
                json.dumps(dict(title=node['title'],steps=dict(enumerate(node['actions']))),ensure_ascii=False),
                validator=validate,response_schema=schema))['labels']
        if not args.reuse_indexed_memory:
            with ThreadPoolExecutor(max_workers=args.build_workers) as pool:
                labels=list(pool.map(index,source['procedural']))
            flat=[label for series in labels for label in series]
            for label, vector in zip(flat,encoder([label['goal'] for label in flat])):
                label['embedding']=vector
            for node, series in zip(source['procedural'],labels):
                node['subtasks']=series
        write(out/'memory.json',source)
        args.evidence_memory=out/'memory.json'
        for condition, episode, model in [('T_Visual',Episode,VisualModel),
                                          ('S_Text',SubtaskFlowEpisode,Model),
                                          ('S_Visual',SubtaskFlowEpisode,VisualModel)]:
            args.condition='B3' if model is VisualModel else 'Text'
            target=out/condition
            write(target/'config.json',{**config,'condition':condition,'memory_sha256':memory_digest(source),
                                       'stationary_policy':'original_MAGNET_unchanged','use_prompt':USE_V2})
            runs=[]
            for rollout in range(1,args.rollouts+1):
                folder=target/f'rollout_{rollout:02d}' if args.rollouts>1 else target
                print(f'CONDITION {condition} rollout={rollout}/{args.rollouts}',flush=True)
                episodes,_=run_round(args,folder,tasks,budgets,source,encoder,model_factory=model,episode_factory=episode)
                result=dict(condition=condition,rollout=rollout,episodes=episodes,
                            successes=sum(e['success'] for e in episodes),n=len(episodes))
                write(folder/'summary.json',result)
                runs.append({k:v for k,v in result.items() if k!='episodes'})
            if args.rollouts>1:
                successes=sum(row['successes'] for row in runs)
                n=sum(row['n'] for row in runs)
                write(target/'summary.json',dict(condition=condition,rollouts=runs,
                      successes=successes,n=n,success_rate=successes/n))



if __name__=='__main__':
    main()
