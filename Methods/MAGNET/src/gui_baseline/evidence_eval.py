"""Parallel frozen-memory B2/B3 pilot. Original Episode and retrieval are preserved."""
import argparse,fcntl,hashlib,json,threading
from pathlib import Path
from .cold_continual import ROOT,Model,argument_parser,task_selection,parse_json
from .cold_parallel import run_round,assignments,preflight,memory_digest
from .cold_vision import MiniCPMEncoder
from .common import read,write

SELECT='''Select historical evidence for the current decision. The CURRENT screenshot is authoritative.
First determine the current operation stage and what uncertainty matters. Historical evidence
cannot prove that an action succeeded now, supply current answers, or override the current UI.
Use only listed view_ids. Choose at most TWO views total, or an empty list if text/current screen
is sufficient, references are inapplicable, or the evidence cannot resolve this uncertainty.
Return JSON {"current_stage":"...","uncertainty":"...","reason":"...","view_ids":["..."]}.
Do not select an after view as proof that the current task is done. Conditions describe only
annotation_view_ids; do not generalize a crop's historical values or coordinates.'''
USE='''\nSOURCE-GROUNDED MEMORY POLICY:
Use the CURRENT screenshot to identify the operation stage and actual target state.
The following historical state metadata and selection are fallible references, not executed actions.
Do not repeat a workflow's entry steps when another valid entry is visible. Do not infer an action
succeeded from its description. Check toggle state before clicking, and verify completion against
all task parameters. If the same action makes no visible progress, reconsider the current route.
Reference images, when attached, show OLD instances. If absent, use the same text and current UI.
'''

class EvidenceModel(Model):
    def __init__(self,args):
        super().__init__(args);self.memory=read(args.evidence_memory)
    def call(self,folder,system,user,images=(),prefix=None,json_output=True,validator=None,response_schema=None):
        folder=Path(folder)
        if folder.name not in ['planner','actor']:
            return super().call(folder,system,user,images,prefix,json_output,validator,response_schema)
        kind='procedural' if folder.name=='planner' else 'stationary'
        trace=folder.parent/(kind+'_retrieval.json')
        ids=[x['id'] for x in read(trace)['selected']] if trace.exists() else []
        nodes=[n for n in self.memory[kind] if n['id'] in ids]
        manifest=[];by_view={}
        for n in nodes:
            state=n['evidence_state'];available=[]
            for v in state['available_views']:
                if v['view_id'] not in state['annotation_view_ids']:continue
                key=f'{kind}_{n["id"]}/{v["view_id"]}'
                by_view[key]=v;available.append({**v,'view_id':key})
            manifest.append({'id':n['id'],'knowledge':n['title'] if kind=='procedural' else n['function'],
                'state':{**state,'available_views':available}})
        if nodes:
            selection_user=json.dumps({'role':folder.name,'current_request':user,'memory':manifest},ensure_ascii=False)
            def validate(value):
                if not isinstance(value['view_ids'],list) or len(value['view_ids'])>2:raise ValueError('Expected at most two views')
                if any(v not in by_view for v in value['view_ids']):raise ValueError('Unknown view ID')
            schema={'type':'object','properties':{
                'current_stage':{'type':'string'},'uncertainty':{'type':'string'},'reason':{'type':'string'},
                'view_ids':{'type':'array','items':{'type':'string','enum':list(by_view)},'maxItems':2}},
                'required':['current_stage','uncertainty','reason','view_ids'],'additionalProperties':False}
            decision=parse_json(super().call(folder.parent/(folder.name+'_evidence_selection'),SELECT,
                selection_user,[images[0]],validator=validate,response_schema=schema))
            selected=[by_view[k] for k in decision['view_ids']]
        else:decision={'view_ids':[],'reason':'No retrieved memory'};selected=[]
        supplied=[('HISTORICAL '+v['view_type'],v['path']) for v in selected] if self.args.condition=='B3' else []
        # Drop original stationary patches: B2 and B3 are controlled by this gate.
        images=[images[0]]+supplied
        user+=USE+json.dumps({'state_metadata':manifest,'evidence_selection':decision},ensure_ascii=False)
        write(folder.parent/(folder.name+'_evidence.json'),{'condition':self.args.condition,
            'decision':decision,'loaded_images':[str(p) for _,p in supplied],
            'selected_images':[v['path'] for v in selected]})
        return super().call(folder,system,user,images,prefix,json_output,validator,response_schema)

def main():
    p=argument_parser();p.set_defaults(model_url='http://127.0.0.1:8177/v1',rounds=0,
        tasks='ClockStopWatchRunning,ContactsAddContact,SimpleCalendarAddRepeatingEvent,SystemBluetoothTurnOffVerify,AudioRecorderRecordAudio,CameraTakePhoto,MarkorEditNote,SystemCopyToClipboard')
    p.add_argument('--evidence-memory',required=True);p.add_argument('--output',required=True)
    p.add_argument('--condition',choices=['B2','B3'],required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--android-urls',nargs='+',default=[f'http://127.0.0.1:{15006+i}' for i in range(4)])
    p.add_argument('--android-containers',nargs='+',default=['magnet_cold_aw']+[f'magnet_cold_aw_{i}' for i in range(1,4)])
    p.add_argument('--build-workers',type=int,default=6)
    args=p.parse_args();tasks,budgets=task_selection(args);jobs=assignments(args,tasks)
    if not preflight(args,jobs):raise SystemExit('Preflight failed')
    args.task_index=0;args.max_steps=max(budgets.values());args.round_seed=args.seed;args.round_number=0
    args.ir_tasks=['SystemCopyToClipboard'];args.stop_event=threading.Event()
    target=Path(args.output);target.mkdir(parents=True,exist_ok=True)
    snapshot=read(args.evidence_memory)
    config={k:v for k,v in vars(args).items() if k!='stop_event'}
    config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config.update(tasks=tasks,budgets=budgets,memory_sha256=memory_digest(snapshot),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selection_prompt=SELECT,use_prompt=USE)
    if (target/'config.json').exists() and read(target/'config.json')!=config:raise ValueError('Changed frozen configuration')
    write(target/'config.json',config)
    locks=ROOT/'Experiments/memory_baselines/artifacts/locks';locks.mkdir(exist_ok=True)
    with (locks/'androidworld_http.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        encoder=MiniCPMEncoder(args.embedding_model,args.device)
        episodes,_=run_round(args,target,tasks,budgets,snapshot,encoder,model_factory=EvidenceModel)
        write(target/'summary.json',{'condition':args.condition,'episodes':episodes,
            'successes':sum(e['success'] for e in episodes),'n':len(episodes),'memory_frozen':True})
if __name__=='__main__':main()
