"""Frozen, benchmark-isolated procedural memory; text/visual are paired views."""
from __future__ import annotations
import copy,json,re
from pathlib import Path
import numpy as np
from .common import read,write,encode
from .prompts import MEMORY_NOTICE


def payload_text(node):
    fields={k:node[k] for k in ('id','title','app','precondition','goal','parameters','steps','postcondition')}
    # Evidence/source actions and literal parameters stay in provenance, never the actor payload.
    fields['parameters']=[p['name'] for p in node['parameters']]
    fields['steps']=[{'instruction':s['instruction'],'expected_state':s['expected_state']} for s in node['steps']]
    return json.dumps(fields,ensure_ascii=False)


def validate_node(candidate, source):
    steps=candidate['steps'];by_id={s['step']:s for s in source['steps']}
    if not 2<=len(steps)<=8:raise ValueError('A procedural unit requires 2-8 recorded actions')
    ids=[s['source_step'] for s in steps]
    if len(set(ids))!=len(ids) or ids!=sorted(ids) or any(i not in by_id for i in ids):
        raise ValueError('Invalid/duplicate source action citations')
    positions=[next(i for i,s in enumerate(source['steps']) if s['step']==key) for key in ids]
    if positions!=list(range(positions[0],positions[0]+len(positions))) or ids!=list(range(ids[0],ids[0]+len(ids))):raise ValueError('Workflow skips source actions')
    # Reject repeated no-op sequences rather than manufacturing procedural knowledge.
    actions=[json.dumps(by_id[i]['action'],sort_keys=True) for i in ids]
    if len(set(actions))<2:raise ValueError('Identical repeated actions are not a procedure')
    for key in ['title','app','precondition','goal','postcondition']:
        if not isinstance(candidate[key],str) or not candidate[key].strip():raise ValueError('Empty procedural field: '+key)
    evidence=[]
    for key,phase in [(ids[0],'before'),(ids[-1],'after')]:
        path=Path(by_id[key][phase])
        if not path.is_file():raise ValueError('Missing source observation')
        evidence.append(str(path))
    return ids,evidence


def parameterize(node, source):
    node=copy.deepcopy(node)
    corpus=source['goal']+' '+json.dumps([s['action'] for s in source['steps']],ensure_ascii=False)
    accepted=[]
    for p in node.get('parameters',[]):
        name,value=p.get('name',''),p.get('value','')
        if not re.fullmatch('[a-z][a-z0-9_]{1,40}',name) or not isinstance(value,str):continue
        if len(value)<3 or value not in corpus or value.lower() in {'on','off','max','min','true','false','settings','clock','markor'}:continue
        accepted.append({'name':name,'value':value})
        for key in ('title','precondition','goal','postcondition'):
            node[key]=node[key].replace(value,'<'+name+'>')
        for step in node['steps']:
            for key in ('instruction','expected_state'):step[key]=step[key].replace(value,'<'+name+'>')
    node['parameters']=accepted
    return node


class ProceduralBank:
    def __init__(self, root, embedding_url='http://127.0.0.1:8087'):
        self.root=Path(root);self.data=read(self.root/'bank.json');self.url=embedding_url
        if self.data.get('stationary_memory') is not False:raise ValueError('This baseline prohibits stationary memory')
        self.nodes=self.data['nodes'];self.vectors=np.load(self.root/'vectors.npy',allow_pickle=False)
        if len(self.vectors)!=len(self.nodes):raise ValueError('Invalid index size')
        self.frozen=(self.root/'bank.json').read_bytes();self.frozen_vectors=self.vectors.copy()
        self.assets={str(p):p.read_bytes() for p in [self.root/'index.json']+[self.root/e for n in self.nodes for e in n['evidence']]}

    def verify(self):
        if (self.root/'bank.json').read_bytes()!=self.frozen or not np.array_equal(np.load(self.root/'vectors.npy',allow_pickle=False),self.frozen_vectors):
            raise ValueError('Frozen procedural memory changed')
        if any(Path(p).read_bytes()!=value for p,value in self.assets.items()):raise ValueError("Frozen evidence/index changed")
        return True

    def select(self, benchmark, task_id, goal, top_k=3, max_images=2, max_chars=16000):
        if benchmark!=self.data['benchmark']:raise ValueError('Cross-benchmark memory is prohibited')
        if task_id in self.data['source_task_ids']:raise ValueError('Validation task occurs in construction sources')
        if not self.nodes:return {'text':'','images':[],'selected':[],'candidates':[],'ranking':'embedding_cosine','stationary_memory':False}
        vector=np.asarray(encode(self.url,[goal],query=True)[0],dtype=np.float32)
        ranked=sorted([(float(self.vectors[i]@vector),n) for i,n in enumerate(self.nodes)],key=lambda x:(-x[0],x[1]['id']))[:20]
        selected=[];images=[];texts=[];candidates=[]
        for score,node in ranked:
            candidates.append({'id':node['id'],'cosine':score})
            if len(selected)>=top_k:continue
            evidence=node['evidence']
            text=payload_text(node)
            if len(images)+len(evidence)>max_images or len(MEMORY_NOTICE)+sum(map(len,texts))+len(text)>max_chars:continue
            selected.append({'id':node['id'],'score':score,'source_task_id':node['source_task_id']})
            texts.append(text);images.extend(str(self.root/p) for p in evidence)
        # Both arms reserve the SAME visual budget and therefore receive identical text/IDs.
        return {'text':MEMORY_NOTICE+'\n'.join(texts) if texts else '', 'images':images,
                'selected':selected,'candidates':candidates,'ranking':'embedding_cosine','stationary_memory':False}

    @staticmethod
    def view(context, visual):
        result=copy.deepcopy(context)
        if not visual:result['images']=[]
        return result
