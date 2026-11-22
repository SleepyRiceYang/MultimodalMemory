"""Paper-aligned procedural title recall, retention/time ranking and append-only workflow variants."""
from __future__ import annotations
import copy,json,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from .common import read,write,encode


def empty_bank(benchmark, construction_ids):
    return {'schema':1,'benchmark':benchmark,'construction_task_ids':sorted(construction_ids),
            'nodes':[],'counter':0,'next_id':0,'processed_episodes':[]}


def retention(counter, node):
    count=node['access_count']
    return math.exp(-max(0,counter-node['last_access'])/count) if count else 0.


def workflow_identity(node):
    return json.dumps({k:node[k] for k in ('title','app','precondition','goal','plan','postcondition')},sort_keys=True)


class EvolutionBank:
    def __init__(self, data, embedding_url, options):
        self.data=copy.deepcopy(data);self.url=embedding_url;self.options=options;self._query_vectors={}
        allowed=set(self.data['construction_task_ids'])
        if any(not set(n['source_task_ids'])<=allowed for n in self.data['nodes']):
            raise ValueError('Memory contains a non-construction source')

    def save(self, path):write(path,self.data)

    def add(self, candidate, episode_ids, created_round):
        if not set(candidate['source_task_ids'])<=set(self.data['construction_task_ids']):
            raise ValueError('Attempt to learn from a held-out task')
        key=workflow_identity(candidate)
        for old in self.data['nodes']:
            if workflow_identity(old)==key:
                old['supports']=sorted(set(old['supports'])|set(episode_ids))
                old['source_task_ids']=sorted(set(old['source_task_ids'])|set(candidate['source_task_ids']))
                return old['id'],False
        node=copy.deepcopy(candidate);number=self.data['next_id'];self.data['next_id']+=1
        node.update(id=f'{self.data["benchmark"]}/workflow/{number}',created_order=number,
                    created_at=datetime.now(timezone.utc).isoformat(),created_round=created_round,
                    access_count=0,last_access=0,supports=sorted(set(episode_ids)))
        node['embedding']=encode(self.url,[node['title']])[0]
        self.data['nodes'].append(node)
        return node['id'],True

    def select(self, goal, *, visual, update):
        before=self.data['counter'];nodes=self.data['nodes']
        if not nodes:
            return {'workflows':[],'images':[],'image_labels':[],'selected':[],
                    'counter_before':before,'counter_after':before,'candidates':[]}
        if goal not in self._query_vectors:
            self._query_vectors[goal]=np.asarray(encode(self.url,[goal],query=True)[0],dtype=np.float32)
        query=self._query_vectors[goal]
        recalled=sorted([(float(np.asarray(n['embedding'],dtype=np.float32)@query),n) for n in nodes],
                        key=lambda p:(-p[0],p[1]['id']))
        recalled=[p for p in recalled if p[0]>=self.options['recall_threshold']][:self.options['top_n']]
        candidate_rows=[{'id':n['id'],'cosine':s,'retention':retention(before,n)} for s,n in recalled]
        # MAGNET paper Algorithm 2 uses retention first, creation time second.
        ranked=sorted(recalled,key=lambda p:(-retention(before,p[1]),-p[1]['created_order'],p[1]['id']))
        selected=[];workflows=[];images=[];labels=[];used=0
        for cosine,node in ranked:
            if len(selected)>=self.options['top_k']:break
            body={k:node[k] for k in ('id','title','app','precondition','goal','plan','postcondition')}
            text=json.dumps(body,ensure_ascii=False)
            # Reserve identical image capacity in both arms, then omit images for Text.
            if used+len(text)>self.options['max_text_chars'] or len(images)+len(node['evidence'])>self.options['max_images']:continue
            used+=len(text);workflows.append(body)
            selected.append({'id':node['id'],'cosine':cosine,'retention':retention(before,node),
                             'created_order':node['created_order']})
            for phase,path in zip(('BEFORE','AFTER'),node['evidence']):
                if not Path(path).is_file():raise FileNotFoundError(path)
                images.append(path);labels.append(f'EXPERIENCE {node["id"]} {phase}; NOT current GUI')
        if update and selected:
            self.data['counter']+=1
            ids={n['id'] for n in selected}
            for node in nodes:
                if node['id'] in ids:node['access_count']+=1;node['last_access']=self.data['counter']
        return {'workflows':workflows,'selected':selected,'images':images if visual else [],
                'image_labels':labels if visual else [],'counter_before':before,'counter_after':self.data['counter'],
                'candidates':candidate_rows}
