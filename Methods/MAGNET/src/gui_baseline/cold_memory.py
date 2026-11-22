"""Dual memory for AndroidWorld cold-start continual adaptation (MAGNET §2.3)."""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path

import numpy as np


def empty_memory():
    return {'schema':1,'initialization':'empty','counter':0,'next_id':0,
            'procedural':[],'stationary':[],'completed_round':-1}


def retention(counter, item):
    return math.exp(-max(0,counter-item['last_access'])/item['access_count']) if item['access_count'] else 0.


class DualMemory:
    def __init__(self,data,encoder,options):
        self.data=copy.deepcopy(data);self.encoder=encoder;self.options=options

    def metadata(self,round_number,source):
        number=self.data['next_id'];self.data['next_id']+=1
        return {'id':number,'created_order':number,'created_round':round_number,
                'access_count':0,'last_access':0,'source':source}

    def select(self,kind,query):
        nodes=self.data[kind]
        audit={'kind':kind,'query':query,'counter_before':self.data['counter'],'candidates':[],'selected':[]}
        if not nodes:return [],audit
        vector=self.encoder([query])[0]
        candidates=[(float(np.dot(n['embedding'],vector)),n) for n in nodes]
        candidates=sorted(candidates,key=lambda p:(-p[0],p[1]['id']))
        candidates=[p for p in candidates if p[0]>=self.options[kind+'_threshold']][:self.options[kind+'_top_n']]
        counter=self.data['counter']
        audit['candidates']=[{'id':n['id'],'cosine':s,'retention':retention(counter,n)} for s,n in candidates]
        # Paper Algorithm 2: retention, then creation time; not vector-only ranking.
        candidates.sort(key=lambda p:(-retention(counter,p[1]),-p[1]['created_order'],p[1]['id']))
        selected=[]
        for cosine,node in candidates[:self.options[kind+'_top_k']]:
            item={k:v for k,v in node.items() if k!='embedding'}
            if kind=='stationary':
                image=max(node['images'],key=lambda x:(retention(counter,x),x['created_order']))
                item['image']=image['path'];item['image_id']=image['id']
                self.data['counter']+=1;image['access_count']+=1;image['last_access']=self.data['counter']
            self.data['counter']+=1;node['access_count']+=1;node['last_access']=self.data['counter']
            selected.append(item);audit['selected'].append({'id':node['id'],'cosine':cosine,
                **({'image_id':item['image_id']} if kind=='stationary' else {})})
        audit['counter_after']=self.data['counter']
        return selected,audit

    def replay_accesses(self,audit):
        """Commit selected IDs in canonical order, without re-ranking past requests."""
        nodes={n['id']:n for n in self.data[audit['kind']]}
        for selected in audit['selected']:
            node=nodes[selected['id']]
            if audit['kind']=='stationary':
                image=next(x for x in node['images'] if x['id']==selected['image_id'])
                self.data['counter']+=1;image['access_count']+=1;image['last_access']=self.data['counter']
            self.data['counter']+=1;node['access_count']+=1;node['last_access']=self.data['counter']

    def add_workflow(self,title,actions,round_number,source):
        node={**self.metadata(round_number,source),'title':title,'actions':actions,
              'embedding':self.encoder([title])[0]}
        self.data['procedural'].append(node)
        return node['id']

    def add_visual(self,function,path,round_number,source):
        embedding=self.encoder([function])[0];nodes=self.data['stationary'];node=None
        if nodes:
            cosine,node=max(((float(np.dot(n['embedding'],embedding)),n) for n in nodes),key=lambda p:p[0])
            if cosine<self.options['stationary_threshold']:node=None
        if node is None:
            node={**self.metadata(round_number,source),'function':function,'embedding':embedding,'images':[]}
            nodes.append(node)
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if any(x['sha256']==digest for x in node['images']):return node['id'],False
        node['images'].append({**self.metadata(round_number,source),'path':str(Path(path).resolve()),'sha256':digest})
        return node['id'],True
