from __future__ import annotations
import base64, io, json, time, os
from pathlib import Path
import requests
from PIL import Image
from openai import OpenAI


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def image_part(path, max_side=1280):
    with Image.open(path) as source:
        im=source.convert('RGB');im.thumbnail((max_side,max_side));buf=io.BytesIO();im.save(buf,format='PNG')
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}}


class JsonModel:
    def __init__(self, url, model='qwen3-vl-8b-instruct'):
        self.client=OpenAI(base_url=url,api_key='EMPTY',timeout=180,max_retries=2,default_headers={'X-Experiment-Benchmark':os.environ.get('GUI_POOL_CLIENT_TAG','untagged')});self.model=model

    def generate(self, folder, system, payload, images=(), max_tokens=2048, *, image_labels=None, response_schema=None, validator=None, repair_schema=None):
        folder=Path(folder);spec={'system':system,'payload':payload,'images':[str(p) for p in images],
                                 'model':self.model,'max_tokens':max_tokens}
        if response_schema is not None:spec['response_schema']=response_schema
        if repair_schema is not None:spec['repair_schema']=repair_schema
        if image_labels is not None:
            if len(image_labels)!=len(images):raise ValueError('Image labels and images differ in length')
            spec['image_labels']=list(image_labels)
        if (folder/'result.json').exists():
            if read(folder/'request.json')!=spec:raise ValueError(f'Cached request changed: {folder}')
            result=read(folder/'result.json')
            if validator is not None:validator(result)
            return result
        write(folder/'request.json',spec)
        content=[{'type':'text','text':json.dumps(payload,ensure_ascii=False)}]
        for i,path in enumerate(images):
            if image_labels is not None:content.append({'type':'text','text':image_labels[i]})
            content.append(image_part(path))
        for attempt in range(2):
            print(f'GENERATE {folder.name} attempt={attempt+1}/2',flush=True)
            format_schema=response_schema or (repair_schema if attempt>0 else None)
            response=self.client.chat.completions.create(model=self.model,temperature=0,max_tokens=max_tokens,
                response_format=({'type':'json_schema','json_schema':{'name':'planner','strict':True,'schema':format_schema}} if format_schema is not None else {'type':'json_object'}),messages=[{'role':'system','content':system},{'role':'user','content':content}])
            write(folder/f'response_{attempt}.json',response.model_dump())
            try:
                if response.choices[0].finish_reason=='length':raise ValueError('Truncated JSON generation')
                result=json.loads(response.choices[0].message.content)
                if not isinstance(result,dict):raise ValueError('Expected JSON object')
                if validator is not None:validator(result)
                write(folder/'result.json',result);return result
            except (ValueError,TypeError) as exc:write(folder/f'error_{attempt}.json',{'error':str(exc)})
        raise ValueError(f'Invalid generation after two attempts: {folder}')


def encode(url, texts, query=False):
    session=requests.Session();session.trust_env=False
    r=session.post(url.rstrip('/')+'/encode',json={'texts':texts,'query':query},timeout=120);r.raise_for_status()
    return r.json()['embeddings']
