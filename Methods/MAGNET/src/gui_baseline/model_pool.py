"""Local OpenAI-compatible least-inflight pool. Request bodies are forwarded unchanged."""
import asyncio,json,os,time
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
from fastapi import FastAPI,Request
from fastapi.responses import Response,JSONResponse,StreamingResponse


class Router:
    def __init__(self,urls,limit=4):
        self.urls=urls;self.limit=limit;self.active=[0]*len(urls);self.calls=[0]*len(urls)
        self.failures=[0]*len(urls);self.cursor=0;self.condition=asyncio.Condition();self.clients={}
    async def acquire(self):
        async with self.condition:
            await self.condition.wait_for(lambda:any(n<self.limit for n in self.active))
            n=len(self.urls);order=[(self.cursor+i)%n for i in range(n)]
            index=min(order,key=lambda i:self.active[i]);self.active[index]+=1;self.calls[index]+=1
            self.cursor=(index+1)%n;return index
    async def release(self,index,failed=False):
        async with self.condition:
            self.active[index]-=1;self.failures[index]+=int(failed);self.condition.notify_all()
    def stats(self):
        return [dict(url=u,active=self.active[i],requests=self.calls[i],failures=self.failures[i]) for i,u in enumerate(self.urls)]

router=Router(os.environ.get('GUI_POOL_BACKENDS','http://127.0.0.1:8113,http://127.0.0.1:8114,http://127.0.0.1:8116').split(','))
@asynccontextmanager
async def lifespan(app):
    app.state.client=httpx.AsyncClient(timeout=httpx.Timeout(180,connect=10),trust_env=False)
    yield
    await app.state.client.aclose()
app=FastAPI(lifespan=lifespan)

@app.get('/health')
async def health():
    async def check(url):
        try:
            r=await app.state.client.get(url+'/v1/models');r.raise_for_status()
            return {'url':url,'ready':any(m['id']=='qwen3-vl-8b-instruct' for m in r.json()['data'])}
        except Exception as exc:return {'url':url,'ready':False,'error':str(exc)}
    backends=await asyncio.gather(*(check(u) for u in router.urls))
    return {'ready':all(b['ready'] for b in backends),'backends':backends,'routing':router.stats()}

@app.get('/metrics')
async def metrics():return {'backends':router.stats(),'clients':router.clients}

@app.get('/v1/models')
async def models():
    r=await app.state.client.get(router.urls[0]+'/v1/models')
    return Response(r.content,status_code=r.status_code,media_type='application/json')

@app.post('/v1/chat/completions')
async def generate(request:Request):
    body=await request.body();index=await router.acquire();released=False
    tag=request.headers.get('x-experiment-benchmark','untagged')
    router.clients.setdefault(tag,[0]*len(router.urls))[index]+=1
    headers={'content-type':'application/json'}
    if request.headers.get('authorization'):headers['authorization']=request.headers['authorization']
    response=None
    try:
        req=app.state.client.build_request('POST',router.urls[index]+'/v1/chat/completions',content=body,headers=headers)
        response=await app.state.client.send(req,stream=True)
        if json.loads(body).get('stream'):
            async def chunks():
                try:
                    async for chunk in response.aiter_raw():yield chunk
                finally:
                    await response.aclose();await router.release(index,response.status_code>=400)
            released=True
            return StreamingResponse(chunks(),status_code=response.status_code,
                media_type=response.headers.get('content-type','text/event-stream'),headers={'x-pool-backend':router.urls[index]})
        content=await response.aread()
        return Response(content,status_code=response.status_code,media_type='application/json',headers={'x-pool-backend':router.urls[index]})
    except httpx.HTTPError as exc:
        return JSONResponse({'error':{'message':str(exc),'type':'pool_backend_error'}},status_code=502)
    finally:
        if not released:
            if response:await response.aclose()
            await router.release(index,response is None or response.status_code>=400)
