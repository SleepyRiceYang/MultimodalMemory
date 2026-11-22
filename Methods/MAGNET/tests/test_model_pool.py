"""Pool saturation, forwarding, planner repair and original checkpoint compatibility."""
import asyncio,copy,json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase,TestCase,mock
import httpx
from gui_baseline import model_pool
from gui_baseline.model_pool import Router
from gui_baseline.common import JsonModel,read
from gui_baseline.evolution import PLANNER_SCHEMA,validate_plan


class RoutingTests(IsolatedAsyncioTestCase):
    async def test_capacity_and_round_robin(self):
        pool=Router(['one','two','three'],4)
        ids=[await pool.acquire() for _ in range(12)]
        self.assertEqual(ids,[0,1,2]*4);self.assertEqual(pool.active,[4,4,4])
        waiting=asyncio.create_task(pool.acquire());await asyncio.sleep(.01)
        self.assertFalse(waiting.done());await pool.release(1)
        self.assertEqual(await asyncio.wait_for(waiting,1),1)
        self.assertEqual(pool.active,[4,4,4])

    async def test_request_forwarding_and_error_release(self):
        router=Router(['http://backend'],1);seen=[]
        async def backend(request):
            seen.append(request.content)
            return httpx.Response(400,content=b'{"error":"invalid request"}',headers={'content-type':'application/json'})
        raw=b'{"model":"test","messages":[],"temperature":0}'
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            with mock.patch.object(model_pool,'router',router):
                model_pool.app.state.client=client
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=model_pool.app),base_url='http://pool') as front:
                    response=await front.post('/v1/chat/completions',content=raw,headers={'X-Experiment-Benchmark':'osworld'})
        self.assertEqual(seen,[raw]);self.assertEqual(response.status_code,400)
        self.assertEqual(router.active,[0]);self.assertEqual(router.failures,[1])
        self.assertEqual(router.clients['osworld'],[1])


class PlannerRepairTests(TestCase):
    def test_invalid_plan_is_not_cached_and_only_repair_changes_format(self):
        def response(value):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(value)))],
                model_dump=lambda:{'value':value})
        good={'subtasks':['Open Settings'],'next_subtask':'Open Settings','reason':'Current screen'}
        fake=mock.Mock();fake.chat.completions.create.side_effect=[response({**good,'subtasks':[]}),response(good)]
        with TemporaryDirectory() as folder,mock.patch('gui_baseline.common.OpenAI',return_value=fake):
            model=JsonModel('http://pool/v1')
            out=model.generate(folder,'original prompt',{'goal':'Wifi'},max_tokens=64,repair_schema=PLANNER_SCHEMA,validator=validate_plan)
            self.assertEqual(out,good);self.assertEqual(read(Path(folder)/'result.json'),good)
            calls=fake.chat.completions.create.call_args_list
            self.assertEqual(calls[0].kwargs['response_format'],{'type':'json_object'})
            self.assertEqual(calls[1].kwargs['response_format']['type'],'json_schema')
            self.assertEqual(calls[0].kwargs['messages'],calls[1].kwargs['messages'])
            model.generate(folder,'original prompt',{'goal':'Wifi'},max_tokens=64,repair_schema=PLANNER_SCHEMA,validator=validate_plan)
            self.assertEqual(fake.chat.completions.create.call_count,2)
