import argparse
import asyncio
import traceback
from asyncio.exceptions import TimeoutError, CancelledError
from typing import TypeVar

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, APIRouter

from src.configs import ConfigLoader
from src.typings import *
from .task import Task, Session


class RunningSampleData:
    index: int
    session_id: int
    session: Session
    asyncio_task: asyncio.Task

    def __init__(self, index, session_id, session, task):
        self.index = index
        self.session_id = session_id
        self.session = session
        self.asyncio_task = task


_T = TypeVar("_T")


class TaskWorker:
    def __init__(
        self,
        task: Task,
        router: APIRouter,
        controller_address=None,
        self_address=None,
        heart_rate=8,
        register=True,
    ) -> None:
        self.session_map: Dict[int, RunningSampleData] = dict()
        self.session_lock = None
        self.app = app
        self.task = task
        self.controller_address = controller_address
        self.self_address = self_address
        self.heart_rate = heart_rate
        self.heart_thread = None
        self.router = router

        self.router.get("/get_indices")(self.get_indices)
        self.router.get("/get_sessions")(self.get_sessions)
        self.router.get("/worker_status")(self.worker_status)
        self.router.post("/sample_status")(self.sample_status)
        self.router.post("/start_sample")(self.start_sample)
        self.router.post("/interact")(self.interact)
        self.router.post("/cancel")(self.cancel)
        self.router.post("/cancel_all")(self.cancel_all)
        self.router.post("/calculate_overall")(self.calculate_overall)

        self.router.on_event("startup")(self._initialize)
        self.router.on_event("shutdown")(self.shutdown)

        if register:
            self.router.on_event("startup")(self.register)

    def _initialize(self):
        self.session_lock = asyncio.Lock()

    async def _call_controller(self, api: str, data: dict):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.controller_address + api,
                json=data,
            ) as response:
                if response.status != 200:
                    raise HTTPException(
                        400,
                        "Error: Controller returned error"
                        + "\n"
                        + (await response.text()),
                    )
                result = await response.json()
        return result

    async def register(self):
        asyncio.create_task(self.heart_beat())

    async def heart_beat(self):
        while True:
            try:
                await self._call_controller(
                    "/receive_heartbeat",
                    {
                        "name": self.task.name,
                        "address": self.self_address,
                        "concurrency": self.task.concurrency,
                        "indices": self.task.get_indices(),
                    },
                )
            except Exception as e:
                print("Heartbeat failed:", e)
            await asyncio.sleep(self.heart_rate)

    async def task_start_sample_wrapper(self, index: SampleIndex, session: Session, session_id: int):
        try:
            result = await self.task.start_sample(index, session)
        except Exception as _:
            self.session_map.pop(session_id)
            error = traceback.format_exc()
            await session.controller.env_finish(TaskOutput(
                index=index,
                status=SampleStatus.TASK_ERROR,
                result=error,
                history=session.history,
            ))
            return
        self.session_map.pop(session_id)
        await session.controller.env_finish(TaskOutput(
            index=index,
            status=result.status,
            result=result.result,
            history=session.history,
        ))

    async def start_sample(self, parameters: WorkerStartSampleRequest):
        print("job received")
        async with self.session_lock:
            if parameters.session_id in self.session_map:
                raise HTTPException(status_code=400, detail="Session ID already exists")
            print("session map:", self.session_map)
            if len(self.session_map) >= self.task.concurrency:
                raise HTTPException(
                    status_code=406,
                    detail="Sample concurrency limit reached: %d" % self.task.concurrency,
                )
            session = Session()
            print("session created")
            task_executor = self.task_start_sample_wrapper(
                parameters.index, session, parameters.session_id
            )
            t = asyncio.get_event_loop().create_task(task_executor)
            self.session_map[parameters.session_id] = RunningSampleData(
                index=parameters.index,
                session_id=parameters.session_id,
                session=session,
                task=t,
            )

        print("about to pull agent")
        env_output = await session.controller.agent_pull()
        print("output got")
        return {
            "session_id": parameters.session_id,
            "output": env_output.dict(),
        }

    async def interact(self, parameters: InteractRequest):
        print("interacting")
        async with self.session_lock:
            running = self.session_map.get(parameters.session_id, None)
        if running is None:
            raise HTTPException(status_code=400, detail="No such session")
        if running.session.controller.agent_lock.locked():
            raise HTTPException(
                status_code=400,
                detail="Task Executing, please do not send new request.",
            )
        print("awaiting agent pull in interact")
        response = await running.session.controller.agent_pull(
            parameters.agent_response
        )
        if response.status == SampleStatus.TASK_ERROR:
            raise HTTPException(status_code=501, detail={
                "session_id": parameters.session_id,
                "output": response.dict(),
            })
        return {
            "session_id": parameters.session_id,
            "output": response.dict(),
        }

    async def cancel_all(self):
        async with self.session_lock:
            sessions = list(self.session_map.keys())
            cancelling = []
            for session_id in sessions:
                cancelling.append(self.cancel(CancelRequest(session_id=session_id)))
        await asyncio.gather(*cancelling)

    async def cancel(self, parameters: CancelRequest):
        async with self.session_lock:
            if parameters.session_id not in self.session_map:
                raise HTTPException(status_code=400, detail="No such session")
            running = self.session_map.get(parameters.session_id)
            print("canceling", running)
            running.session.controller.env_input = AgentOutput(status=AgentOutputStatus.CANCELLED)
            running.session.controller.env_signal.release()
            print("awaiting task")
            try:
                await asyncio.wait_for(running.asyncio_task, timeout=30)
            except (TimeoutError, CancelledError):
                print("Warning: Task Hard Cancelled")
                self.session_map.pop(parameters.session_id)
            return {
                "session_id": parameters.session_id,
            }

    async def worker_status(self):
        return {
            "concurrency": self.task.concurrency,
            "current": len(self.session_map),
        }

    async def sample_status(self, parameters: SampleStatusRequest):
        async with self.session_lock:
            if parameters.session_id not in self.session_map:
                raise HTTPException(status_code=400, detail="No such session")
            running = self.session_map[parameters.session_id]
        return {
            "session_id": parameters.session_id,
            "index": running.index,
            "status": running.session.controller.get_status(),
        }

    async def get_sessions(self):
        return {sid: session.index for sid, session in self.session_map.items()}

    async def get_indices(self):
        return self.task.get_indices()

    async def calculate_overall(self, request: CalculateOverallRequest):
        return self.task.calculate_overall(request.results)

    async def shutdown(self):
        self.task.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("name", type=str, help="Task name")
    parser.add_argument(
        "--config", "-c", type=str, default="configs/tasks/task_assembly.yaml"
    )
    parser.add_argument(
        "--controller", "-C", type=str, default="http://localhost:5000/api"
    )
    parser.add_argument("--self", "-s", type=str, default="http://localhost:5001/api")
    parser.add_argument("--port", "-p", type=int, default=5001)

    args = parser.parse_args()

    conf = ConfigLoader().load_from(args.config)
    asyncio_task = InstanceFactory.parse_obj(conf[args.name]).create()

    app = FastAPI()
    router_ = APIRouter()
    task_worker = TaskWorker(
        asyncio_task,
        router_,
        controller_address=args.controller,
        self_address=args.self,
    )
    app.include_router(router_, prefix="/api")
    uvicorn.run(app=app, host="0.0.0.0", port=args.port)
