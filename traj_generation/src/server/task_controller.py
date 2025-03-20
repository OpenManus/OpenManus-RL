import argparse
import asyncio
import time
from asyncio.exceptions import TimeoutError

import aiohttp
import uvicorn
from aiohttp import ClientTimeout
from fastapi import FastAPI, HTTPException, APIRouter

from src.typings import *


class TimeoutLock(asyncio.Lock):
    def __init__(self, timeout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout

    async def acquire(self) -> Literal[True]:
        try:
            return await asyncio.wait_for(super().acquire(), self.timeout)
        except TimeoutError:
            print(ColorMessage.yellow("LOCK TIMEOUT"))
            raise

    def handle(self, lock: asyncio.Lock):
        class _Handler:
            def __init__(self, timeout_lock: TimeoutLock, handle_lock: asyncio.Lock):
                self.timeout_lock = timeout_lock
                self.handle_lock = handle_lock
                self.locked = False

            async def __aenter__(self):
                # assert self.handle_lock.locked()
                try:
                    await self.timeout_lock.acquire()
                    self.locked = True
                finally:
                    self.handle_lock.release()

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if self.locked:
                    self.timeout_lock.release()

        return _Handler(self, lock)


class SessionData:
    name: str
    index: SampleIndex
    start: float
    last_update: float
    worker_id: int
    lock: TimeoutLock

    def __init__(self, name: str, index: SampleIndex, worker_id: int) -> None:
        self.name = name
        self.index = index
        self.start = time.time()
        self.last_update = time.time()
        self.worker_id = worker_id
        self.lock = TimeoutLock(1)

    def dump(self):
        return {
            "name": self.name,
            "index": self.index,
            "start": self.start,
            "last_update": self.last_update,
            "worker_id": self.worker_id,
            "locked": self.lock.locked(),
        }


class WorkerData:
    id: int
    address: str
    capacity: int
    _current: int
    last_visit: float
    status: WorkerStatus
    lock: TimeoutLock

    def __init__(self, id_: int, address: str, capacity: int) -> None:
        self.id = id_
        self.address = address
        self.capacity = capacity
        self._current = 0
        self.last_visit = time.time()
        self.status = WorkerStatus.ALIVE
        self.lock = TimeoutLock(2)

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, value):
        assert value >= 0
        self._current = value

    def dump(self):
        return {
            "id": self.id,
            "address": self.address,
            "capacity": self.capacity,
            "current": self.current,
            "last_visit": self.last_visit,
            "status": self.status,
            "locked": self.lock.locked(),
        }


class TaskData:
    indices: List[SampleIndex]
    workers: Dict[int, WorkerData]

    def __init__(self, indices: List[SampleIndex]) -> None:
        self.indices = indices
        self.workers = {}
        self.next_worker_id = 0

    def get_worker_id(self):
        wid = self.next_worker_id
        self.next_worker_id += 1
        return wid

    def dump(self):
        return {
            "indices": self.indices,
            "workers": {i: w.dump() for i, w in self.workers.items()},
        }


class Sessions:
    def __init__(self):
        self.sessions: Dict[int, SessionData] = {}
        self.lock = None

    def dump(self):
        return {sid: session.dump() for sid, session in self.sessions.items()}

    def init_lock(self):
        self.lock = asyncio.Lock()

    def items(self):
        # assert self.lock.locked()
        return self.sessions.items()

    def keys(self):
        # assert self.lock.locked()
        return self.sessions.keys()

    def __contains__(self, item) -> bool:
        return item in self.sessions

    def __iter__(self):
        return iter(self.sessions)

    def __getitem__(self, key: int) -> SessionData:
        return self.sessions[key]

    def __setitem__(self, key: int, value):
        # assert self.lock.locked()
        # assert key not in self.sessions
        self.sessions[key] = value

    def __delitem__(self, key: int):
        # assert self.lock.locked()
        # assert key in self.sessions
        # assert self.sessions[key].lock.locked()
        del self.sessions[key]


class TaskController:
    def __init__(
        self,
        router: APIRouter,
        heart_rate: int = 11,
        session_expire_time: int = 240,
        clean_worker_time: int = 35,
    ) -> None:
        self.session_expire_time = session_expire_time
        self.clean_worker_time = clean_worker_time
        self.tasks: Dict[str, TaskData] = {}

        self.sessions = Sessions()
        self.session_next_id = 0

        self.tasks_lock = None

        self.router = router
        self.heart_rate = heart_rate

        self.router.get("/list_workers")(self.list_workers)
        self.router.get("/list_sessions")(self.list_sessions)
        self.router.get("/get_indices")(self.get_indices)
        self.router.post("/start_sample")(self.start_sample)
        self.router.post("/interact")(self.interact)
        self.router.post("/cancel")(self.cancel)
        self.router.post("/cancel_all")(self.cancel_all)
        self.router.post("/receive_heartbeat")(self.receive_heartbeat)
        self.router.post("/calculate_overall")(self.calculate_overall)
        self.router.post("/clean_worker")(self.clean_worker)
        self.router.post("/clean_session")(self.clean_session)
        self.router.post("/sync_all")(self.sync_all)

        self.router.on_event("startup")(self._initialize)
        self.router.on_event("startup")(lambda: asyncio.create_task(self._session_gc()))

    def _initialize(self):
        self.sessions.init_lock()
        self.tasks_lock = asyncio.Lock()

    async def _call_worker(
        self,
        name: str,
        worker_id: int,
        api: str,
        data: dict = None,
        method: str = "post",
        locked: bool = False,
        timeout: float = 240,
    ) -> dict:
        async with aiohttp.ClientSession(
            timeout=ClientTimeout(total=timeout)
        ) as session:
            try:
                if method == "post":
                    response = await session.post(
                        self.tasks[name].workers[worker_id].address + api,
                        json=data,
                    )
                elif method == "get":
                    response = await session.get(
                        self.tasks[name].workers[worker_id].address + api,
                        params=data,
                    )
            except Exception as e:
                print(ColorMessage.red(f"task {name} worker {worker_id} error {e}"))
                async with self.tasks_lock:
                    worker = self.tasks[name].workers[worker_id]
                    if not locked:
                        async with worker.lock:
                            worker.status = WorkerStatus.DEAD
                    else:
                        worker.status = WorkerStatus.DEAD
                raise HTTPException(400, "Error: Worker not responding\n" + str(e))
            if response.status != 200:
                raise HTTPException(
                    response.status,
                    "Error: Worker returned error" + "\n" + (await response.text()),
                )
            result = await response.json()
        return result

    async def list_workers(self):
        t = time.time()
        async with self.tasks_lock:
            for task in self.tasks.values():
                for worker in task.workers.values():
                    if t - worker.last_visit > self.heart_rate:
                        worker.status = WorkerStatus.COMA
        return {name: task.dump() for name, task in self.tasks.items()}

    async def list_sessions(self):
        return self.sessions.dump()

    async def receive_heartbeat(self, data: RegisterRequest):
        async with self.tasks_lock:
            if data.name not in self.tasks:
                self.tasks[data.name] = TaskData(indices=data.indices)
            elif data.indices != self.tasks[data.name].indices:
                raise HTTPException(
                    400, "Error: Task already exists with different indices"
                )
            for worker in self.tasks[data.name].workers.values():
                if worker.address == data.address:
                    worker.last_visit = time.time()
                    break
            else:
                wid = self.tasks[data.name].get_worker_id()
                self.tasks[data.name].workers[wid] = WorkerData(
                    id_=wid,
                    address=data.address,
                    capacity=data.concurrency,
                )
                return

        if worker.status != WorkerStatus.ALIVE:
            result = await self._sync_worker_status(data.name, worker.id)
            if not result:
                raise HTTPException(400, "Error: Worker status abnormal")

    async def start_sample(self, data: StartSampleRequest):
        print("starting")
        async with self.tasks_lock:
            if data.name not in self.tasks:
                raise HTTPException(406, "Error: Task does not exist")
            if data.index not in self.tasks[data.name].indices:
                raise HTTPException(400, "Error: Index out of bounds")
            max_delta = 0
            target_worker = None
            t = time.time()
            for worker in self.tasks[data.name].workers.values():
                if t - worker.last_visit > self.heart_rate:
                    worker.status = WorkerStatus.COMA
                if worker.status != WorkerStatus.ALIVE:
                    continue
                delta = worker.capacity - worker.current
                if delta > max_delta:
                    max_delta = delta
                    target_worker = worker
            if target_worker is None:
                raise HTTPException(406, "Error: No workers available")
            target_worker.current += 1
        print("worker selected")

        await self.sessions.lock.acquire()
        sid = self.session_next_id
        self.session_next_id += 1
        self.sessions[sid] = SessionData(
            name=data.name,
            index=data.index,
            worker_id=target_worker.id,
        )

        async with self.sessions[sid].lock.handle(self.sessions.lock):
            print("sending job")
            try:
                result = await self._call_worker(
                    data.name,
                    target_worker.id,
                    "/start_sample",
                    {
                        "index": data.index,
                        "session_id": sid,
                    },
                )
            except HTTPException as e:
                print(ColorMessage.red("job sending error"), e)
                async with self.tasks_lock:
                    async with self.sessions.lock:
                        target_worker.current -= 1
                        del self.sessions[sid]
                if e.status_code == 406:
                    await self._sync_worker_status(data.name, target_worker.id)
                raise

            print("job sent")

            if SampleStatus(result["output"]["status"]) != SampleStatus.RUNNING:
                print(ColorMessage.green("finishing session"), result["output"]["status"])
                await self._finish_session(sid)

            return result

    async def interact(self, data: InteractRequest):
        await self.sessions.lock.acquire()
        if data.session_id not in self.sessions:
            self.sessions.lock.release()
            raise HTTPException(400, "Error: Session does not exist")
        session = self.sessions[data.session_id]
        session.last_update = time.time()
        async with session.lock.handle(self.sessions.lock):
            result = await self._call_worker(
                session.name,
                session.worker_id,
                "/interact",
                data.dict(),
            )

            if "output" not in result:
                raise HTTPException(
                    400, "Error: Worker returned error" + "\n" + str(result)
                )

            async with self.tasks_lock:
                worker = self.tasks[session.name].workers[session.worker_id]
                async with worker.lock:
                    worker.last_visit = time.time()

            async with self.sessions.lock:
                self.sessions[data.session_id].last_update = time.time()

            print("[Server] interact result")

            if SampleStatus(result["output"]["status"]) != SampleStatus.RUNNING:
                print(ColorMessage.green("finishing session"), result["output"]["status"])
                await self._finish_session(data.session_id)

            return result

    async def _finish_session(self, sid: int):
        async with self.sessions.lock:
            if sid not in self.sessions:
                return
            # assert self.sessions[sid].lock.locked()
            session = self.sessions[sid]
            del self.sessions[sid]
            async with self.tasks_lock:
                if (
                    session.name in self.tasks
                    and session.worker_id in self.tasks[session.name].workers
                ):
                    worker = self.tasks[session.name].workers[session.worker_id]
                    async with worker.lock:
                        worker.current -= 1

    async def cancel(self, data: CancelRequest):
        sid = data.session_id
        await self.sessions.lock.acquire()
        if sid not in self.sessions:
            self.sessions.lock.release()
            raise HTTPException(400, "Error: Session does not exist")
        session = self.sessions[sid]
        async with session.lock.handle(self.sessions.lock):
            result = await self._call_worker(
                session.name,
                session.worker_id,
                "/cancel",
                data.dict(),
                timeout=5,
            )
            await self._finish_session(data.session_id)
            return result

    async def get_indices(self, name: str):
        async with self.tasks_lock:
            if name not in self.tasks:
                raise HTTPException(400, "Error: Task does not exist")
            return self.tasks[name].indices

    async def calculate_overall(self, data: CalculateOverallRequest):
        await self.tasks_lock.acquire()
        if data.name not in self.tasks:
            self.tasks_lock.release()
            raise HTTPException(400, "Error: Task does not exist")

        t = time.time()

        for worker in self.tasks[data.name].workers.values():
            if t - worker.last_visit > self.heart_rate:
                worker.status = WorkerStatus.COMA
            if worker.status == WorkerStatus.ALIVE:
                target_worker = worker
                break
        else:
            self.tasks_lock.release()
            raise HTTPException(400, "Error: No workers available")
        async with target_worker.lock.handle(self.tasks_lock):
            result = await self._call_worker(
                data.name,
                target_worker.id,
                "/calculate_overall",
                data.dict(),
            )

            return result

    async def _sync_worker_status(self, name: str, worker_id: int) -> bool:
        print(f"syncing {name} task worker {worker_id}")
        await self.tasks_lock.acquire()
        target_worker = self.tasks[name].workers[worker_id]
        async with target_worker.lock.handle(self.tasks_lock):
            # get status
            try:
                result = await self._call_worker(
                    name,
                    worker_id,
                    "/get_sessions",
                    method="get",
                    locked=True,
                )
            except Exception as e:
                print(
                    f"syncing {name} task worker {worker_id} at {target_worker.address} failed",
                    e,
                )
                async with self.tasks_lock:
                    target_worker.status = WorkerStatus.DEAD
                    return False

            result = {int(key): val for key, val in result.items()}

            # sync sessions
            async with self.sessions.lock:
                for sid, index in result.items():
                    if sid in self.sessions:
                        if (
                            self.sessions[sid].index == index
                            and self.sessions[sid].worker_id == worker_id
                            and self.sessions[sid].name == name
                        ):
                            continue
                        break
                    self.sessions[sid] = SessionData(
                        name,
                        index,
                        worker_id,
                    )
                else:
                    sessions = await self._gather_session(
                        lambda sid_, s: s.worker_id == worker_id
                        and s.name == name
                        and sid_ not in result
                    )
                    if sessions is None:
                        return False
                    for sid in sessions:
                        del self.sessions[sid]
                    target_worker.status = WorkerStatus.ALIVE
                    target_worker.current = len(result)
                    return True

            # session cannot match, hard sync
            print(ColorMessage.yellow("natural syncing failed, try to cancel all"))
            try:
                await self._call_worker(
                    name,
                    worker_id,
                    "/cancel_all",
                    locked=True,
                )
            except Exception as e:
                print(ColorMessage.red(
                    f"syncing {name} task worker {worker_id} at {target_worker.address} failed"
                ), e)
                async with self.tasks_lock:
                    self.tasks[name].workers[worker_id].status = WorkerStatus.DEAD
                    return False
            async with self.sessions.lock:
                sessions = await self._gather_session(
                    lambda _, s: s.worker_id == worker_id and s.name == name
                )
                if sessions is None:
                    return False
                for sid in sessions:
                    del self.sessions[sid]
            target_worker.current = 0
            target_worker.status = WorkerStatus.ALIVE
            target_worker.capacity = result["concurrency"]
            return True

    async def _gather_session(self, condition, allow_partial=False):
        # assert self.sessions.lock.locked()
        sessions = [sid for sid in self.sessions if condition(sid, self.sessions[sid])]
        locked = {sid: False for sid in sessions}

        async def acquire_lock(sid):
            try:
                await self.sessions[sid].lock.acquire()
                locked[sid] = True
            except TimeoutError:
                pass

        await asyncio.gather(*[acquire_lock(sid) for sid in sessions])
        partial = []
        for sid, lock in locked.items():
            if lock:
                partial.append(sid)
            else:
                sessions = None
        if allow_partial:
            return partial
        elif sessions is None:
            for sid, lock in locked.items():
                if lock:
                    self.sessions[sid].lock.release()
        return sessions

    async def sync_all(self):
        syncing = []

        async def sync_single(task_name, task_worker_id):
            try:
                await self._sync_worker_status(task_name, task_worker_id)
            except TimeoutError:
                print(ColorMessage.yellow(f"{task_name}#{task_worker_id} sync failed"))

        async with self.tasks_lock:
            for name in self.tasks:
                for worker_id in self.tasks[name].workers:
                    syncing.append(sync_single(name, worker_id))
        await asyncio.gather(*syncing)

    async def cancel_all(self):
        cancelling = []

        async def cancel_worker(task_name, task_worker):
            async with task_worker.lock:
                async with self.sessions.lock:
                    sessions = await self._gather_session(
                        lambda _, s: s.worker_id == task_worker.id and s.name == task_name
                    )
                    if sessions is None:
                        return
                try:
                    await self._call_worker(
                        task_name,
                        task_worker.id,
                        "/cancel_all",
                        locked=True,
                        timeout=30,
                    )
                except Exception as e:
                    print(ColorMessage.yellow(f"worker {task_name}#{task_worker.id} cancel all failed"), e)
                    async with self.tasks_lock:
                        self.tasks[task_name].workers[task_worker.id].status = WorkerStatus.DEAD
                    for sid in sessions:
                        self.sessions[sid].lock.release()
                else:
                    async with self.sessions.lock:
                        for sid in sessions:
                            del self.sessions[sid]
                    task_worker.current = 0

        async with self.tasks_lock:
            for name, task in self.tasks.items():
                for worker in task.workers.values():
                    if worker.status != WorkerStatus.ALIVE:
                        continue
                    cancelling.append(cancel_worker(name, worker))

        await asyncio.gather(*cancelling)

    async def clean_session(self):
        print("cleaning sessions")
        async with self.sessions.lock:
            sessions = await self._gather_session(
                lambda _, s: time.time() - s.last_update > self.session_expire_time,
                allow_partial=True,
            )
        for sid in sessions:
            session = self.sessions[sid]
            if (
                session.name not in self.tasks
                or session.worker_id not in self.tasks[session.name].workers
            ):
                await self._finish_session(sid)
                continue
            try:
                await self._call_worker(
                    session.name,
                    session.worker_id,
                    "/cancel",
                    {
                        "session_id": sid,
                    },
                )
            except HTTPException:
                session.lock.release()
                await self._sync_worker_status(session.name, session.worker_id)
            else:
                await self._finish_session(sid)

    async def _session_gc(self):
        while True:
            await asyncio.sleep(self.session_expire_time)
            try:
                await self.clean_session()
            except Exception as e:
                print(ColorMessage.red("session gc error"), e)

    async def clean_worker(self):
        print("clean workers")
        # both lists are edited, both locks are required
        async with self.tasks_lock, self.sessions.lock:
            task_to_be_removed = []
            t = time.time()
            for name, task in self.tasks.items():
                # gather dead workers
                worker_to_be_removed = []
                for i, worker in task.workers.items():
                    if t - worker.last_visit > self.heart_rate:
                        worker.status = WorkerStatus.COMA
                    if (
                        worker.status == WorkerStatus.DEAD
                        or worker.status == WorkerStatus.COMA
                        and worker.current == 0
                    ):
                        try:
                            await task.workers[i].lock.acquire()
                            worker_to_be_removed.append(i)
                        except TimeoutError:
                            pass

                # remove dead workers
                for i in worker_to_be_removed:
                    sessions = await self._gather_session(
                        lambda _, s: s.worker_id == i and s.name == name
                    )
                    if sessions is not None:
                        for sid in sessions:
                            del self.sessions[sid]
                        del task.workers[i]
                if not task.workers:
                    task_to_be_removed.append(name)
            for name in task_to_be_removed:
                del self.tasks[name]

    async def _worker_gc(self):
        while True:
            await asyncio.sleep(self.clean_worker_time)
            try:
                await self.clean_worker()
            except Exception as e:
                print(ColorMessage.red("worker gc error"), e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=5000)

    cmd_args = parser.parse_args()

    app = FastAPI()
    router_ = APIRouter()
    controller = TaskController(router_)
    app.include_router(router_, prefix="/api")
    uvicorn.run(app, host="0.0.0.0", port=cmd_args.port)
