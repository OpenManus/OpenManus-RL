import asyncio
import json
import os
import random
import shutil
import string
import time
from typing import Dict, List

from src.server.task import Task, Session
from src.typings import TaskSampleExecutionResult, SampleIndex, SampleStatus, TaskOutput
from .judger.cal_metric import calculate
from .server import Server
from .utils import run_cmd


class CardGame(Task):
    def __init__(self, **kwargs):
        self.port = kwargs.pop("port", 12345)
        self.test_time = kwargs.pop("test_time", 1)
        self.root_dir = "./src/server/tasks/card_game"
        self.cache_dir = "./data/card_game/.cache"
        self.server = Server(port=self.port, workers=kwargs["workers"])
        self.port = self.server.port
        super().__init__(**kwargs)

        self.data = self.get_data()

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.data)))

    def calculate_overall(self, results: List[TaskOutput]) -> Dict:
        print("calculate_overall")
        results = [result.result for result in results]
        print(results)
        return self._cal_metric(results)

    @staticmethod
    def _cal_metric(prediction):
        win_round = 0
        total_round = 0
        total_hp = 0
        total_damage = 0
        for dict_ in prediction:
            x = dict_["meta"]
            for (key, val) in x.items():
                win_round += val["win_round"]
                total_round += val["test_times"]
                total_damage += val["damage"]
                total_hp += 1600
        if total_round == 0:
            return {
                "win_round": win_round,
                "total_round": total_round,
                "win_rate": -1,
                "damage_rate": -1,
                "score": -1,
            }
        else:
            return {
                "win_round": win_round,
                "total_round": total_round,
                "win_rate": win_round / total_round,
                "damage_rate": total_damage / total_hp,
                "score": win_round / total_round * 0.7 + min(0.3, total_damage / total_hp * 0.3),
            }

    @staticmethod
    def _random_string(length):
        chars = string.ascii_lowercase + string.digits
        return "".join(random.choices(chars, k=length))

    @staticmethod
    def _delete_dir(directory):
        try:
            shutil.rmtree(directory)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (directory, e))

    # run for one round
    def get_data(self) -> List:
        ret = []
        for stage in [2]:
            for base in ["baseline1", "baseline2"]:
                for agent in [0, 1]:
                    for i in range(self.test_time):
                        ret.append(
                            ({"stage": stage, "base": base, "agent": agent}, None)
                        )
        return ret

    async def start_sample(self, index, session: Session) -> TaskSampleExecutionResult:
        data_item = self.data[index][0]
        location = {
            "ai": f"python+{self.root_dir}/AI/main.py+en+%d+%d+%s+%d",
            "baseline1": f"python+{self.root_dir}/AI/basline1.py+%d+%d+%s",
            "baseline2": f"python+{self.root_dir}/AI/basline2.py+%d+%d+%s",
        }
        # a round consist of 50 subrounds
        folder = self._random_string(16)
        total = {}

        stage = data_item["stage"]
        base = data_item["base"]
        agent = data_item["agent"]
        if agent == 0:
            result_dir = f"{self.cache_dir}/result/{folder}_{stage}_ai_{base}"
        else:
            result_dir = f"{self.cache_dir}/result/{folder}_{stage}_{base}_ai"
        stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(int(time.time())))
        save_dir = f"{result_dir}/{stamp}"
        os.makedirs(save_dir)

        cmd = f"python {self.root_dir}/judger/judger.py {self.root_dir}/logic/bin/main %s %s config {save_dir}/replay.json"

        if agent == 0:
            cmd = cmd % (
                location["ai"] % (stage, 0, save_dir, self.port),
                location[base] % (stage, 1, save_dir),
            )
        else:
            cmd = cmd % (
                location[base] % (stage, 1, save_dir),
                location["ai"] % (stage, 0, save_dir, self.port),
            )

        task = asyncio.get_event_loop().create_task(self.server.start(folder, session))
        print("task created")
        task1 = asyncio.to_thread(run_cmd, cmd)
        msg = (await task1)[1]
        print("cmd executed")
        time.sleep(1)
        print("awaiting task")
        await task
        print("task done")
        current_log = self.server.log[folder]

        if agent == 0:
            meta = {"ai1": "ai", "ai2": base}
        else:
            meta = {"ai1": base, "ai2": "ai"}

        if "\"0\" : 0" in msg:
            meta["winner"] = "1"
        else:
            meta["winner"] = "0"

        with open(save_dir + "/meta.json", "w") as f:
            f.write(json.dumps(meta))
        ret = calculate(result_dir=result_dir, agent=agent)
        total[result_dir] = ret

        self._delete_dir(result_dir)

        status = self.server.status.get(folder, 0)
        if status == 0:
            finish_reason = SampleStatus.COMPLETED
        elif status == 1:
            finish_reason = SampleStatus.AGENT_INVALID_ACTION
        elif status == 2:
            finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
        elif status == 3:
            finish_reason = SampleStatus.AGENT_CONTEXT_LIMIT
        else:
            finish_reason = SampleStatus.UNKNOWN

        return TaskSampleExecutionResult(
            status=finish_reason,
            result={
                "data": json.dumps(data_item),
                "meta": total,
                "game": current_log,
                "state": meta,
                "folder": folder,
                "msg": msg
            },
        )
