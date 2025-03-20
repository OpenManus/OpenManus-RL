import os
from typing import Dict, Any

from src.server.task import Task, Session
from src.server.tasks.alfworld.environment import SingleAlfredTWEnv
from src.server.tasks.alfworld.utils import *
from src.typings import TaskOutput, TaskSampleExecutionResult, SampleStatus, AgentOutputStatus
from copy import deepcopy
import traceback


class ALFWorld(Task):

    def __init__(self, **kwargs):
        # load data_path 
        self.data_path = kwargs.get("data_path", None)
        if self.data_path is None:
            raise Exception("missing parameter data_path")
        os.environ["ALFWORLD_DATA"] = self.data_path

        # load config for alfworld benchmark
        self.config_path = kwargs.get("config_path", None)
        if self.config_path is None:
            raise Exception("missing parameter config_path")
        self.config = load_config(self.config_path)

        # load prompts
        self.prompts_path = kwargs.get("prompts_path", None)
        if self.prompts_path is None:
            raise Exception("missing parameter prompts_path")
        self.prompts = load_prompts(self.prompts_path)

        # prepare data_files
        self.data_files = []
        self.split = kwargs.get("split", "dev")
        data_path = os.path.join("data/alfworld", f"{self.split}.json")
        with open(data_path, "r") as f:
            content = json.loads(f.read())
        for _, v in content.items():
            self.data_files.extend(v)
        self.data_files = [os.path.join(self.data_path, file) for file in self.data_files]
        print(f"> successfully loaded {len(self.data_files)} games")

        # other configs
        self.max_step = kwargs.get("max_step", 50)
        self.prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }

        super().__init__(**kwargs)

    def get_indices(self) -> List[Any]:
        return list(range(len(self.data_files)))

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        """
            TaskOutput.result 0/1
        """
        overall = {
            "total": len([config for config in results if config]),
            "pass": len([config for config in results if
                         (config and config.result and int(config.result.get("result", 0) == 1))]),
        }
        overall["wrong"] = overall["total"] - overall["pass"]
        overall["success_rate"] = overall["pass"] / overall["total"] if overall["total"] else 0
        return {
            "overall": overall,
        }

    # def predict_all(self, agent: Agent, inputs: List) -> List:
    #     print(f"Start Predicting All ...")

    #     count = self.workers
    #     if self.worker_limit:
    #         count = min(self.workers, self.worker_limit)

    #     executor = ProcessPoolExecutor(max_workers=count)
    #     results = []
    #     processes = []

    #     parameters = [(input_, agent, index) for index, input_ in enumerate(inputs)]
    #     for idx, parameter in enumerate(parameters):
    #         future = executor.submit(self.call_wrap, parameter)
    #         processes.append(future)

    #     with tqdm(total=len(parameters)) as pbar:
    #         for process in as_completed(processes):
    #             results.append(process.result())
    #             pbar.update(1)

    #     return results

    # def call_wrap(self, parameters):
    #     data_item, agent, index = parameters
    #     session = agent.create_session()
    #     result = self.predict_single(session, data_item, agent.name)
    #     self.save_single(index, data_item, result, session)
    #     return result

    async def start_sample(self, index, session: Session) -> TaskSampleExecutionResult:
        print("start sample")
        data_item = self.data_files[index]
        print("creating env")
        env = SingleAlfredTWEnv(self.config, data_item)
        print("initializing env")
        env = env.init_env(batch_size=1)
        try:
            print("running env")
            result, log_info, finish_reason = await self.alfworld_run(session, env)
        except Exception as e:
            print("error", e)
            traceback.print_exc()
            return TaskSampleExecutionResult(status=SampleStatus.UNKNOWN, result={"result": False, "error": e})
        log_info.update({"result": result})
        return TaskSampleExecutionResult(status=finish_reason, result=log_info)

    def release(self):
        if getattr(self, "env", None) is not None:
            del self.env

    @staticmethod
    def get_task_instruction():
        # return "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. In the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you should choose from two actions: \"THOUGHT\" or \"ACTION\". If you choose \"THOUGHT\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THOUGHT: your thoughts.\n ACTION: your next action\n\"; If you choose \"ACTION\", you should directly output the action in this turn. Your output must strictly follow this format:\"ACTION: your next action\n\". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\n"
        return "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: \"THOUGHT\" or \"ACTION\". If you choose \"THOUGHT\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"THOUGHT: your thoughts.\n ACTION: your next action\n\"; If you choose \"ACTION\", you should directly output the action in this turn. Your output must strictly follow this format:\"ACTION: your next action\n\". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.\n\n"

    def get_prompt(self, filename: str):
        # return []
        for k, v in self.prefixes.items():
            if filename.startswith(k):
                example = self.prompts[v]
                return deepcopy(example)
        raise Exception(f"unsupported name: {filename}")
        # return self.prompts["naive_example"]

    @staticmethod
    def inject_info(session: Session, history: List):
        current_role = "user"
        traverse = {"user": "agent", "agent": "user"}
        for his in history:
            session.inject({"role": current_role, "content": his})
            current_role = traverse[current_role]

    @staticmethod
    def get_available_actions(actions):
        actions = "\n".join(actions)
        return " AVAILABLE ACTIONS: " + actions + "\n"

    async def alfworld_run(self, session: Session, env: Any):
        finish_reason = SampleStatus.COMPLETED
        # env init
        ob, info = env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        # history = self.get_prompt(name)
        # add instruction
        # history[0] = self.get_task_instruction() + "Here is one example.\n" + history[0]
        # self.inject_info(session, history)
        log_info = {"log": []}
        session.inject({"role": "user", "content": self.get_task_instruction()})
        session.inject(
            {"role": "agent", "content": "OK. I'll follow your instructions and try my best to solve the task."})

        # 1-shot naive example
        history = self.get_prompt(name)
        history[0] = "Here is one example.\n" + history[0]
        self.inject_info(session, history)

        init_prompt = "Here is your task. " + ob + self.get_available_actions(info.get('admissible_commands', [[]])[0])
        log_info["init_prompt"] = init_prompt
        session.inject({"role": "user", "content": init_prompt})
        # init 
        # for his in session.history:
        #     print(his)
        # print("============ history end ==============")

        # interact
        for i in range(0, self.max_step):
            output = await session.action()
            if output.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                finish_reason = SampleStatus.AGENT_CONTEXT_LIMIT
                break
            output = output.content or ""

            # process action
            admissible_commands = info.get('admissible_commands', [[]])[0]
            action = process_action(output, admissible_commands)
            if not action:
                finish_reason = SampleStatus.AGENT_INVALID_ACTION
                break
            session.history[-2].content = session.history[-2].content.split("AVAILABLE ACTIONS")[
                0]  # reduce the prompt length

            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            session.inject({"role": "user", "content": observation + self.get_available_actions(
                info.get('admissible_commands', [[]])[0])})

            # save
            payload = {
                "round": i + 1,
                "output": output,
                "action": action,
                "admissible_commands": admissible_commands,
                "observation": observation,
                "done": done,
            }
            log_info["log"].append(payload)

            # print("====== action ======")
            # print("output: ", output)
            # print("action: ", action)
            # print("obs and commands: ", session.history[-1]["content"])
            # print("====== end action =======")
            # print("====== ob =======")
            # print("ob: ", observation)
            # print("====== end ob =======\n")
            # print("action: ", action, "ob: ", observation)

            # failure test
            if len(log_info["log"]) > 3:
                pre_logs = log_info["log"][-3:]
                pre_acts = [pre_log["output"] for pre_log in pre_logs]
                if len(list(set(pre_acts))) == 1:
                    print("repeat actions for 3 times: failure")
                    return 0, log_info, SampleStatus.AGENT_INVALID_ACTION

            if done:
                return reward, log_info, finish_reason
        else:
            finish_reason = SampleStatus.TASK_LIMIT_REACHED
        return 0, log_info, finish_reason
