import datetime
import json
import os
import random
import threading
import time
import signal
from typing import Dict, List, Union
from typing import Tuple, Callable, Iterator
import contextlib
import sys
from tqdm.contrib import DummyTqdmFile

import yaml
from tqdm import tqdm

from src.client.task import TaskError
from .client import TaskClient, AgentClient
from .configs import ConfigLoader
from .typings import AssignmentConfig, SampleIndex, TaskOutput, TaskClientOutput
from .utils import ColorMessage
from .utils import Graph, MaxFlow
from time import sleep
import contextlib
import sys
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

class Assigner:
    def __init__(self, config: AssignmentConfig, auto_retry: bool = True) -> None:
        """
        Logic:
            1. Check if output folder exists (resume or create)
            2. Walk through all the folders in output folder, and remove the finished samples
            3. Create agents
        """
        self.auto_retry = auto_retry
        self.tqdm_ordered_by_agent = {}
        self.overall_tqdm = None
        self.config = config
        self.free_worker = config.concurrency.copy(deep=True)
        self.agents: Dict[str, AgentClient] = {}
        self.tasks: Dict[str, TaskClient] = {}
        self.task_indices: Dict[str, List[SampleIndex]] = {}
        self.task_worker_fail_count: Dict[str, int] = {}
        self.assignment_lock = threading.Lock()
        self.remaining_tasks: Dict[
            str, Dict[str, List[int]]
        ] = {}  # {agent: {task: [index]}}
        self.completions: Dict[
            str, Dict[str, List[TaskOutput]]
        ] = {}  # {agent: {task: [{index: int, result: JSONSerializable}]}}
        self.finished_count = 0
        self.started_count = 0
        self.running_count = 0
        
        # Flag to indicate whether we're shutting down
        self.shutting_down = False
        
        # Timer for score updates
        self.score_update_timer = None
        self.score_update_interval = 60  # Update score every 60 seconds

        # Step 1. Check if output folder exists (resume or create)

        if not os.path.exists(self.config.output):
            os.makedirs(self.config.output)
            # Write config file
            with open(os.path.join(self.config.output, "config.yaml"), "w") as f:
                f.write(yaml.dump(self.config.dict()))

        # Step 2. walk through all the folders in output folder({output}/agent/task/runs.jsonl),
        # and remove the finished samples

        for assignment in self.config.assignments:
            agent = assignment.agent
            task = assignment.task
            runs_file = os.path.join(self.get_output_dir(agent, task), "runs.jsonl")
            result_file = os.path.join(self.get_output_dir(agent, task), "overall.json")
            if os.path.exists(result_file):
                continue
            if agent not in self.remaining_tasks:
                self.remaining_tasks[agent] = {}
            if task not in self.remaining_tasks[agent]:
                self.remaining_tasks[agent][task] = []
            if task not in self.tasks:
                print(ColorMessage.green(f"creating {task} client..."))
                self.tasks[task] = self.config.definition.task[task].create()
                self.task_indices[task] = self.tasks[task].get_indices()
            self.remaining_tasks[agent][task] = self.task_indices[task].copy()
            if not os.path.exists(runs_file):
                continue
            with open(runs_file, "r") as f:
                for line in f:
                    try:
                        run = json.loads(line)
                        run.pop("time")
                        index = run.pop("index")
                        assert index is not None
                        run = TaskClientOutput.parse_obj(run)
                        assert isinstance(run.output, TaskOutput)
                    except:
                        continue
                    if index in self.remaining_tasks[agent][task]:
                        self.remaining_tasks[agent][task].remove(index)
                        self.record_completion(agent, task, index, run.output)
                    else:
                        print(
                            ColorMessage.yellow(
                                f"Warning: {agent}/{task}#{index} is finished, but not in the index list."
                            )
                        )

        count = sum(
            [
                len(self.remaining_tasks[agent][task])
                for agent in self.remaining_tasks
                for task in self.remaining_tasks[agent]
            ]
        )
        print(
            ColorMessage.cyan(f"Message: {count} samples remaining.")
        )

        for agent in self.remaining_tasks:
            agent_ = json.dumps(agent)
            tasks_ = len(self.remaining_tasks[agent])
            samples_ = sum(
                [
                    len(self.remaining_tasks[agent][task])
                    for task in self.remaining_tasks[agent]
                ]
            )
            if samples_ == 0:
                continue
            print(
                ColorMessage.cyan(
                    f"Agent {agent_} needs to run {tasks_} tasks with total {samples_} samples:"
                )
            )
            for task in self.remaining_tasks[agent]:
                print(
                    ColorMessage.cyan(
                        f"    Task {json.dumps(task)}: {len(self.remaining_tasks[agent][task])}"
                    )
                )

        # Create agents

        for agent in self.remaining_tasks:
            self.agents[agent] = self.config.definition.agent[agent].create()

    def get_output_dir(self, agent: str, task: str) -> str:
        return os.path.join(self.config.output, agent, task)

    def worker_generator(
        self, interval=10
    ) -> Iterator[Tuple[str, str, SampleIndex]]:

        node_list = ["SRC", "DST"]
        agent_node_index = {}
        task_node_index = {}
        for agent in self.agents:
            node_list.append(agent)
            agent_node_index[agent] = len(node_list) - 1
        for task in self.tasks:
            node_list.append(task)
            task_node_index[task] = len(node_list) - 1

        while not self.shutting_down:

            # Step 0. Get real time task free worker

            with self.assignment_lock:
                for task in self.tasks:
                    self.free_worker.task[task] = self.tasks[task].get_concurrency()
                print("Running Count: {}".format(self.running_count))

            # Step 1. init edges: SRC -> agent -> task -> DST

            with self.assignment_lock:
                edges = {}
                for agent in self.agents:
                    edges[(0, agent_node_index[agent])] = self.free_worker.agent[agent]
                for task in self.tasks:
                    edges[(task_node_index[task], 1)] = self.free_worker.task[task]
                tot_remaining_samples = 0
                for agent in self.remaining_tasks:
                    for task in self.remaining_tasks[agent]:
                        tot_remaining_samples += len(self.remaining_tasks[agent][task])
                        edges[(agent_node_index[agent], task_node_index[task])] = len(
                            self.remaining_tasks[agent][task]
                        )
            if tot_remaining_samples == 0:
                if self.running_count == 0:
                    break
                else:
                    time.sleep(interval / 2 + random.random() * interval)
                    continue

            # Step 2. Create graph and calculate max flow

            graph = Graph(node_count=len(node_list), edges=edges)
            max_flow = MaxFlow(graph, src=0, dst=1)

            if max_flow.max_flow == 0:
                time.sleep(interval / 2 + random.random() * interval)
                continue

            # Step 3. yield all (agent, task, index) tuples

            for (src, dst), e in max_flow.edges_dict.items():
                if (
                    src not in agent_node_index.values()
                    or dst not in task_node_index.values()
                ):
                    continue
                if e.flow == 0:
                    continue
                agent = node_list[src]
                task = node_list[dst]
                for _ in range(e.flow):
                    with self.assignment_lock:
                        if len(self.remaining_tasks[agent][task]) == 0:
                            continue
                        index = self.remaining_tasks[agent][task].pop()
                        self.free_worker.agent[agent] -= 1
                        self.free_worker.task[task] -= 1
                    print(ColorMessage.green(f"Assigned {agent}/{task}#{index}"))
                    yield agent, task, index

            # Step 4. sleep for a while
            time.sleep(interval / 2 + random.random() * interval)

    def save_all_results(self):
        """Calculate and save partial results for all agent-task pairs"""
        for agent in self.completions:
            for task in self.completions[agent]:
                self.save_partial_results(agent, task)
        
    def save_partial_results(self, agent, task):
        """Calculate and save partial results for a specific agent-task pair"""
        try:
            if task in self.tasks and agent in self.completions and task in self.completions[agent]:
                task_client = self.tasks[task]
                overall = task_client.calculate_overall(self.completions[agent][task])
                output_dir = self.get_output_dir(agent, task)
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "overall.json"), "w") as f:
                    f.write(json.dumps(overall, indent=4, ensure_ascii=False))
                return overall
        except Exception as e:
            print(ColorMessage.yellow(f"Warning: Failed to save partial results for {agent}/{task}: {str(e)}"))
        return None
    
    def display_current_scores(self):
        """Calculate and display the current scores for all agent-task pairs"""
        print("\n" + "="*50)
        print(ColorMessage.cyan(f"Current Results ({self.finished_count}/{self.started_count} samples completed):"))
        
        for agent in self.completions:
            for task in self.completions[agent]:
                try:
                    overall = self.save_partial_results(agent, task)
                    if overall:
                        # Different tasks may have different score formats, try to extract main metric
                        if "custom" in overall and isinstance(overall["custom"], dict):
                            main_metrics = []
                            for key, value in overall["custom"].items():
                                if isinstance(value, (int, float)):
                                    main_metrics.append(f"{key}: {value:.4f}")
                            if main_metrics:
                                print(ColorMessage.green(f"  {agent}/{task}: {', '.join(main_metrics)}"))
                            else:
                                print(ColorMessage.green(f"  {agent}/{task}: Results saved to overall.json"))
                        else:
                            print(ColorMessage.green(f"  {agent}/{task}: Results saved to overall.json"))
                except Exception as e:
                    print(ColorMessage.yellow(f"  {agent}/{task}: Error calculating score - {str(e)}"))
        
        print("="*50 + "\n")
    
    def update_scores_timer(self):
        """Timer function to periodically update and display scores"""
        if self.shutting_down:
            return
            
        self.display_current_scores()
        
        # Schedule the next update if we're still running
        if not self.shutting_down:
            self.score_update_timer = threading.Timer(self.score_update_interval, self.update_scores_timer)
            self.score_update_timer.daemon = True
            self.score_update_timer.start()

    def start(self, tqdm_out=None):
        self.started_count = sum(
            [
                len(self.remaining_tasks[agent][task])
                for agent in self.remaining_tasks
                for task in self.remaining_tasks[agent]
            ]
        )
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            print(ColorMessage.yellow("\nReceived Ctrl+C. Waiting for current tasks to complete..."))
            self.shutting_down = True
            if self.score_update_timer:
                self.score_update_timer.cancel()
                
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start the score update timer
        self.score_update_timer = threading.Timer(self.score_update_interval, self.update_scores_timer)
        self.score_update_timer.daemon = True
        self.score_update_timer.start()
        
        generator = self.worker_generator()
        self.overall_tqdm = tqdm(
            total=self.started_count,
            desc="Total",
            position=0,
            file=tqdm_out,
        )
        for idx, agent in enumerate(self.remaining_tasks.keys()):
            self.tqdm_ordered_by_agent[agent] = tqdm(
                total=sum(
                    [
                        len(self.remaining_tasks[agent][task])
                        for task in self.remaining_tasks[agent]
                    ]
                ),
                desc=agent,
                position=idx + 1,
                file=tqdm_out,
            )
        try:
            while True:
                try:
                    agent, task, index = next(generator)
                except StopIteration:
                    break
                self.start_worker(agent, task, index, self.finish_callback)
                
            # If we're shutting down, wait for running tasks to finish
            if self.shutting_down:
                print(ColorMessage.yellow("Waiting for running tasks to finish..."))
                while self.running_count > 0:
                    time.sleep(1)
                    
        finally:
            # Cancel the timer if it's still active
            if self.score_update_timer:
                self.score_update_timer.cancel()
                
            # Calculate and save results for all tasks, even if incomplete
            print(ColorMessage.yellow("Saving partial results..."))
            self.save_all_results()
            
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint_handler)

            self.overall_tqdm.close()
            for agent in self.tqdm_ordered_by_agent:
                self.tqdm_ordered_by_agent[agent].close()

            # Display final results
            print(ColorMessage.cyan("\nFinal Results:"))
            self.display_current_scores()
                
            final_message = (
                "\n\n============================================\n"
                + ColorMessage.cyan(f"Message: {self.started_count} sample(s) started. ")
                + "\n"
                + ColorMessage.green(
                    f"   >> {self.finished_count} sample(s) finished successfully."
                )
                + "\n"
            )
            if self.started_count != self.finished_count:
                final_message += (
                    ColorMessage.red(
                        f"   >> {self.started_count - self.finished_count} sample(s) failed."
                    )
                    + "\n"
                )
            final_message += (
                ColorMessage.cyan(
                    f"   >> results are saved to {self.config.output}"
                )
                + "\n"
            )
            final_message += "============================================\n\n"
            print(final_message)

    def record_completion(
        self, agent: str, task: str, index: SampleIndex, result: TaskOutput
    ):
        def calculate_overall_worker():
            nonlocal agent, task, index, result
            self.save_partial_results(agent, task)

        with self.assignment_lock:
            if agent not in self.completions:
                self.completions[agent] = {}
            if task not in self.completions[agent]:
                self.completions[agent][task] = []
            result.index = index
            self.completions[agent][task].append(result)
            
            # Always save partial results if all samples for this task are completed
            overall_calculation = len(self.completions[agent][task]) == len(self.task_indices[task])
            
        if overall_calculation:
            output_dir = self.get_output_dir(agent, task)
            if os.path.exists(os.path.join(output_dir, "overall.json")):
                return
            threading.Thread(target=calculate_overall_worker).start()

    def finish_callback(
        self, agent: str, task: str, index: SampleIndex, result: TaskClientOutput
    ):
        if result.error == TaskError.NOT_AVAILABLE.value:
            print(
                ColorMessage.yellow(
                    f"Warning: {task} is not available, retrying."
                )
            )
            with self.assignment_lock:
                if not self.shutting_down:
                    self.remaining_tasks[agent][task].insert(0, index)
                self.free_worker.agent[agent] += 1
                self.free_worker.task[task] += 1
                self.running_count -= 1
            return

        if result.error is not None:
            print(ColorMessage.yellow(f"Warning: {agent}/{task}#{index} "
                                      f"failed with error {result.error} {result.info} {result.output}"))
            if self.auto_retry and not self.shutting_down:
                with self.assignment_lock:
                    self.remaining_tasks[agent][task].insert(0, index)

        output_folder = self.get_output_dir(agent, task)
        os.makedirs(output_folder, exist_ok=True)
        timestamp: int = int(time.time() * 1000)
        time_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        write_to_file = (
            json.dumps(
                {
                    "index": index,
                    **result.dict(),
                    "time": {"timestamp": timestamp, "str": time_str},
                }
            )
            + "\n"
        )
        if not result.error:
            target_file = os.path.join(output_folder, "runs.jsonl")
            with self.assignment_lock:
                self.finished_count += 1
            self.record_completion(agent, task, index, result.output)
            self.overall_tqdm.update(1)
            self.tqdm_ordered_by_agent[agent].update(1)
        else:
            target_file = os.path.join(output_folder, "error.jsonl")
        with open(target_file, "a+", encoding="utf-8") as f:
            f.write(write_to_file)

        with self.assignment_lock:
            self.free_worker.agent[agent] += 1
            self.free_worker.task[task] += 1
            self.running_count -= 1

    def start_worker(
        self,
        agent: str,
        task: str,
        index: SampleIndex,
        finish_callback: Union[
            Callable[[str, str, SampleIndex, TaskClientOutput], None], None
        ] = None,
    ):
        def worker_thread():
            nonlocal agent, task, index, finish_callback

            result = self.tasks[task].run_sample(index, self.agents[agent])

            if finish_callback:
                finish_callback(agent, task, index, result)

        with self.assignment_lock:
            self.running_count += 1
        threading.Thread(target=worker_thread).start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default="configs/assignments/default.yaml"
    )
    parser.add_argument(
        "--auto-retry", "-r", action="store_true", dest="retry"
    )
    parser.add_argument(
        "--score-interval", "-i", type=int, default=60,
        help="Interval in seconds to display current scores (default: 60)"
    )
    args = parser.parse_args()

    loader = ConfigLoader()
    config_ = loader.load_from(args.config)
    value = AssignmentConfig.parse_obj(config_)
    value = AssignmentConfig.post_validate(value)
    v = value.dict()
    with std_out_err_redirect_tqdm() as orig_stdout:
        assigner = Assigner(value, args.retry)
        if args.score_interval:
            assigner.score_update_interval = args.score_interval
        assigner.start(tqdm_out=orig_stdout)