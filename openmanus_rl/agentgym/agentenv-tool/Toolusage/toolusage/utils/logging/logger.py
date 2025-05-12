import wandb
import os
import json
import re
import logging
import plotly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class SummaryLogger:
    def __init__(self, log_path, baseline_dir= "data/baseline_results"):

        self.dimension_scoring = {  
            "Memory": {"alfworld": 1, "scienceworld": 2, "babyai": 1, "jericho": 1, "pddl": 2, "webshop": 1, "webarena": 3, "tool-query": 2, "tool-operation": 3},  
            "Planning": {"alfworld": 1, "scienceworld": 2, "babyai": 2, "jericho": 3, "pddl": 3, "webshop": 2, "webarena": 3, "tool-query": 2, "tool-operation": 2},  
            "World Modeling": {"alfworld": 3, "scienceworld": 3, "babyai": 2, "jericho": 3, "pddl": 1, "webshop": 1, "webarena": 3, "tool-query": 1, "tool-operation": 1},  
            "Self-reflection": {"alfworld": 3, "scienceworld": 2, "babyai": 2, "jericho": 1, "pddl": 3, "webshop": 2, "webarena": 2, "tool-query": 1, "tool-operation": 1},  
            "Grounding": {"alfworld": 2, "scienceworld": 3, "babyai": 2, "jericho": 1, "pddl": 3, "webshop": 3, "webarena": 3, "tool-query": 3, "tool-operation": 3},  
            "Spatial Navigation": {"alfworld": 2, "scienceworld": 2, "babyai": 2, "jericho": 2, "pddl": 1, "webshop": 1, "webarena": 2, "tool-query": 1, "tool-operation": 1}  
        }  

        self.baseline_dir = baseline_dir
        self.current_run_metrics = []
        self.log_path = os.path.join(log_path, "all_results.txt")
        self.log_dimension_path = os.path.join(log_path, "dimension.txt")

    
    def check_metric_item_is_logged(self, metric_type, file_name):
        with open(file_name) as f:
            for line in f:
                if metric_type in line:
                    return True
        return False
    
    def log_run_result(self, task_name, success_rate, reward_score, grounding_acc, hard_sr, easy_sr, hard_pr, easy_pr):
        result = {"task_name":  task_name,
                    "success_rate":  success_rate,
                    "progress_rate":  reward_score,
                    "grounding_acc": grounding_acc,
                    "success_rate_hard": hard_sr,
                    "success_rate_easy": easy_sr,
                    "progress_rate_hard": hard_pr,
                    "progress_rate_easy": easy_pr
                    }
            
        self.current_run_metrics.append(result)
        
        if not self.check_metric_item_is_logged(task_name, self.log_path):
            with open(self.log_path, "a+") as f:
                f.write(json.dumps(result) + "\n")
    
    def load_baseline_results(self, task_name, baseline_dir):
        # load baseline success rate, reward score, grounding accuracy from baseline_dir
        baseline_results = {}
        for model_name in os.listdir(baseline_dir):
            model_path = os.path.join(baseline_dir, model_name)
            file_path = os.path.join(model_path, "all_results.txt")
            if not os.path.exists(file_path):
                continue
            else:
                for line in open(file_path, "r"):
                    try:
                        result = json.loads(line.strip())
                        if result["task_name"] == task_name:
                            baseline_results[model_name] = result
                    except:
                        continue
        
        return baseline_results
    
    def log_summary_metric(self):
        tasks_type = {
            "embodied": ["alfworld", "scienceworld", "babyai"],
            "game": ["jericho", "pddl"],
            "web": ["webshop", "webarena"],
            "tool": ["tool-query", "tool-operation"],
            "all": ["alfworld", "scienceworld", "babyai", "jericho", "pddl", "webshop", "webarena", "tool-query", "tool-operation"]
        }
        
        # if all tasks in a type have been run, then calculate the average success rate, reward score for this type
        
        metrics_table = wandb.Table(columns=["Metric Name", "Metric Value (%)"])
        
        metrics_dict = dict()
        
        for type in tasks_type:
            tasks = tasks_type[type]
            results = []
            
            for task_name in tasks:
                for task_result in self.current_run_metrics:
                    if task_result["task_name"] == task_name:
                        results.append(task_result)
            
            if len(results) == len(tasks): 
                
                all_metrics = {}
                all_metrics["task_name"] = type+"_summary"
                
                for metric in ["success_rate", "progress_rate", "grounding_acc", "success_rate_hard", "success_rate_easy", "progress_rate_hard", "progress_rate_easy"]:
                    mean_metric = np.mean(np.array([task_result[metric] for task_result in results]))
                    all_metrics[metric] = mean_metric
                    metric_name = " ".join([word.capitalize() for word in metric.split("_")])
                    metrics_table.add_data(f"Average {type.capitalize()} {metric_name}", mean_metric)
                    
                if not self.check_metric_item_is_logged(type+"_summary", self.log_path):
                    with open(self.log_path, "a+") as f:
                        f.write(json.dumps(all_metrics) + "\n")
                    
                success_rate = all_metrics["success_rate"]
                progress_rate = all_metrics["progress_rate"]
                
                metrics_table.add_data(f"Average {type.capitalize()} Progress Rate", progress_rate)
                metrics_table.add_data(f"Average {type.capitalize()} Success Rate", success_rate)
                
                metrics_dict[type] = {"success_rate": success_rate, "progress_rate": progress_rate}
                
                if type == "all":
                    dimension_metrics = {}
                    DIMENSION_CATEGORIES = self.dimension_scoring.keys()
                    for dimension in DIMENSION_CATEGORIES:
                        weights = self.dimension_scoring[dimension]
                        weights_sum = sum([weights[task_name] for task_name in tasks_type["all"]] )
                        score = 0
                        for task_result in results:
                            task_name = task_result["task_name"]
                            score += weights[task_name] * task_result["success_rate"] * 100 
                        score /= weights_sum
                        dimension_metrics[dimension] = score
                    
                    with open(self.log_dimension_path, "w") as f:
                        f.write(json.dumps(dimension_metrics))

        wandb.log({"summary/metrics": metrics_table})
        
        # if all tasks in a type have been run, then draw a bar plot to compare the avg performance of different models
        
        if "all" in metrics_dict:
            
            # first calculate the average performance of each baseline model
            avg_baseline_results = dict()

            for task_name in tasks_type["all"]:
                baseline_results = self.load_baseline_results(task_name, self.baseline_dir)
                baseline_models = list(baseline_results.keys())
                for model_name in baseline_models:
                    if model_name not in avg_baseline_results:
                        avg_baseline_results[model_name] = {"success_rate": [], "progress_rate": []}
                    avg_baseline_results[model_name]["success_rate"].append(baseline_results[model_name]["success_rate"])
                    avg_baseline_results[model_name]["progress_rate"].append(baseline_results[model_name]["progress_rate"])
            
            all_baseline_models = list(avg_baseline_results.keys())
            for model_name in all_baseline_models:
                if len(avg_baseline_results[model_name]["success_rate"]) == len(tasks_type["all"]):
                    
                    avg_baseline_results[model_name]["success_rate"] = np.mean(avg_baseline_results[model_name]["success_rate"])
                    avg_baseline_results[model_name]["progress_rate"] = np.mean(avg_baseline_results[model_name]["progress_rate"])
                else:
                    del avg_baseline_results[model_name]
            
            metrics = metrics_dict["all"]
            baseline_models = list(baseline_results.keys())
            models = ["Current Run"] + [ model_name.capitalize() for model_name in baseline_models]
            
            accuracys = [metrics["success_rate"]] + [ avg_baseline_results[model_name]["success_rate"] for model_name in baseline_models ]
            rewards = [metrics["progress_rate"]] + [ avg_baseline_results[model_name]["progress_rate"] for model_name in baseline_models ]
            
            marker_color_acc=['rgba(0,128, 255, 1)'] + ['rgba(0,128, 255, 0.6)'] * len(baseline_models)
            marker_color_reward=['rgba(51, 255,153, 1)'] + ['rgba(51, 255,153, 0.6)'] * len(baseline_models)
            
            data=[
                go.Bar(name='Progress Rate (%)', x=models, y=rewards, marker_color=marker_color_reward),
                go.Bar(name='Success Rate (%)', x=models, y=accuracys, marker_color=marker_color_acc)
            ]
            
            layout = go.Layout(
                width=800,
                height=400,
                xaxis={'categoryorder':'total descending'},
                title='Average Metrics for All Tasks Compared to Baseline Models',
            )
            
            fig = go.Figure(data=data, layout=layout)
            wandb.log({"summary/avg_metrics_comparison": wandb.Plotly(fig)})        
        
                
    def log_summary(self):
        
        # first log the average success rate, reward score, grounding accuracy for all tasks and types
        self.log_summary_metric()
        
        # first get all baselines
        all_results = dict()
        
        for task_result in self.current_run_metrics:
            task_name = task_result["task_name"]
            results_dict = dict()
            results_dict["Current Run"] = task_result
            
            # add results to results_dict
            results_dict.update(self.load_baseline_results(task_name, self.baseline_dir))
            all_results[task_name] = results_dict
            
        valid_baseline_models = set(all_results[list(all_results.keys())[0]].keys())
        for task_name in all_results:
            valid_baseline_models = valid_baseline_models.intersection(set(all_results[task_name].keys()))
            
        for task_name in all_results:
            for model_name in list(all_results[task_name].keys()):
                if model_name not in valid_baseline_models:
                    del all_results[task_name][model_name]
        # draw a radar chart for all tasks that have been run in this run
        
        CATEGORIES = all_results.keys()
        N = len(CATEGORIES)
        result_df = pd.DataFrame(columns=["model_name", "task_name", "success_rate"])
        for task_name in CATEGORIES:
            for model_name in all_results[task_name]:
                dash = False if model_name == "Current Run" else True
                result_df = result_df.append({"model_name": model_name, "task_name": task_name, "success_rate": 100 * all_results[task_name][model_name]["success_rate"], "baseline": dash}, ignore_index=True)
        
        radar_results = px.line_polar(result_df,
                    r = 'success_rate',
                    theta = 'task_name',
                    line_close = True,
                    category_orders = {"category": CATEGORIES},
                    color = 'model_name',
                    markers=True,
                    labels={'success_rate': 'Success Rate (%)', 'task_name': 'Task Name', 'model_name': 'Model Name'},
                    line_dash="baseline"
                    )

        radar_results.update_layout(
            width=700,
            height=400,
            title='Success Rate (%) w.r.t Tasks for All Models',  
            title_x=0.1,
            legend_title_text='', 
        )
        wandb.log({"summary/all_results": wandb.Html(plotly.io.to_html(radar_results))})
        
        
        #  draw a radar chart based on dimension scoring
        
        DIMENSION_CATEGORIES = self.dimension_scoring.keys()
        dimension_df = pd.DataFrame(columns=["model_name", "dimension", "score"])
        
        for dimension in DIMENSION_CATEGORIES:
            weights = self.dimension_scoring[dimension]
            weights = [ weights[task_name] for task_name in CATEGORIES ]
            weights_sum = sum(weights)
            
            for model_name in all_results[list(CATEGORIES)[0]]:
                score = []
                for i, task_name in enumerate(CATEGORIES):
                    score.append((100 * all_results[task_name][model_name]["success_rate"], self.dimension_scoring[dimension][task_name]))
                    
                score = [ i[0] * i[1]  for i in score ]
                score = sum(score) / weights_sum
                
                dash = False if model_name == "Current Run" else True
                dimension_df = dimension_df.append({"model_name": model_name, "dimension": dimension, "score": score, "baseline": dash}, ignore_index=True)
        
        radar_dimension = px.line_polar(dimension_df,
                    r = 'score',
                    theta = 'dimension',
                    line_close = True,
                    category_orders = {"category": DIMENSION_CATEGORIES},
                    color = 'model_name',
                    markers=True,
                    line_dash="baseline",
                    labels={'score': 'Score', 'dimension': 'Dimension', 'model_name': 'Model Name'},
                    )
                
        radar_dimension.update_layout(
            width=700,
            height=400,
            title='Agent Ability Dimension Score w.r.t Models',  
            title_x=0.1,
            legend_title_text='',
        )
         
        wandb.log({"summary/agent_abilities": wandb.Html(plotly.io.to_html(radar_dimension))}) 
        # wandb.log({"summary/agent_abilities": wandb.Plotly(radar_dimension)})          
        

class TaskLogger:
    def __init__(self, task_name, log_path, max_num_steps=30, baseline_dir= "data/baseline_results"):
        self.task_name = task_name
        
        columns=["id", "is_done", "env", "reward", "grounding_accuracy", "reward_wrt_step", "trajectory"]
        self.table = wandb.Table(columns=columns)
        self.max_num_steps = max_num_steps
        self.baseline_dir = baseline_dir
        self.columns = columns
        
        self.log_path = os.path.join(log_path, "logs", f"{task_name}.jsonl")
        self.log_summary_path = os.path.join(log_path, f"{task_name}.txt")
        
        with open(self.log_path, "w") as f:
            f.write("")
            f.close()
            
        with open(self.log_summary_path, "w") as f:
            f.write("")
            f.close()
            
        
        self.baseline_metrics, self.baseline_reward_wrt_step = self.load_baseline_results()
    
    def extract_variables(self, line):
        pattern = r"\[EXP\] (\d+): \[success_rate\]: (.*), \[progress_rate\]: (.*), \[grounding_acc\]: (.*), \[score_state\]: (.*)"
        match = re.match(pattern, line)
        if match:
            i = int(match.group(1))
            sr_temp = match.group(2)
            if sr_temp=="True": sr_temp = 1 
            if sr_temp=="False": sr_temp = 0
            sr = float(sr_temp)
            score = float(match.group(3))
            grounding_acc = float(match.group(4))
            score_state_str = match.group(5)
            score_state = eval(score_state_str)
        
            # make score_state index integer, and value float
            score_state = [ (int(step), float(score)) for step, score in score_state ]
            return_dict = {
                "EXP": i,
                "success_rate": sr,
                "progress_rate": score,
                "grounding_acc": grounding_acc,
                "score_state": score_state
            }
            return return_dict
    
    def complete_score_state(self, score_state):
        complete_state = []
        current_score = 0
        for step in range(self.max_num_steps):
            if score_state and step == score_state[0][0]:
                current_score = score_state.pop(0)[1]
            complete_state.append((step, current_score))
        return complete_state

    
    def load_baseline_results(self):
        # load baseline success rate, reward score, grounding accuracy from baseline_dir
        baseline_results = {}
        for model_name in os.listdir(self.baseline_dir):
            model_path = os.path.join(self.baseline_dir, model_name)
            file_path = os.path.join(model_path, "all_results.txt")
            if not os.path.exists(file_path):
                continue
            else:
                for line in open(file_path, "r"):
                    try:
                        result = json.loads(line.strip())
                        if result["task_name"] == self.task_name:
                            baseline_results[model_name] = result
                    except:
                        continue
        
        # load baseline reward_wrt_step from baseline_dir
        baseline_reward_wrt_step = {}
        for model_name in os.listdir(self.baseline_dir):
            model_path = os.path.join(self.baseline_dir, model_name)
            file_path = os.path.join(model_path, f"{self.task_name}.txt")
            if not os.path.exists(file_path):
                continue
            else:
                results = []
                for line in open(file_path, "r"):
                    result = self.extract_variables(line)
                    result['score_state'] = self.complete_score_state(result['score_state'])
                    results.append(result) 
                # acculated score
                reward_score_list = []

                # initialize reward score
                for i in range(self.max_num_steps):
                    reward_score_list.append(0)

                for result in results:
                    for step, score in result['score_state']:
                        reward_score_list[step] += score

                # normalize reward score
                for i in range(self.max_num_steps):
                    reward_score_list[i] /= len(results)

                reward_score_list = [ i*100 for i in reward_score_list[:self.max_num_steps] ]

                # at step 0, reward score is 0
                reward_score_list.insert(0, 0)
                
                baseline_reward_wrt_step[model_name] = reward_score_list
                        
        return baseline_results, baseline_reward_wrt_step
            
        
    def log_example_data(self, id, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory):
        if self.task_name not in ["webarena"]:
            is_done = bool(is_done)
        reward = float(reward)
        grounding_accuracy = float(grounding_accuracy)
        
        # draw a line plot in wandb for reward_wrt_steps
        reward_wrt_steps = [0] * self.max_num_steps
        for step, reward in score_change_record:
            for i in range(int(step), self.max_num_steps):
                reward_wrt_steps[i] = reward
        # data = [[x, y] for (x, y) in zip(range(self.max_num_steps), reward_wrt_steps)]
        # table = wandb.Table(data=data, columns=["step", "reward"])
        plt.figure(figsize=(4, 4))
        # draw the following plot on the figures
        plt.plot(range(self.max_num_steps), reward_wrt_steps, color='blue', marker='o', linestyle='solid', linewidth=1, markersize=2)
        # plt.xlabel("step")
        # plt.ylabel("reward")
        ax = plt.gca()#获取边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim(0, 1)
        
        
        # build a piece of html code for the trajectory
        html_head = '''
            <!DOCTYPE html>  
            <html>  
            <head>  
            <style>  
                .goal { color: black; font-size: 10px; }  
                .observation { color: grey; font-size: 10px; }  
                .progress { color: brown; font-size: 10px; }  
                .action { color: green; font-size: 10px; }
            </style>  
            </head>  
            <body>  
            
        '''
        
        html_body = ""
        for item in trajectory:
            type = list(item.keys())[0]
            type_name=type
            step_id = item["id"]
            content = item[type]
            if isinstance(content,str) and len(content.split('\n')) > 5:
                content = "\n".join(content.split('\n')[:5]) + "\n   ..."
                
            if type == "Progress Rate": type = "progress"
            html = '''
            <p class="{type}"><b>Step {step_id} </b><b>{type_name}: </b>{content}</p>
            '''.format(type=type.lower(), step_id = step_id, type_name=type_name, content=content)
            html_head += html
        
        html_tail = '''
            </body>
            </html>
        '''
        html_text = html_head + html_body + html_tail
        
        
        self.table.add_data(id, is_done, env_details, reward, grounding_accuracy, 
                            wandb.Image(plt), # add the plot as an image
                            wandb.Html(html_text) # add the trajectory as html
                            )
        
    def update(self):
        new_table = wandb.Table(
                columns=self.table.columns, data=self.table.data
            )
        wandb.log({"{task}/predictions".format(task=self.task_name) : new_table})
    
    def save_sample_data_to_file_detailed(self, id, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory, example_prompt):
        if self.task_name not in ["webarena"]:
            is_done = bool(is_done)
        reward = float(reward)
        grounding_accuracy = float(grounding_accuracy)
        
        sample_result = dict()
        sample_result["id"] = id
        sample_result.update(env_details)
        sample_result["is_done"] = is_done
        sample_result["progress_rate"] = reward
        sample_result["grounding_acc"] = grounding_accuracy
        sample_result["score_change_record"] = score_change_record
        sample_result["trajectory"] = {}
        
        for item in trajectory:
            type = list(item.keys())[0]
            step_id = int(item["id"])
            content = item[type]
            
            step_name = f"Interaction Turn {step_id}"
            
            if step_name not in sample_result["trajectory"]:
                sample_result["trajectory"][step_name] = dict()
                # sample_result["trajectory"][int(step_id)]["Interaction Turn"] = step_id
            sample_result["trajectory"][step_name][type] = content
        
        if example_prompt is not None:
            sample_result["example_prompt"] = example_prompt

        with open(self.log_path, "a+") as f:
            f.write(json.dumps(sample_result, indent=2)+'\n')
    
    def save_sample_data_to_file_overview(self, id, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory):
        with open(self.log_summary_path, "a+") as f:
            f.write(f"[EXP] {id}: [success_rate]: {is_done}, [progress_rate]: {reward}, [grounding_acc]: {grounding_accuracy}, [score_state]: {score_change_record} \n")

        
           
    def log_example(self, id, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory, example_prompt=None):
        self.save_sample_data_to_file_detailed(id, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory, example_prompt) # log to file
        self.save_sample_data_to_file_overview(id, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory) 
        
        self.log_example_data(id, is_done, reward, grounding_accuracy, score_change_record, env_details, trajectory) # log to wandb table
        self.update()
        
    
    
    def log_summary(self, success_rate, reward_score, grounding_acc, score_steps, hard_sr=None, hard_rs=None, easy_sr=None, easy_rs=None):
        # wandb.log({"{task}/success_rate".format(task=self.task_name) : success_rate,
        #            "{task}/reward_score".format(task=self.task_name) : reward_score,
        #            "{task}/grounding_acc".format(task=self.task_name) : grounding_acc})
        
        # log success rate, reward score, grounding accuracy to a table
        metrics_table = wandb.Table(columns=["Metric Name", "Metric Value (%)"])
        metrics_table.add_data("Progress Rate", reward_score)
        metrics_table.add_data("Success Rate", success_rate)
        metrics_table.add_data("Grounding Accuracy", grounding_acc)
        wandb.log({"{task}/metrics".format(task=self.task_name) : metrics_table})
        
        # Limit this display due to too much information
        # if hard_sr is not None:
        #     wandb.log({"{task}/hard_success_rate".format(task=self.task_name) : hard_sr,
        #                "{task}/easy_success_rate".format(task=self.task_name) : easy_sr})
            
        
        # draw a bar chart for alfworld accuracy and reward with figure size 6*6
        
        baseline_models = list(self.baseline_metrics.keys())
        models = ["Current Run"] + baseline_models
        
        accuracys = [success_rate] + [ self.baseline_metrics[model_name]["success_rate"] for model_name in baseline_models ]
        rewards = [reward_score] + [ self.baseline_metrics[model_name]["progress_rate"] for model_name in baseline_models ]
        grounding_accs = [grounding_acc] + [ self.baseline_metrics[model_name]["grounding_acc"] for model_name in baseline_models ]
        
        marker_color_acc=['rgba(0,128, 255, 1)'] + ['rgba(0,128, 255, 0.6)'] * len(baseline_models)
        marker_color_reward=['rgba(0, 204,102, 1)'] + ['rgba(0, 204,102, 0.6)'] * len(baseline_models)
        marker_color_grounding=['rgba( 248,173,30, 1)'] + ['rgba(248,173,30, 0.6)'] * len(baseline_models)

        data=[
            go.Bar(name='Progress Rate (%)', x=models, y=rewards, marker_color=marker_color_reward),
            go.Bar(name='Success Rate (%)', x=models, y=accuracys, marker_color=marker_color_acc),
            go.Bar(name='Grounding Accuracy (%)', x=models, y=grounding_accs, marker_color=marker_color_grounding)
        ]

        layout = go.Layout(
            width=800,
            height=400,
            xaxis={'categoryorder':'total descending'},
            title='{task} Metrics Compared to Baseline Models'.format(task=self.task_name.capitalize()),
            
        )

        fig = go.Figure(data=data, layout=layout)
        wandb.log({"{task}/metrics_comparison".format(task=self.task_name) : wandb.Plotly(fig)})
                
        
        # draw a line plot in wandb for reward_wrt_steps
        df = pd.DataFrame(columns=["models", "steps", "score"])

        # add current run
        reward_score_list = []
        for i in range(self.max_num_steps):
            reward_score_list.append(0)
        
        for score_step_example in score_steps:
            score_step_example = self.complete_score_state(score_step_example)
            for step, score in score_step_example:
                reward_score_list[step] += score
        
        for i in range(self.max_num_steps):
            reward_score_list[i] /= len(score_steps)
        
        reward_score_list = [ i*100 for i in reward_score_list[:self.max_num_steps] ]
        reward_score_list.insert(0, 0)
        
        
        for step, score in enumerate(reward_score_list):
            df = df.append({"models": "Current Run", "steps": step, "score": score, "baseline": False}, ignore_index=True)
        
        # add baseline runs
        for model in list(self.baseline_metrics.keys()):
            for step, score in enumerate(self.baseline_reward_wrt_step[model]):
                df = df.append({"models": model, "steps": step, "score": score, "baseline": True}, ignore_index=True)
                
        # legend of line_fig only contains models
        
        line_fig = px.line(df, x="steps", y="score", color='models', title="Average Progress Rate (%) w.r.t Steps for {task} Tasks".format(task=self.task_name), width=800, height=400, line_dash="baseline",
                           labels={"models": "Model Name", "baseline": "Is Baseline"})
        
        plot = wandb.Plotly(line_fig)
        
        wandb.log({"{task}/task_reward_w.r.t_steps".format(task=self.task_name): plot})
        
        if hard_sr is not None: 
            
            easy_sr_data = [easy_sr] + [ self.baseline_metrics[model_name]["success_rate_easy"] for model_name in baseline_models ]
            hard_sr_data = [hard_sr] + [ self.baseline_metrics[model_name]["success_rate_hard"] for model_name in baseline_models ]
            
            difficulty_sr_data = [
                go.Bar(name='Success Rate For Easy Examples(%)', y=models, x=easy_sr_data, marker_color=['rgba(102, 255, 255, 1)'] + ['rgba(102, 255,255, 0.6)'] * len(baseline_models), orientation='h'),
                go.Bar(name='Success Rate For Hard Examples(%)', y=models, x=hard_sr_data, marker_color=['rgba(0, 128, 255, 0.4)'] + ['rgba(0, 128,255, 0.4)']  * len(baseline_models), orientation='h')
            ]
            layout_sr_difficulty = go.Layout(
                width=800,
                height=400,
                yaxis={'categoryorder':'total ascending'},
                barmode='overlay',
                title='{task} Success Rate w.r.t Difficulty'.format(task=self.task_name.capitalize()),
            )
            
            fig_sr_difficulty = go.Figure(data=difficulty_sr_data, layout=layout_sr_difficulty)
            
            easy_rs_data = [easy_rs] + [ self.baseline_metrics[model_name]["progress_rate_easy"] for model_name in baseline_models ]
            hard_rs_data = [hard_rs] + [ self.baseline_metrics[model_name]["progress_rate_hard"] for model_name in baseline_models ]
            
            difficulty_rs_data = [
                go.Bar(name='Progress Rate For Easy Examples(%)', y=models, x=easy_rs_data, marker_color=['rgba(0,255,128, 1)'] + ['rgba(0,255,128, 0.6)'] * len(baseline_models), orientation='h'),
                go.Bar(name='Progress Rate For Hard Examples(%)', y=models, x=hard_rs_data, marker_color=['rgba(0, 153,76, 0.6)'] + ['rgba(0, 153,76, 0.6)'] * len(baseline_models), orientation='h')
            ]
            
            layout_rs_difficulty = go.Layout(
                width=800,
                height=400,
                barmode='overlay',
                yaxis={'categoryorder':'total ascending'},
                title='{task} Progress Rate w.r.t Difficulty'.format(task=self.task_name.capitalize()),
            )
            fig_rs_difficulty = go.Figure(data=difficulty_rs_data, layout=layout_rs_difficulty)
            fig_rs_difficulty.for_each_trace(lambda t: t.update(name = '<b>' + t.name +'</b>') if t.name in "Current Run" else())
            
            wandb.log({"{task}/success_rate_w.r.t_difficulty".format(task=self.task_name) : wandb.Plotly(fig_sr_difficulty)})    
            wandb.log({"{task}/progress_score_w.r.t_difficulty".format(task=self.task_name) : wandb.Plotly(fig_rs_difficulty)})