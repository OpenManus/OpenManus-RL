# --------------------- Tool Use --------------------- #

TOOL_USE_TEMPLATE_NO_HIS = """
You are an expert research assistant capable of using various tools to gather information and solve complex problems.

Task: {task_description}

Available Tools:
{available_tools}

Current Observation: {current_observation}

Instructions:
1. Analyze the task and determine what information you need
2. Use available tools to gather information when needed
3. Reason through the information step by step  
4. When you have sufficient information, provide your final answer in <answer></answer> tags

Format for tool usage:
<action>
tool: [tool_name]
parameters: {{"param1": "value1", "param2": "value2"}}
</action>

Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <plan> </plan> tags.
Once you've finished your reasoning, you should either use a tool or provide your final answer within <answer> </answer> tags.
"""
TOOL_USE_TEMPLATE_LAST_STEP = """
You are an expert research assistant capable of using various tools to gather information and solve complex problems.

Task: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the full {history_length} observations and the corresponding actions you took: {action_history}

You are now at step {current_step} and this is the final step.
Current Observation: {current_observation}
You must provide your final answer within <answer> </answer> tags.
Even if the evidence is incomplete, infer the most plausible answer.
Never respond with "unknown", "cannot determine", or similar phrases.
"""

TOOL_USE_TEMPLATE = """
You are an expert research assistant capable of using various tools to gather information and solve complex problems.

Task: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}

You are now at step {current_step}.
Current Observation: {current_observation}

Available Tools:
{available_tools}

You should first recall relevant past experiences and reason from our conversation history, then MUST summarize within <memory_recall> </memory_recall> tags like this:

<memory>
Look at the past observations and actions from our conversation history.
- Please retrieve the most relavent memory for this step including the relevant observation and action in a RAG style along with the step number.
- These memory should be helpful milestones to solve this task.git reset --soft HEAD~1
</memory>

After that, you should reflect on the last action and its outcome, then MUST summarize within <reflection> </reflection> tags like this:

<reflection>
Reflect on the last action and its outcome
- Did I complete the task goal?
- Was last action successful or did it encounter issues?
- Am I making progress toward the task goal?
- If the action did not go as expected and did not result in progress, provide constructive feedback to guide the next planning step.
</reflection>

Given from the analysis from the memory and reflection, if we get the final answer, we should provide it within <answer> </answer> tags.
If we don't get the final answer, you should plan the next step based on memory and reflection, then MUST summarize within <plan> </plan> tags like this:

<plan>
Plan the next step based on memory and reflection
- Given what I've learned, what should I do next?
- Please explain why this plan is helpful for the next action?
- What do I expect this action to achieve?
</plan>

Finally, choose ONE admissible action for the current step and present it within the <action> </action> tags. 
<action>
action: [tool_name]  
parameters: {{"param1": "value1", "param2": "value2"}}
</action>

"""
