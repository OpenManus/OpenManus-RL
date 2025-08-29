# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
Please respond with your structured analysis following the format you were instructed:
First, recall relevant past experiences and reason from our conversation history within <memory_recall> </memory_recall> tags.
Then, reflect on your last action's outcome within <reflection> </reflection> tags.
Next, plan the next step based on memory and reflection within <think> </think> tags.
Finally, specify your admissible action from the list within <action> </action> tags.
"""

ALFWORLD_OPENMANUS_INITIAL_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

Current observation: {current_observation}
Available actions: [{admissible_actions}]

Please begin by analyzing the situation and planning your approach:

<think>
Analyze the current situation and devise a plan to accomplish the task: {task_description}
What are the key steps needed to complete this task?
Based on the current observation, what should be our immediate next step?
How does this action advance our plan toward completing the task?
</think>

Now, present your chosen action:

<action>
action_choice: [selected admissible action from the list]
action_parameters: {{relevant details about the action if applicable}}
</action>

From now on, I will provide you with observations after each action, and you should respond with memory recall, reflection, thinking, and your next action in this format:

<memory_recall>
[Recall relevant past experiences and reason from our conversation history]
- What similar situations have I encountered?
- What strategies worked or failed before?
- What objects or locations have I discovered?
- What was my previous reasoning and plans?
</memory_recall>

<reflection>
[Reflect on the last action and its outcome]
- What did my last action accomplish?
- Was it successful or did it encounter issues?
- How does this outcome affect my plan?
- Am I making progress toward the task goal?
</reflection>

<think>
[Plan the next step based on memory and reflection]
- Given what I've learned, what should I do next?
- How does this action fit into my overall strategy?
- What do I expect this action to achieve?
</think>

<action>
action_choice: [selected admissible action from the list]
action_parameters: {{relevant details about the action if applicable}}
</action>
"""

# Keep the old template name for backward compatibility
ALFWORLD_OPENMANUS_TEMPLATE = ALFWORLD_OPENMANUS_INITIAL_TEMPLATE