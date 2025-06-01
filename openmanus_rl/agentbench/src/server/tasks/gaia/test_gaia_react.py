import requests
import json

def test_gaia_react_format():
    # 服务器地址
    base_url = "http://127.0.0.1:8081"
    
    # 1. 创建新环境
    create_response = requests.post(
        f"{base_url}/create",
        json={
            "data_dir": "data/",
            "level": "level1",
            "dataset": "validation"
        }
    )
    print(f"Create response: {create_response.text}")
    env_idx = create_response.text.strip('"')  # 移除引号
    print(f"Created environment with ID: {env_idx}")
    
    if not env_idx:
        print("Failed to create environment")
        return
    
    # 2. 获取初始观察
    obs_response = requests.get(
        f"{base_url}/observation",
        params={"env_idx": env_idx}
    )
    print(f"Observation response: {obs_response.text}")
    try:
        initial_observation = obs_response.json().get("observation")
    except:
        initial_observation = obs_response.text
    print(f"\nInitial observation:\n{initial_observation}")
    
    # 3. 测试ReAct格式的响应
    # 使用完整的ReAct格式，包含Thought和Action
    react_response = """Thought: I need to search for information about the capital of France.
Action: web_search Action Input: What is the capital of France?"""
    
    # 4. 执行动作
    step_response = requests.post(
        f"{base_url}/step",
        json={
            "env_idx": env_idx,
            "action": react_response
        }
    )
    print(f"Step response: {step_response.text}")
    try:
        result = step_response.json()
    except:
        result = {"response": step_response.text}
    print(f"\nStep result:\n{json.dumps(result, indent=2)}")
    
    # 5. 验证输出格式
    print("\nValidating ReAct format:")
    print("1. Response contains 'Thought:' section:", "Thought:" in react_response)
    print("2. Response contains 'Action:' section:", "Action:" in react_response)
    print("3. Action format is correct:", "Action Input:" in react_response)
    
    # 提取并验证action部分
    action_line = None
    for line in react_response.split('\n'):
        if line.strip().startswith('Action:'):
            action_line = line.strip()
            break
    
    if action_line:
        print("4. Action line format is valid:", action_line.startswith('Action:'))
        print("5. Tool name is valid:", action_line.split("Action:")[1].split("Action Input:")[0].strip() in ["web_search", "bash", "python_execute", "browser_use", "terminate"])
    
    # 6. 获取可用动作
    actions_response = requests.get(
        f"{base_url}/available_actions",
        params={"env_idx": env_idx}
    )
    print(f"Available actions response: {actions_response.text}")
    try:
        available_actions = actions_response.json()
    except:
        available_actions = {"response": actions_response.text}
    print(f"\nAvailable actions:\n{json.dumps(available_actions, indent=2)}")

if __name__ == "__main__":
    test_gaia_react_format() 