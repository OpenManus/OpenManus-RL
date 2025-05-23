import argparse

from src.configs import ConfigLoader
from src.typings import InstanceFactory
from .agent import AgentClient


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/agents/api_agents.yaml')
    parser.add_argument('--agent', type=str, default='gpt-3.5-turbo-0613')
    return parser.parse_args()


def interaction(agent: AgentClient):
    try:
        history = []
        while True:
            print("================= USER  ===================")
            user = input(">>> ")
            history.append({"role": "user", "content": user})
            try:
                agent_response = agent.inference(history)
                print("================ AGENT ====================")
                print(agent_response)
                history.append({"role": "assistant", "content": agent_response})
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                exit(0)
    except KeyboardInterrupt:
        print("\n[Exit] KeyboardInterrupt")
        exit(0)


if __name__ == '__main__':
    args = parse_args()
    loader = ConfigLoader()
    config = loader.load_from(args.config)
    assert args.agent in config, f"Agent {args.agent} not found in {args.config}"
    agent_config = config[args.agent]
    factory = InstanceFactory(**agent_config)
    agent_client: AgentClient = factory.create()
    interaction(agent_client)
