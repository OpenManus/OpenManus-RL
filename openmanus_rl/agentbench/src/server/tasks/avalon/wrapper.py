from copy import deepcopy
from typing import Dict, Union
from src.server.task import Session
from .utils import get_team_result, get_vote_result, get_assassination_result, get_believed_player_sides
from .prompts import CHECK_CHOOSE_TEAM_PROMPT, CHECK_VOTE_ON_QUEST_PROMPT, CHECK_VOTE_ON_TEAM_PROMPT, CHECK_ASSASSINATE_PROMPT, CHECK_BELIEVED_SIDES_PROMPT
from src.typings import SampleStatus
from src.typings import AgentContextLimitException
from .avalon_exception import AvalonAgentActionException

class FakeSession:
    history: list=[]    # Fake history

    async def action(self, input: Dict):
        # try:
        #     return input["naive_result"]
        # except:
        #     return "No naive results provided."
        pass

    def inject(self, input: Dict):
        pass

class SessionWrapper:
    def __init__(self, session: Union[Session, FakeSession]):
        self.session = session

    def balance_history(self):
        '''
            TODO: Make this function look better
        '''
        if len(self.session.history) % 2 != 0:
            self.inject({
                'role': 'user',
                'content': ''
            })

    def get_history(self):
        return self.session.history

    def overwrite_history(self, history: list):
        self.session.history = deepcopy(history)

    def inject(self, input: Dict):
        if isinstance(self.session, Session):
            self.session.inject({
                'role': input['role'],
                'content': input['content']
            })
        elif isinstance(self.session, FakeSession):
            pass

    async def action(self, input: Dict):
        if isinstance(self.session, Session):
            self.balance_history()
            self.session.inject({
                'role': input['role'],
                'content': input['content']
            })
            response = await self.session.action()

            if response.status == SampleStatus.AGENT_CONTEXT_LIMIT:
                raise AgentContextLimitException()
            if response.content is None:
                raise RuntimeError("Response content is None.")
            return response.content
        elif isinstance(self.session, FakeSession):
            return input.pop('naive_result', None)
        
    async def parse_result(self, input: Dict, result: str):
        print(result)
        mode = input['mode']
        past_history = deepcopy(self.session.history) # Store the history before the action
        print("Past history: ", past_history)
        self.session.history = [] # Clear the history
        if mode == "choose_quest_team_action":
            team_size = input['team_size']
            self.session.inject({
                "role": "user",
                "content": result + '\n\n' + CHECK_CHOOSE_TEAM_PROMPT
            })
            answer = await self.session.action()
            answer = answer.content
            answer = get_team_result(answer)
            if len(answer) != team_size:
                # Run another action to get the correct team size
                self.session.history = deepcopy(past_history)
                self.session.inject({
                    "role": "user",
                    "content": f"You should choose a team of size {team_size}, instead of size {len(answer)} as you did. Please output a list of player ids with the correct team size."
                })
                answer = await self.session.action()
                answer = answer.content
                past_history = deepcopy(self.session.history) # Update the history
                self.session.history = [] # Clear the history

                self.session.inject({
                    "role": "user",
                    "content": answer + '\n\n' + CHECK_CHOOSE_TEAM_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                try:
                    answer = get_team_result(answer)
                    assert len(answer) == team_size
                    assert isinstance(answer, list)
                except:
                    raise AvalonAgentActionException("Invalid team size with retry.")

        elif mode == "vote_on_team":
            self.session.inject({
                "role": "user",
                "content": result + '\n\n' + CHECK_VOTE_ON_TEAM_PROMPT
            })
            answer = await self.session.action()
            answer = answer.content
            answer = get_vote_result(answer)

            result_dict = {
                "No": 0,
                "Yes": 1
            }

            if answer not in ["No", "Yes"]:
                # Run another action to get the correct vote result
                self.session.history = deepcopy(past_history)
                self.session.inject({
                    "role": "user",
                    "content": f"You surely are a player in the game. Please output `Yes` or `No` to vote on the team."
                })
                answer = await self.session.action()
                answer = answer.content
                past_history = deepcopy(self.session.history) # Update the history
                self.session.history = [] # Clear the history

                self.session.inject({
                    "role": "user",
                    "content": answer + '\n\n' + CHECK_VOTE_ON_TEAM_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                answer = get_vote_result(answer)
            try:
                answer = result_dict[answer]
            except:
                raise AvalonAgentActionException("Invalid (team) vote result with retry.")

        elif mode == "vote_on_mission":
            self.session.inject({
                "role": "user",
                "content": result + '\n\n' + CHECK_VOTE_ON_QUEST_PROMPT
            })
            answer = await self.session.action()
            answer = answer.content
            answer = get_vote_result(answer)

            result_dict = {
                "No": 0,
                "Yes": 1
            }

            if answer not in ["No", "Yes"]:
                # Run another action to get the correct vote result
                self.session.history = deepcopy(past_history)
                self.session.inject({
                    "role": "user",
                    "content": "You surely are a player in the game, and you are a member in the quest. Please output `Yes` or `No` to vote on the quest."
                })
                answer = await self.session.action()
                answer = answer.content
                past_history = deepcopy(self.session.history) # Update the history
                self.session.history = [] # Clear the history

                self.session.inject({
                    "role": "user",
                    "content": answer + '\n\n' + CHECK_VOTE_ON_QUEST_PROMPT
                })
                answer = await self.session.action()
                answer = answer.content
                answer = get_vote_result(answer)
            try:
                answer = result_dict[answer]
            except:
                raise AvalonAgentActionException("Invalid (quest) vote result with retry.")

        elif mode == "assassination":
            self.session.inject({
                "role": "user",
                "content": result + '\n\n' + CHECK_ASSASSINATE_PROMPT
            })
            answer = await self.session.action()
            answer = answer.content
            answer = int(get_assassination_result(result, answer))
        elif mode == "get_believed_sides":
            self.session.inject({
                "role": "user",
                "content": result + '\n\n' + CHECK_BELIEVED_SIDES_PROMPT
            })
            answer = await self.session.action()
            answer = answer.content
            scores = get_believed_player_sides(answer)
            answer = []
            for i in range(5):
                answer.append(scores[i])

        # Restore the history
        self.session.history = deepcopy(past_history)
        print("Answer: ", answer)
        return answer