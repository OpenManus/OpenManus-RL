import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from typing import List, Dict, Any
import asyncio

from src.configs import ConfigLoader
from src.server.task import Task, Session
from src.client.agent import AgentClient
from src.typings import (
    InstanceFactory,
    SampleIndex,
    SampleStatus,
    TaskSampleExecutionResult,
    AgentOutputStatus,
    TaskOutput,
)


class Prompter:
    def rules_solver(self):
        raise NotImplementedError

    def content_solver(self):
        raise NotImplementedError

    def rules(self):
        raise NotImplementedError

    def solver_instruction(self, story):
        raise NotImplementedError

    def hosting_instruction(self, story, answer, story_key, answer_key):
        raise NotImplementedError

    def check_solver_valid(self, solver):
        raise NotImplementedError

    def solver_hint_invalid(self):
        raise NotImplementedError

    def solver_hint_appeared_in_story(self):
        raise NotImplementedError

    def solver_hint_asked_before(self):
        raise NotImplementedError

    def hosting_append_messages(self, known, solver):
        raise NotImplementedError

    def hosting_summary(self, solver):
        raise NotImplementedError

    def hosting_summary_opposite(self, solver):
        raise NotImplementedError

    def hosting_simplify(self, reasoning):
        raise NotImplementedError

    def hosting_compare(self, key1, merge):
        raise NotImplementedError

    def check_yes(self, message: str):
        raise NotImplementedError

    def check_no(self, message: str):
        raise NotImplementedError


class ENPrompter(Prompter):
    def rules_solver(self):
        return (
            '1. At the beginning of the game, you will receive a narrative, referred to as "story". Based on the '
            'story, you need to ask questions that can be answered with "yes", "no", or "irrelevant" to guess out '
            'the "truth".\n2. By asking questions, you narrow down the range of possibilities until you '
            "eventually guess out the truth.\n3. Each time, you can only ask one question.\n4. Remember that your "
            "role is a player. You cannot declare the end of the game, give up on reasoning, or request a new "
            "game.\n5. You cannot directly repeat information already provided in the story.\n6. You cannot "
            'directly ask for details about the story in the form of "why" questions; you need to make your own '
            "guesses for truth.\n7. You cannot directly inquire about the story; you must make your own "
            "deductions."
        )

    def content_solver(self):
        return (
            f"You are a game player, and you are playing Lateral Thinking Puzzle, also known as Situation "
            f"Puzzle.\nLateral Thinking Puzzle is a deductive reasoning game, and here are the game "
            f"rules:\n{self.rules_solver()}\n\nNext, please make full use of the information provided above to "
            f'engage in game reasoning. Keep in mind that your questions should be answerable with "yes", "no", '
            f'or "irrelevant", and you can only ask one question at a time.'
        )

    def rules(self):
        return (
            '1. You know both the "story" and the "truth". When a user wants to play Lateral Thinking Puzzle, '
            'you provide them with the "story". The user only knows the "story" and is unaware of the '
            '"truth".\n2. The user asks questions that can be answered with "yes," "no," or "irrelevant". Their '
            'questions are aimed at guessing the "truth". Based on the "truth", you respond to the user\'s '
            'questions using "yes," "no," or "irrelevant" to guide them towards guessing the correct truth.\n3. '
            'If the user directly asks for details about the truth using the form of "why" questions, inform them '
            "that they need to make their own guesses.\n4. You must fully understand and accurately interpret the "
            "information from the truth. Based on the information of the truth and the user's past questions, "
            "you answer the user's questions. The user's questions may not necessarily contain information from "
            "the truth, but your responses must align with the facts of the truth.\n5. You can only answer "
            '"irrelevant" when the truth cannot provide a direct or indirect answer. Note that this is the only '
            'condition for responding "irrelevant"; otherwise, you should answer "yes" or "no."\n6. You cannot '
            "directly disclose the information from the truth to the user, even if they ask directly.\n7. You "
            "need to judge the user's questions as a whole and understand their overall intent. Avoid answering "
            "based solely on a particular point; your responses must align with the facts of the truth.\n8. "
            "During the user's process of guessing the truth, if they come close to some truths but still have "
            "gaps in understanding the complete truth of the truth, you can provide certain entry point hints. "
            "However, you cannot directly reveal information from the truth."
        )

    def solver_instruction(self, story):
        return (
            f"{self.content_solver()}\nHere is your story:\n{story}\n\nYou can start guessing the content of the "
            f"truth, and I will answer your questions. Please note that your questions should be answerable with "
            f'"yes", "no", or "irrelevant".'
        )

    def hosting_instruction(self, story, answer, story_key, answer_key):
        return [
            {
                "role": "user",
                "content": f"I need you to be the host of a game called Lateral Thinking Puzzle.\n\nLateral Thinking "
                f"Puzzle is a game consist of a story and a truth. Your story is: '{story}'\nYour t"
                f"ruth is: '{answer}'\n\nHere are the game rules:\n{self.rules()}\n\nDuring the game "
                f"process, please adhere to the above game rules to ensure a positive gaming experience "
                f"for the users. Pay close attention to the questions asked and ensure that your responses "
                f"comply with both the game rules and the information from the truth. When a user requests "
                f"to play the game, provide them with the story and help them guess the truth by answering "
                f'with "yes", "no", or "irrelevant". Remember that with each response, you must fully '
                f"understand and abide by the aforementioned game rules, as well as the story and the "
                f"truth. This will ensure a smooth user experience and avoid situations where you cannot "
                f"answer or violate the game rules.",
            },
            {
                "role": "assistant",
                "content": "Alright, I understand that my role is to be the host of the Lateral Thinking Puzzle and "
                "help users guess the truth by answering their questions. I have fully grasped all the "
                "information regarding the story and the truth and have carefully read all the rules. I "
                "assure that I will abide by all the rules throughout the game process.",
            },
            {
                "role": "user",
                "content": "Please summarize the key points of the story to ensure that you have understood it.",
            },
            {"role": "assistant", "content": story_key},
            {
                "role": "user",
                "content": "Please summarize the key points of the truth to ensure that you have understood it.",
            },
            {"role": "assistant", "content": answer_key},
            {
                "role": "user",
                "content": "Please restate the rules to ensure that you have understood all of them.",
            },
            {"role": "assistant", "content": self.rules()},
            {
                "role": "user",
                "content": "Alright, we can now start the game. Remember, before each response, you should review the "
                'key points of the story, the key points of the truth, and the rules. Answer with "yes", '
                '"no", or "irrelevant".',
            },
            {
                "role": "assistant",
                "content": f"Alright, as the host of the game, I will adhere to the above rules and ensure that my "
                f"responses comply with the rules and the information from the truth. Below is your story: "
                f"\n{story}\n\nYou can start guessing the content of the truth, and I will answer your "
                f'questions. Please note that your questions should be answerable with "yes", "no", '
                f'or "irrelevant".',
            },
        ]

    def check_solver_valid(self, solver):
        return not (
            "new story" in solver
            or "other story" in solver
            or "new game" in solver
            or "other game" in solver
            or "new question" in solver
            or "other question" in solver
            or "other round" in solver
            or "new round" in solver
            or "Great job" in solver
            or "further question" in solver
        )

    def solver_hint_invalid(self):
        return (
            "Please continue guessing the truth based on the story and ask a question that can be answered with "
            '"yes", "no", or "irrelevant".'
        )

    def solver_hint_appeared_in_story(self):
        return "This information is already in the story, please ask a new question."

    def solver_hint_asked_before(self):
        return (
            "This is a question that has already been asked. Please ask a new question."
        )

    def hosting_append_messages(self, known, solver):
        if len(known) > 0:
            return [
                {
                    "role": "user",
                    "content": f"{' '.join(known)}\nPlease answer with \"yes\", \"no\", or \"irrelevant\".",
                },
                {"role": "assistant", "content": "Yes."},
                {
                    "role": "user",
                    "content": f'{solver}\nPlease answer with "yes", "no", or "irrelevant".',
                },
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": f'{solver}\nPlease answer with "yes", "no", or "irrelevant".',
                }
            ]

    def hosting_summary(self, solver):
        return (
            f"Please restate the following content as a declarative sentence and simplify it into one sentence:"
            f"\n{solver}"
        )

    def hosting_summary_opposite(self, solver):
        return (
            f"Please restate the following content as a declarative sentence by using the opposite meaning and "
            f"then simplify it into one sentence:\n{solver}"
        )

    def hosting_simplify(self, reasoning):
        return f"Please simplify the following content into one sentence:\n{' '.join(reasoning)}"

    def hosting_compare(self, key1, merge):
        return (
            f"Please compare the information between Sentence 1 and Sentence 2 to determine if Sentence 2 contains "
            f'all the information in Sentence 1, including key details and descriptions. Please answer with "yes" '
            f'or "no".\nSentence 1: {key1}\nSentence 2: {merge}'
        )

    def check_yes(self, message: str):
        message = message.strip()
        return message.startswith("Yes") or message.startswith("yes")

    def check_no(self, message: str):
        message = message.strip()
        return message.startswith("No") or message.startswith("no")


class ZHPrompter(Prompter):
    def rules_solver(self):
        return (
            "1. 游戏的开始你会获得一段隐去关键信息的叙述，即汤面，你要根据汤面提出问题，问题需要能够通过“是”、“否”或“无关”来回答。\n2. 你通过提问缩小真相的范围，直到最终猜到真相（即汤底）。\n3. "
            "你每次只能提一个问题。\n4. 记住你的身份是玩家，不可以擅自宣布游戏结束，不可以放弃推理，也不可以要求出新的游戏题目。\n5. 你不能直接重复汤面中已有的信息。\n6. "
            "你不能直接用为什么的形式询问汤面的细节，需要你自己猜测。\n7. 你不能直接询问汤底，需要自己推测。"
        )

    def content_solver(self):
        return (
            f"你是一个游戏玩家，你在玩一个横向推理游戏，名叫情境猜谜，也叫海龟汤。\n海龟汤是一个推理类游戏，以下是游戏规则：\n{self.rules_solver()}\n\n"
            f"接下来，请充分利用以上信息，进行游戏推理，注意你的提问需要是能用“是”、“否”或“无关”回答的问题，每次只能提一个问题。"
        )

    def rules(self):
        return (
            "1. 你知道汤面和汤底，当用户需要玩海龟汤时，给予用户汤面，用户只知道汤面，不知道汤底。\n2. "
            "用户提出可以用“是”、“否”、“无关”来回答的问题，用户提问是为了猜测到汤底，你根据汤底回答用户的提问，只能用“是”、“否”或“无关”回答用户，从而引导用户猜到正确的汤底。\n3. "
            "如果用户直接以“为什么”的形式询问汤面的细节，请告知用户需要自己猜测。\n4. "
            "你要充分理解和准确解读汤底的信息，根据汤底的信息和用户过往提问信息对用户的提问做出回答，用户的提问不一定包含汤底的信息，但是你的回答必须符合汤底的事实。\n5. "
            "只有在汤底无法提供直接或间接的答案时，你才可以回答“无关”，注意这是回答“无关”的唯一条件，其他时候你要回答“是”或“否”。\n6. 你不能直接将汤底的信息告诉用户，就算用户直接问也不行。\n7. "
            "要整体判断用户的提问，理解用户整体的意思，不可片面通过某一个点作答，所答必须符合汤底事实。\n8. "
            "当用户在猜测汤底的过程中，猜到部分真相但与汤底的完整真相还有差距时，你可以提供一定的切入点提示，但不能直接透露汤底的信息。"
        )

    def solver_instruction(self, story):
        return f"{self.content_solver()}\n以下是你的汤面：\n{story}\n\n你可以开始猜测汤底的内容，我会回答你的问题。请注意，我只能回答“是”或“否”。"

    def hosting_instruction(self, story, answer, story_key, answer_key):
        return [
            {
                "role": "user",
                "content": f"我需要你做一个游戏的主持人，游戏名叫海龟汤。\n\n海龟汤游戏由汤面和汤底组成，你的汤面是："
                f"“{story}”\n你的汤底是：“{answer}”\n\n游戏规则：\n{self.rules()}\n\n游戏过程中，请你用以上游戏"
                f"规则约束你的行为，因为这样可以为用户带来良好的游戏体验。仔细审题，确保回答符合游戏规则和汤底的信息"
                f"，当用户发出游戏请求时，给出汤面，并通过回答“是”、“否”或“无关”帮助用户猜到汤底。注意每次回答你"
                f"都必须确保自己充分理解并遵守了以上游戏规则和你的汤面及汤底，保证用户体验，不要出现无法回答的情况，"
                f"更不要违反游戏规则。",
            },
            {
                "role": "assistant",
                "content": "好的，我明白自己的角色是海龟汤游戏的主持人，要通过回答用户的问题帮助用户猜到汤底。我已充分理"
                "解汤面和汤底的所有信息，并认真阅读了所有规则，保证自己会在游戏过程中遵守所有规则。",
            },
            {"role": "user", "content": "请总结汤面的关键点，以确保你理解了汤面。"},
            {"role": "assistant", "content": story_key},
            {"role": "user", "content": "请总结汤底的关键点，以确保你理解了汤底。"},
            {"role": "assistant", "content": answer_key},
            {"role": "user", "content": "请复述一遍规则，以确保你理解了所有规则。"},
            {"role": "assistant", "content": self.rules()},
            {
                "role": "user",
                "content": "好的，我们现在可以开始游戏了。记住，每次回答前你要回顾汤面的关键点、汤底的关键点、规则，回答是或否或无关。",
            },
            {
                "role": "assistant",
                "content": f"好的，作为游戏的主持人，我将遵守以上规则，并确保回答符合规则和汤底的信息。下面是你的汤面："
                f"\n{story}\n\n你可以开始猜测汤底的内容，我会回答你的问题。请注意，你的问题需要能够用“是”"
                f"、“否”或“无关”来回答。",
            },
        ]

    def check_solver_valid(self, solver):
        return not ("感谢" in solver or "谢谢" in solver or "再见" in solver)

    def solver_hint_invalid(self):
        return "请你继续根据汤面猜测汤底的信息，提出一个可以用“是”或“否”或“无关”回答的问题。"

    def solver_hint_appeared_in_story(self):
        return "这是汤面中已有的信息，请提一个新问题。"

    def solver_hint_asked_before(self):
        return "这是已经提问过的内容，请提一个新问题。"

    def hosting_append_messages(self, known, solver):
        if len(known) > 0:
            return [
                {
                    "role": "user",
                    "content": f"{' '.join(known)}\n请回答是或否或无关。",
                },
                {"role": "assistant", "content": "是。"},
                {
                    "role": "user",
                    "content": f"{solver}\n请回答是或否或无关。",
                },
            ]
        else:
            return [
                {
                    "role": "user",
                    "content": f"{solver}\n请回答是或否或无关。",
                }
            ]

    def hosting_summary(self, solver):
        return f"请将以下内容改写成陈述句，并简化成一句话：\n{solver}"

    def hosting_summary_opposite(self, solver):
        return f"请将以下内容取反义然后改写成陈述句，并简化成一句话：\n{solver}"

    def hosting_simplify(self, reasoning):
        return f"请将以下内容简化成一句话：\n{' '.join(reasoning)}"

    def hosting_compare(self, key1, merge):
        return f"请对比第一句话和第二句话之间的信息，判断第二句话是否完整地概括了第一句话的全部信息，包括关键细节和描述。请用是或否回答。\n第一句话：{key1}\n第二句话：{merge}"

    def check_yes(self, message: str):
        message = message.strip()
        return message.startswith("是")

    def check_no(self, message: str):
        message = message.strip()
        return message.startswith("不是") or message.startswith("否")


class LateralThinkingPuzzle(Task):
    def __init__(self, rounds=25, filepath=None, eval_yaml=None, **configs):
        # TODO: refactor
        # change into list, which contains dict with problems and language
        self.rounds = rounds
        self.filepath = filepath
        table = pd.read_excel(self.filepath)
        self.inputs = []
        self.targets = []
        for i in range(table.values.shape[0]):
            story = table.values[i, 0].strip()
            answer = table.values[i, 1].strip()
            story_key = table.values[i, 2].strip()
            answer_key = table.values[i, 3].strip()
            self.inputs.append((story, answer, story_key, answer_key))
            self.targets.append(answer)
        self.eval_yaml = eval_yaml
        self.eval_agent: AgentClient = InstanceFactory.parse_obj(
            ConfigLoader().load_from(self.eval_yaml)["gpt-3.5-turbo-0613"]
        ).create()
        super().__init__(**configs)

    def get_indices(self) -> List[SampleIndex]:
        return list(range(len(self.inputs)))

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        outputs = [result.result for result in results]
        # adapt from self.metrics
        return {
            "main": sum([output["progress"] for output in outputs if output])
            / len(outputs),
            "SGA": sum([output["accuracy"] for output in outputs if output])
            / len(outputs),
            "RE": sum([output["efficiency"] for output in outputs if output])
            / len(outputs),
            "QR": sum([output["relevance"] for output in outputs if output])
            / len(outputs),
            "GP": sum([output["progress"] for output in outputs if output])
            / len(outputs),
        }

    @property
    def metrics(self):
        return {
            "main": lambda outputs, targets: sum(
                [output["progress"] for output in outputs if output]
            )
            / len([output["accuracy"] for output in outputs if output]),
            "SGA": lambda outputs, targets: sum(
                [output["accuracy"] for output in outputs if output]
            )
            / len([output["accuracy"] for output in outputs if output]),
            "RE": lambda outputs, targets: sum(
                [output["efficiency"] for output in outputs if output]
            )
            / len([output["accuracy"] for output in outputs if output]),
            "QR": lambda outputs, targets: sum(
                [output["relevance"] for output in outputs if output]
            )
            / len([output["accuracy"] for output in outputs if output]),
            "GP": lambda outputs, targets: sum(
                [output["progress"] for output in outputs if output]
            )
            / len([output["accuracy"] for output in outputs if output]),
        }

    async def start_sample(
        self, index: SampleIndex, session: Session
    ) -> TaskSampleExecutionResult:  # return OUTPUT object, need to be json serializable
        data_item = self.inputs[index]
        story, answer, story_key, answer_key = (
            data_item[0],
            data_item[1],
            data_item[2],
            data_item[3],
        )
        prompter = ENPrompter()  # TODO: change this
        host = prompter.solver_instruction(story)
        results = {}
        correct = 0
        relevance = 0
        finished = False
        logs = [f"Story: {story}", f"Truth: {answer}"]
        hit_keys = {}
        hosting = prompter.hosting_instruction(story, answer, story_key, answer_key)
        story_key = story_key.split("\n")
        answer_key = answer_key.split("\n")
        hints = len(answer_key)
        history = []
        known = []
        reasoning = []
        bingo = 0
        abort = 0
        finish_round = self.rounds
        finish_reason = SampleStatus.COMPLETED
        for i in range(self.rounds):
            solver = await session.action({"role": "user", "content": host})
            if solver.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                finish_reason = SampleStatus.AGENT_CONTEXT_LIMIT
                break
            elif solver.status != AgentOutputStatus.NORMAL:
                finish_reason = SampleStatus.UNKNOWN
                break
            if not solver.content:
                break
            solver = solver.content
            solver = solver.split("\n")[0].strip()
            logs.append(f"Round {i+1}: {solver}")
            if not prompter.check_solver_valid(solver):
                abort += 2
                host = prompter.solver_hint_invalid()
                logs.append(host)
                if abort >= 5:
                    break
                continue

            llm = True
            for key in story_key:
                key = key.strip()
                if key == "":
                    continue
                if key[1] == "." or key[2] == "." or key[0] == "-":
                    key = " ".join(key.split(" ")[1:])
                bleu = sentence_bleu(
                    [key], solver.replace("?", "."), weights=(1, 0, 0, 0)
                )
                if bleu > 0.85:
                    host = prompter.solver_hint_appeared_in_story()
                    llm = False
                    break
            if llm:
                for key in history:
                    key = key.strip()
                    if key == "":
                        continue
                    bleu = sentence_bleu(
                        [key], solver.replace("?", "."), weights=(1, 0, 0, 0)
                    )
                    if bleu > 1:
                        host = prompter.solver_hint_asked_before()
                        llm = False
                        abort += 1
                        break
            if abort >= 5:
                break

            if llm:
                history.append(solver.replace("?", "."))
                hosting = hosting[:10]
                hosting += prompter.hosting_append_messages(known, solver)
                for item in hosting:
                    if item["role"] == "assistant":
                        item["role"] = "agent"
                host = (await asyncio.to_thread(
                    self.eval_agent.inference,
                    hosting,
                )).strip()
                relevant = False
                if prompter.check_yes(host):
                    correct += 1
                    summary = (await asyncio.to_thread(
                        self.eval_agent.inference,
                        [
                            {
                                "role": "user",
                                "content": prompter.hosting_summary(solver),
                            }
                        ],
                    )).strip()
                    relevant = True
                elif prompter.check_no(host):
                    summary = (await asyncio.to_thread(
                        self.eval_agent.inference,
                        [
                            {
                                "role": "user",
                                "content": prompter.hosting_summary_opposite(solver),
                            }
                        ]
                    )).strip()
                    relevant = True
                if relevant:
                    relevance += 1
                    history.append(summary)
                    known.append(summary)
                    reasoning.append(summary)
                    if len(reasoning) >= 2:
                        merge = (await asyncio.to_thread(
                            self.eval_agent.inference,
                            [
                                {
                                    "role": "user",
                                    "content": prompter.hosting_simplify(reasoning),
                                }
                            ],
                        )).strip()
                    else:
                        merge = summary
                    for key in answer_key:
                        key = key.strip()
                        if key == "":
                            continue
                        if key[1] == "." or key[2] == "." or key[0] == "-":
                            key1 = " ".join(key.split(" ")[1:])
                        else:
                            key1 = key
                        if len(solver.replace("?", "")) < len(key1):
                            continue
                        comp_msg = [
                            {
                                "role": "user",
                                "content": prompter.hosting_compare(key1, merge),
                            }
                        ]
                        compare = (await asyncio.to_thread(
                            self.eval_agent.inference,
                            comp_msg,
                        )).strip()
                        if prompter.check_yes(compare):
                            bingo += 1
                            answer_key.remove(key)
                            reasoning = []
                            hit_keys[key1] = merge
                            break
                    if bingo >= len(answer_key) and not finished:
                        finished = True
                        finish_round = i + 1
            logs.append(host)
        else:
            finish_reason = SampleStatus.TASK_LIMIT_REACHED
        results["accuracy"] = correct / self.rounds
        results["efficiency"] = 1 - finish_round / self.rounds
        results["relevance"] = relevance / self.rounds
        results["progress"] = bingo / hints
        results["finish_round"] = finish_round
        results["finished"] = finished
        results["hit_keys"] = hit_keys
        results["history"] = logs
        return TaskSampleExecutionResult(
            status=finish_reason,
            result=results,
        )
