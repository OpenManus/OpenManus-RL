import json
import re
from typing import Callable, Dict, List, Any

from src.server.task import Task, Session
from src.typings import TaskOutput, SampleStatus, AgentOutputStatus
from .Interaction import Container

big_prompt = """
I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose to operate or to answer.
your operation should be like this:
Action: Operation
```sql
SELECT * FROM table WHERE condition;
```
You MUST put SQL in markdown format without any other comments. Your SQL should be in one line.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If you are done operating, and you want to commit your final answer, then write down:
Action: Answer
Final Answer: ["ANSWER1", "ANSWER2", ...]
DO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
Your input will be raw MySQL response, you have to deal with it by yourself.
"""


def build_init_sql(entry):
    name = entry["table"]["table_name"]
    columns = ",".join(
        [
            f"`{column['name']}` TEXT"
            for column in entry["table"]["table_info"]["columns"]
        ]
    )
    column_names = ",".join(
        [f"`{column['name']}`" for column in entry["table"]["table_info"]["columns"]]
    )
    items = []
    items_data = ()
    for row in entry["table"]["table_info"]["rows"]:
        item = "("
        for col in row:
            item += "%s,"
            items_data += (col,)
        item = item[:-1] + ")"
        items.append(item)
    items = ",".join(items)
    sql = f"""CREATE DATABASE IF NOT EXISTS `{name}`;
USE `{name}`;
CREATE TABLE IF NOT EXISTS `{name}` ({columns});
INSERT INTO `{name}` ({column_names}) VALUES {items}; 
COMMIT;
"""
    return sql, items_data


class DBBench(Task):
    def __init__(self, **configs):
        super().__init__(**configs)
        self.data_file = configs.pop("data_file")
        self.max_round = configs.pop("max_round", 5)
        self.dataset = []

        with open(self.data_file) as f:
            if self.data_file.endswith("json"):
                data = json.loads(f.read())
            else:
                data = [json.loads(line) for line in f.readlines()]

        for entry in data:
            if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
                ans = entry.pop("answer_md5")
            else:
                ans = entry.pop("label")
            inp = entry
            self.dataset.append((inp, ans))

        self.container = Container()

    def get_indices(self) -> List[Any]:
        return list(range(len(self.dataset)))

    async def start_sample(self, index: int, session: Session) -> TaskOutput:
        entry = self.dataset[index][0]
        container = self.container
        init_sql, init_data = build_init_sql(entry)
        container.execute(init_sql, data=init_data)
        db = entry["table"]["table_name"]
        session.inject({"role": "user", "content": big_prompt})
        session.inject({"role": "agent", "content": "Ok."})
        prompt = entry["description"] + "\n" + entry["add_description"]
        session.inject({"role": "user", "content": prompt})
        res = (await session.action()).content or ""
        answer = ""
        finish_reason = SampleStatus.COMPLETED
        try:
            action = re.search(r"Action: (.*?)\n", res)
            rounds = 0
            while action and action.group(1) == "Operation" and rounds < self.max_round:
                res = re.search(r"```sql\n([\s\S]*?)\n```", res)
                if not res:
                    finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
                    break
                sql = res.group(1).strip()
                sql = sql.replace("\n", " ")
                response = container.execute(sql, db)
                if response:
                    session.inject({"role": "user", "content": response})
                else:
                    session.inject({"role": "user", "content": ""})
                res = await session.action()
                if res.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                    finish_reason = SampleStatus.AGENT_CONTEXT_LIMIT
                    break
                res = res.content
                action = re.search(r"Action: (.*?)\n", res)
                rounds += 1
            else:
                answer = re.search(r"\nFinal Answer:(.*)", res)
                if answer:
                    answer = answer.group(1)
                else:
                    answer = ""
                    finish_reason = SampleStatus.AGENT_VALIDATION_FAILED
                if rounds >= self.max_round and not answer:
                    finish_reason = SampleStatus.TASK_LIMIT_REACHED
        except Exception as e:
            error = str(e)
            answer = ""
            finish_reason = SampleStatus.UNKNOWN
        else:
            error = ""
        if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
            columns = ",".join(
                [
                    f"`{column['name']}`"
                    for column in entry["table"]["table_info"]["columns"]
                ]
            )
            md5_query = (
                f"select md5(group_concat(rowhash order by rowhash)) as hash "
                f"from( SELECT substring(MD5(CONCAT_WS(',', {columns})), 1, 5) AS rowhash FROM `{db}`) as sub;"
            )
            answer = container.execute(md5_query, db)
        container.execute(f"drop database `{db}`")
        return TaskOutput(
            status=finish_reason,
            result={
                "answer": str(answer),
                "type": entry["type"][0],
                "error": error,
            },
            history=session.history,
        )

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        metrics = self.metrics
        ret = {}
        outputs = []
        answers = []
        for result in results:
            outputs.append(result.result)
            answers.append(self.dataset[result.index][1])
        for key, func in metrics.items():
            ret[key] = func(outputs, answers)
        return ret

    @property
    def metrics(self) -> Dict[str, Callable[[List[Dict[str, Any]], List[str]], float]]:
        def factory(typ):
            def acc(inp: List[Dict[str, Any]], tar: List[str]) -> float:
                correct = 0
                total = 0
                for entry, cor in zip(inp, tar):
                    if not entry:
                        continue
                    ans, t = entry["answer"], entry["type"]
                    if t != typ and not (
                        typ == "SELECT" and t not in ("INSERT", "UPDATE")
                    ):
                        continue
                    if t in ("INSERT", "DELETE", "UPDATE"):
                        correct += ans == cor
                    else:
                        try:
                            ans = list(eval(ans))
                        except:
                            ans = [ans]
                        if len(ans) == 1 and len(cor) == 1:
                            try:
                                correct += float(ans[0]) == float(cor[0])
                            except (ValueError, TypeError):
                                correct += ans[0] == cor[0]
                            else:
                                print(ans, cor)
                        else:
                            try:
                                cor = set(cor)
                                ans = set(ans)
                                correct += ans == cor
                            except:
                                pass
                    total += 1
                if total == 0:
                    print(f"WARNING: {typ} does not exist!")
                    return 0
                return correct / total

            return acc

        types = [
            "other",
            "counting",
            "comparison",
            "ranking",
            "aggregation-SUM",
            "aggregation-MIN",
            "aggregation-MAX",
            "aggregation-AVG",
            "SELECT",
            "INSERT",
            "UPDATE",
        ]

        ret = {}
        for typ in types:
            ret[typ + "_accuracy"] = factory(typ)

        ret["overall_cat_accuracy"] = (
            lambda inp, tar: sum(
                [
                    ret[typ + "_accuracy"](inp, tar)
                    for typ in ("SELECT", "INSERT", "UPDATE")
                ]
            )
            / 3
        )

        return ret

    def release(self):
        self.container.delete()
