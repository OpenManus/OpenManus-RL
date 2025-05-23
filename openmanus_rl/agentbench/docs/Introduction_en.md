# Detailed Explanation of AgentBench

[🌏中文版](Introduction_cn.md)

## 1. Dataset Composition

The Benchmark comprises eight tasks:

- Operating System
- Database
- Knowledge Graph
- Digital Card Game
- Lateral Thinking Puzzle
- Householding (ALFWorld)
- Web Shopping (WebShop)
- Web Browsing (Mind2Web)

### 1.1 Operating System (OS)

Allowing LLMs to access and manipulate OS in the terminal is a fascinating
but challenging mission. Despite attempts on translating natural language to Shell commands (Lin
et al., 2018), few prior efforts evaluate models in executable environments. We aim to evaluate LLMs
in genuine OS’ interactive bash environments (i.e., Ubuntu Docker (Merkel et al., 2014)) on human
questions with deterministic answers (e.g., number of users with non-/home directories in an OS.) or
series of operations for practical goals (e.g., recursively set all directory files to read-only, excluding
mine). We adopt the success rate (SR) as the evaluation metric.

### 1.2 Database (DB)

As database analysis is crucial but also difficult in many daily affairs, it is paramount
to examine LLMs’ abilities to operate on real databases via SQL. Prior research has a significant
emphasis on individual procedures, such as translation between SQL and natural language (Zhong
et al., 2017), or answering questions given individual small tables (Nan et al., 2021; Iyyer et al.,
2017). However, few consider evaluating models on the complete pipeline as a whole. Therefore,
AGENTBENCH evaluates LLMs on authentic SQL interfaces, databases, multiple tables, and different
types of queries as is in the real world. We adopt the SR as the main evaluation metric.

### 1.3 Knowledge Graph (KG)

Engaging with contemporary KGs, which are often
vast in size (e.g., FREEBASE (Bollacker et al., 2008) has over 45M entities and 3B facts), demands a
broad range of skills from an intelligent agent (Gu et al., 2023). Operating in such environments, which
are only partially observable, requires the agent to make decisions with incomplete information and
manage inherent uncertainties with various skills, including language understanding (e.g., intricacies
and subtleties), planning (e.g., breaking down instructions into more manageable components), and
tool using (e.g., interact with KG interfaces). As a result, we propose KG as a representative testing
ground to assess the decision-making abilities of AI agents. We adopt question answering as the basic
task formulation and consequently the answer F1 as the metric.

### 1.4 Digital Card Game (DCG)

Games, especially those that require strategies and planning, could
serve as simulated environments for intelligent agent development. DCG (e.g., Hearthstone (Hoover
et al., 2020)), instead, is an ideal option for text-only LLM evaluation. It usually involves abundant
text descriptions for cards, turn-based competition, and thoughtful playing strategies to win, testing a
model’s understanding of game rules, operating logic, and abilities to form strategic decisions based
on current conditions and past experiences in the game.
In AGENTBENCH we adapt a simplified DCG system—Aquawar1—from the 2021 Tsinghua University Agent Competition (THUAC)
hosted by Student Association for Science and Technology in
Department of Computer Science and Technology (CST-SAST), for evaluating LLM-as-Agent. In
Aquawar, the agent acts as a player managing a team of fishes with different talents to battle against
another team (controlled by our ad-hoc baseline agent) in a turn-based form. We report LLMs’ win
rate as the evaluation metric.

### 1.5 Lateral Thinking Puzzles (LTP)

Lateral thinking puzzles (Sloane, 1992), or situation puzzles, 海
龟汤, is a popular group-playing game around the world. The game usually has a person hosting the
puzzle and others guess by asking riddle-related questions. The host can only respond “yes”, “no”, or
“irrelevant”. The game is terminated when one of the player recovers the critical plots of the puzzle.
Its name derives from the psychological term “lateral thinking” (De Bono, 1970), which refers to the
ability of deducing facts from unconventional perspectives and exploring new ideas.

In this dataset, we first set up an LTP host system for automatic judging. To assess
LLMs’ lateral reasoning prowess, a diverse puzzle dataset is curated from web of varied levels of
difficulty. We break down the true plot into several bullets and measure the portion of guessed-out
bullets (i.e., game progress) when an agent exhausted the maximum number of playing rounds as
the evaluation metric. Through this assessment, we aim to gain insights into the depth and agility of
LLMs’ lateral reasoning abilities.

### 1.6 House-Holding (HH, ALFWorld)

Embodied game environments such
as house-holding, which require strong commonsense grounding, have been well-established for
language agent evaluation (Côté et al., 2019). In AGENTBENCH, we assess the model’s capability in
accomplishing tasks in physical house-holding environments on the classical ALFWorld (Shridhar
et al., 2020b) derived from the well-established text-game toolkit TextWorld (Côté et al., 2019). The
agent needs to accomplish house-holding tasks such as “Put a pan on the dining table”. We adopt the
SR as the evaluation metric.

### 1.7 Web Shopping (WS, WebShop)

Online shopping is a very practical and important
part of modern life. Its trajectory, which comprises searching, viewing, and choosing desirable items
on a real e-commerce website, requires autonomous agents’ strong reasoning and decision-making
abilities. Webshop (Yao et al., 2022), a simulated online shopping environment, exactly serves such
a purpose for evaluating language agents. While it is originally evaluated on specifically trained
models, we propose assessing LLMs with mere prompting.

### 1.8 Web Browsing (WB, Mind2Web)

. General web environment is an ideal sandbox
for training and evaluating intelligent agents. Mind2Web (Deng et al., 2023) is a very recently
released general benchmark for developing and assessing web agents capable of executing intricate
tasks across various website domains, given high-level user instructions. It designs feasible actions for
website interactions, such as clicking, selecting, and typing, thereby facilitating a holistic evaluation
of LLMs as web agents. Compared to Mind2Web’s original setting, we make adaptations to allow its
evaluation on prompted LLMs without additional fine-tuning.

## 2. Framework Introduction

The framework is designed to decouple its various components as much as possible, allowing for independent development,
testing, and deployment of each part. This approach is mainly due to the varied system resource and environment
requirements of different tasks, making a unified design challenging. This setup also facilitates subsequent extensions
and maintenance. To enhance usability, configuration files accompany each part of the framework, which users can modify
as needed. The components can be deployed on separate machines or a single machine, communicating via the HTTP protocol.

![Overall Architecture](../assets/architecture.png)

As illustrated, the entire framework comprises three parts. The first is the Task Server, whose primary purpose, as the
name suggests, is to host a task environment. It provides a task description and offers environmental feedback based on
the Agent's response. The second part is the Agent Server, which offers an interface for an Agent that can infer from
historical data. The third is the Client, which coordinates tasks according to configuration file requirements and
forwards outputs between the Agent and Task.

For instance, to test ChatGLM2-6B's performance on the WebShop and DBBench tasks:

1. Deploy the ChatGLM2-6B model using FastChat to get an Agent Server.
2. Modify the Task Server configuration file to deploy the WebShop and DBBench task environments separately.
3. Edit the Client configuration file, specify the use of FastchatClient, and indicate testing for WebShop and DBBench.
   Then start.

### 2.1 Agent Server

The design of the Agent Server allows for servers of any form. Fastchat currently serves as the Server for local models.
For models that only offer an API, their corresponding interface can be directly implemented within the Agent Client.

### 2.2 Introduction to Task Server

The Task Server mainly consists of two components: the Task Worker and the Task Controller.

The Task Controller oversees all Task Workers and presents a unified interface to the Client. There should only be one
Task Controller globally. It is mainly responsible for:

- Awaiting connections from Task Workers and receiving their registration details.
- Assigning tasks to idle Task Workers upon receiving Client requests.
- Handling subsequent Client requests and forwarding them to the relevant Task Worker.

The two primary interfaces to note are:

- `POST /api/start_sample` This interface initiates a new test case, returning a `session_id` for identification. Task
  allocation to the Task Worker occurs here. The returned content also includes the task description, or the initial
  Prompt.
- `POST /api/interact` This interface facilitates interaction between the Agent and the Task, receiving the Agent's
  output and forwarding it to the associated Task Worker, and then returning the output from the Task Worker (i.e., task
  environment).

Each Task Worker handles a specific task environment. Based on the configuration file's requirements, Task Workers can
autonomously start and load environments. If you want to implement a new task environment, you only need to inherit and
implement the Task class, then specify it in the configuration file. If you wish to run multiple concurrent tasks of the
same type, it's recommended to launch multiple Task Workers. This approach ensures non-interference between environments
and can bypass the `GIL`. However, if a task isn't CPU-intensive or if there are many shared resources between multiple
concurrent tasks, the framework allows the Task to specify the maximum concurrency for a single Worker.

### 2.3 Client

The Client primarily comprises three components:

- The Assigner is responsible for coordinating the concurrent tasks and models currently available, planning and
  allocating test cases, and creating the corresponding number of Workers along with their Agent Client and Task Client.
- The Agent Client implements the corresponding interface required by the Agent Server, exposing the
  unified `AgentClient.inference(self, history)`.
- The Task Client interfaces exclusively with the Task Controller, making its implementation unique. Its core method
  is `TaskClient.run_sample(self, index, agent)`, which ensures the passed `Agent` and `Task` outputs are forwarded to
  each other.

The Assigner reads the relevant configuration file, resulting in a bipartite graph from Agent to Task. Based on this
graph, an s-node and t-node are constructed. The capacity of edges from s to the Agent represents the Agent's
concurrency, while the capacity of edges from Task to t represents the Task's concurrency. The edge capacity between
Agents and Tasks signifies the number of test cases awaiting evaluation. The Assigner's core allocation logic operates
in real-time on this graph, using a maximum flow algorithm. Whenever an Agent or Task becomes available, the algorithm
runs to produce a flow graph. Based on the flow on edges from Agent to Task, the corresponding number of workers are
initiated and allocated to specific test cases. Each worker is responsible for a single test case and possesses an Agent
Client object and a Task Client object.
