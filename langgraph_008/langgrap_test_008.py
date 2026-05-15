import json
import operator
from typing import List, Annotated, TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from common.model import mimo_openai

# ======================
# 提示词（修复成标准 JSON 格式）
# ======================
subject_prompt = """生成2-5个与{topic}相关的子主题。
严格返回JSON格式：
{{"subjects": ["例子1","例子2"]}}
"""

joke_prompt = """生成一个关于{subject}的笑话。
严格返回JSON格式：
{{"joke": "笑话内容"}}
"""

best_joke_prompt = """以下是关于{topic}的笑话列表：
{jokes}

请选出最好笑的一个，只返回它的索引id（从0开始）。
严格返回JSON格式：
{{"id": 0}}
"""


# ======================
# 结构化模型
# ======================
class Subject(BaseModel):
    subjects: list[str]


class Joke(BaseModel):
    joke: str


class BestJoke(BaseModel):
    id: int = Field(description="最佳笑话的索引，从0开始！", ge=0)


# ======================
# 状态定义
# ======================
class OverAllState(TypedDict):
    topic: str
    subject: list
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


class JokeState(TypedDict):
    subject: str


# ======================
# 节点 1：生成子主题
# ======================
def generate_subject(state: OverAllState):
    main_topic = state["topic"]
    resp = mimo_openai.with_structured_output(Subject).invoke(subject_prompt.format(topic=main_topic))
    print("=" * 60)
    print(f"📦 主主题：{main_topic}")
    print(f"✅ 生成子主题：{resp.subjects}")
    print("=" * 60)
    return {"subject": resp.subjects}


# ======================
# 节点 2：并行生成笑话
# ======================
def generate_joke(state: JokeState):
    sub_topic = state["subject"]
    resp = mimo_openai.with_structured_output(Joke).invoke(joke_prompt.format(subject=sub_topic))
    print(f"😂 子主题：{sub_topic}")
    print(f"   笑话：{resp.joke}")
    print("-" * 60)
    return {"jokes": [resp.joke]}


# ======================
# 并行分发（Send 核心）
# ======================
def continue_to_joke(state: OverAllState):
    return [Send("generate_joke", {"subject": s}) for s in state["subject"]]


# ======================
# 节点 3：选择最佳笑话
# ======================
def select_best_joke(state: OverAllState):
    allJokes = state["jokes"]
    topic = state["topic"]
    jokes_str = "\n".join([f"{i}: {j}" for i, j in enumerate(allJokes)])

    resp = mimo_openai.with_structured_output(BestJoke).invoke(
        best_joke_prompt.format(topic=topic, jokes=jokes_str)
    )

    best_id = resp.id
    best_joke = allJokes[best_id]

    print("=" * 60)
    print(f"🏆 最好笑的笑话（ID={best_id}）：")
    print(f"👉 {best_joke}")
    print("=" * 60)

    return {"best_selected_joke": best_joke}


# ======================
# 构建流程图
# ======================
gp = StateGraph(OverAllState)
gp.add_node("generate_subject", generate_subject)
gp.add_node("generate_joke", generate_joke)
gp.add_node("select_best_joke", select_best_joke)

gp.add_edge(START, "generate_subject")
gp.add_conditional_edges("generate_subject", continue_to_joke)
gp.add_edge("generate_joke", "select_best_joke")
gp.add_edge("select_best_joke", END)

# ======================
# 运行 + 美化最终输出
# ======================
gc = gp.compile()
resp = gc.invoke({"topic": "动物"})

print("\n🎉【最终结果】")
print(f"主题：{resp['topic']}")
print(f"最佳笑话：{resp['best_selected_joke']}")