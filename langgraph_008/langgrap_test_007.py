import json
import operator
from typing import List, Dict, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage

from common.model import mimo_openai


# ------------------------------------------------------------
# 定义状态结构
# ------------------------------------------------------------
class JokeState(TypedDict):
    topic: str  # 原始主题
    sub_topics: List[str]  # 生成的子主题列表
    jokes: Annotated[List[Dict], operator.add]  # 收集的笑话（逐步添加）
    best_jokes: List[Dict]  # 最终选出的2个笑话


# ------------------------------------------------------------
# 初始化模型
model = mimo_openai


# 辅助函数：安全解析JSON（去除markdown标记）
def safe_parse_json(text: str):
    import re
    text = re.sub(r"```json\s*|\s*```", "", text.strip())
    return json.loads(text)


# ------------------------------------------------------------
# 节点1：生成子主题
# ------------------------------------------------------------
PROMPT_SUB = """
主题：{topic}
只返回6个子主题的JSON数组，不要任何其他文字：
["主题1","主题2",...]
"""


def generate_subtopics(state: JokeState) -> dict:
    prompt = PromptTemplate(template=PROMPT_SUB, input_variables=["topic"])
    chain = prompt | model | JsonOutputParser()
    sub_topics = chain.invoke({"topic": state["topic"]})
    return {"sub_topics": sub_topics}


# ------------------------------------------------------------
# 节点2：为单个子主题生成6个笑话（可被并行调用）
# ------------------------------------------------------------
PROMPT_JOKE = """
子主题：{sub_tp}
只返回6个笑话的JSON数组，每个元素格式为 {{"sub_tp": "{sub_tp}", "joke": "..."}}，不要任何其他文字。
"""


def generate_jokes_for_subtopic(sub_tp: str) -> List[Dict]:
    prompt = PromptTemplate(template=PROMPT_JOKE, input_variables=["sub_tp"])
    chain = prompt | model | JsonOutputParser()
    jokes = chain.invoke({"sub_tp": sub_tp})
    # 确保每个笑话都包含正确的 sub_tp（有些模型可能忽略，我们强制覆盖）
    for j in jokes:
        j["sub_tp"] = sub_tp
    return jokes


def generate_all_jokes(state: JokeState) -> dict:
    """并行调用每个子主题的笑话生成，结果收集到 jokes 字段中"""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(generate_jokes_for_subtopic, state["sub_topics"]))
    # 扁平化列表
    all_jokes = [joke for jokes_list in results for joke in jokes_list]
    return {"jokes": all_jokes}


# ------------------------------------------------------------
# 节点3：选出最好笑的2个笑话
# ------------------------------------------------------------
PROMPT_BEST = """
笑话列表：
{jokes_json}
只选最好笑的2个，返回JSON数组，每个元素格式为 {{"sub_tp": "...", "joke": "..."}}，不要任何其他文字。
"""


def select_best_jokes(state: JokeState) -> dict:
    print("正在处理：", state["jokes"])
    jokes_json = json.dumps(state["jokes"], ensure_ascii=False, indent=2)
    prompt = PromptTemplate(template=PROMPT_BEST, input_variables=["jokes_json"])
    chain = prompt | model | JsonOutputParser()
    best = chain.invoke({"jokes_json": jokes_json})
    return {"best_jokes": best}


# ------------------------------------------------------------
# 构建 LangGraph 图
# ------------------------------------------------------------
builder = StateGraph(JokeState)

# 添加节点
builder.add_node("generate_subtopics", generate_subtopics)
builder.add_node("generate_jokes", generate_all_jokes)  # 内部并行
builder.add_node("select_best", select_best_jokes)

# 添加边
builder.set_entry_point("generate_subtopics")
builder.add_edge("generate_subtopics", "generate_jokes")
builder.add_edge("generate_jokes", "select_best")
builder.add_edge("select_best", END)

# 编译
graph = builder.compile()

# ------------------------------------------------------------
# 运行示例
# ------------------------------------------------------------
if __name__ == "__main__":
    initial_state = {"topic": "动物"}
    final_state = graph.invoke(initial_state)

    print("=== 子主题 ===")
    for sub in final_state["sub_topics"]:
        print(f"  {sub}")

    print("\n=== 最佳笑话（前2） ===")
    for joke in final_state["best_jokes"]:
        print(f"主题：{joke['sub_tp']}")
        print(f"笑话：{joke['joke']}\n")
