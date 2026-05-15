from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from common.model import mimo_openai

# 记忆存储器
mem = MemorySaver()


# 状态：保存历史对话 + 当前输入
class DemoState(TypedDict):
    history: str  # 对话历史（核心！记忆靠它）
    input: str  # 当前输入
    resp: str  # 模型回答


# 调用模型 + 拼接历史
def call_model(state: DemoState, ):
    history = state.get("history", "")
    user_input = state["input"]

    # 拼接上下文提示词
    prompt = f"""
            历史对话：
            {history}
            
            当前用户说：{user_input}
            请自然回答。
            """

    # 调用模型
    resp = mimo_openai.invoke(prompt)

    print("=" * 50)
    print(f"🧑 用户：{user_input}")
    print(f"🤖 助手：{resp.content}")
    print("=" * 50)

    # 更新对话历史
    new_history = f"{history}\n用户：{user_input}\n助手：{resp.content}"
    return {"resp": resp.content, "history": new_history}


# 构建图
gp = StateGraph(DemoState)
gp.add_node("call_model", call_model)

gp.add_edge(START, "call_model")
gp.add_edge("call_model", END)

# 编译（开启记忆）
gc = gp.compile(checkpointer=mem)

# 同一个线程 ID = 同一个对话
config = {"configurable": {"thread_id": "demo_1"}}

# 第一次对话
gc.invoke({"input": "你好我是 tom"}, config=config)
# 第二次对话（能记住你是 tom）
gc.invoke({"input": "你好我是 lili"}, config={"configurable": {"thread_id": "demo_2"}})
gc.invoke({"input": "你知道我是谁吗？"}, config=config)
gc.invoke({"input": "你知道我是谁吗？"}, config={"configurable": {"thread_id": "demo_2"}})

# 每次对话都会获取到这个memory数据 按照线程德key进行分类
# 打印memory中德数据
print(mem.get(config=config))
print(mem.get(config={"configurable": {"thread_id": "demo_2"}}))