from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


# 1. 定义状态（正确）
class State(TypedDict):
    messages: list[AnyMessage]
    other: str


# 2. 节点函数（正确）
def nodeFunc(state: State):
    messages = state["messages"]
    newMessages = AIMessage(content="你好我是nodeFunc!")

    return {
        "messages": messages + [newMessages],
        "other": "这是一个新的节点数据",
    }


# 3. 构建流程图（重点修复）
gp = StateGraph(State)

# 🔥 修复1：添加节点必须指定【节点名 + 节点函数】
gp.add_node("node", nodeFunc)

# 入口点
gp.set_entry_point("node")

# 编译
gb = gp.compile()

# 4. 调用（修复2：消息必须用消息对象，不能用元组！）
resp = gb.invoke({
    "messages": [
        HumanMessage(content="你好")  # ✅ 必须用 HumanMessage
    ]
})

# 输出
print("✅ 最终结果：")
print(resp)

for me in resp["messages"]:
    me.pretty_print()
