# 集中流的方式输出
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class TypeState(TypedDict):
    step1: str
    step2: str
    step3: str
    messages: list[BaseMessage]


def step1(state: TypeState):
    print("✅ 执行步骤 1：开始")
    new_msg = AIMessage(content="我已收到你的问题，开始处理...")
    return {"step1": "步骤1完成","messages": [new_msg]}


def step2(state: TypeState):
    print("✅ 执行步骤 2：处理中")
    new_msg = AIMessage(content="我已收到你的问题，开始处理...")
    return {"step2": "步骤2完成","messages": [new_msg]}


def step3(state: TypeState):
    print("✅ 执行步骤 3：完成")
    new_msg = AIMessage(content="我已收到你的问题，开始处理...")
    return {"step3": "步骤3完成","messages": [new_msg]}


gp = StateGraph(TypeState)
gp.add_node("step1", step1)
gp.add_node("step2", step2)
gp.add_node("step3", step3)
gp.add_edge(START, "step1")
gp.add_edge("step1", "step2")
gp.add_edge("step2", "step3")
gp.add_edge("step3", END)
gc = gp.compile()

# =============================================================================
# 1️⃣ updates：只输出【本次更新的字段】
# =============================================================================
print("=========================== updates 的方式输出 ===================")
for event in gc.stream({}, stream_mode="updates"):
    print("▶️ 输出结果:", event)

# =============================================================================
# 2️⃣ values：每次输出【完整状态】
# =============================================================================
print("=========================== values 的方式输出 ===================")
for event in gc.stream({}, stream_mode="values"):
    print("▶️ 输出结果:", event)

# =============================================================================
# 3️⃣ debug：最详细（节点、状态、元信息）
# =============================================================================
print("=========================== debug 的方式输出 ===================")
for event in gc.stream({}, stream_mode="debug"):
    print("▶️ 输出结果:", event)

# =============================================================================
# 4️⃣ messages：专门用于消息流（Agent / Chat 场景）
# 你的 State 没有 messages 字段，所以输出空，但我给你正确写法！
# =============================================================================
print("=========================== messages 的方式输出 ===================")
init_input = {
    "messages": [HumanMessage(content="帮我处理任务")]
}
for msg_chunk, meta in gc.stream(init_input, stream_mode="messages"):
    if msg_chunk.content:
        print(msg_chunk.content,end="***",flush=True)
# =============================================================================
# 5️⃣ checkpoints：输出【检查点历史】
# 必须开启 checkpointer 才能用！
# =============================================================================
print("=========================== checkpoints 的方式输出 ===================")

from langgraph.checkpoint.memory import MemorySaver
gc_with_check = gp.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "test123"}}

# 先运行一遍生成 checkpoint
gc_with_check.invoke({}, config=config)

# 再流式输出检查点
for event in gc_with_check.stream({}, config, stream_mode="checkpoints"):
    print("▶️ 检查点输出:", event)