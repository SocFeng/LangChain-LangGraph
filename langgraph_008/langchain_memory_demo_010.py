import os
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.store.base import BaseStore
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore

from common.model import mimo_openai

# ==========================
# 🔥 关键：去掉 embedding，只用基础内存存储（不报错！）
# ==========================
in_memory_store = InMemoryStore()


def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    # 1. 获取用户ID（跨对话共享的核心）
    user_id = config["configurable"].get("user_id")
    namespace = ("memories", user_id)

    # 2. 安全读取用户记忆（容错，为空不崩溃）
    memories = store.search(namespace)
    info = "\n".join([d.value["data"] for d in memories]) if memories else ""

    # 3. 系统提示词
    sys_prompt = f"""你是一个助手，请根据以下信息回答问题。
用户信息：{info}"""

    # 4. 自动记忆用户信息
    last_message = state["messages"][-1]
    if "我是" in last_message.content or "记住" in last_message.content:
        memory = f"用户的名字是：{last_message.content}"
        # 存入用户全局记忆
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    # 5. 调用模型
    resp = mimo_openai.invoke(
        [{"role": "system", "content": sys_prompt}] + state["messages"]
    )
    return {"messages": resp}


# ======================
# 构建图
# ======================
gp = StateGraph(MessagesState)
gp.add_node("call_model", call_model)

gp.add_edge(START, "call_model")
gp.add_edge("call_model", END)

# 编译：checkpointer + store
gc = gp.compile(
    checkpointer=MemorySaver(),
    store=in_memory_store
)

# ======================
# 测试对话
# ======================
config = {"configurable": {"user_id": "user_1", "thread_id": "001"}}
input_messages = {"role": "user", "content": "你好我是我的名字是Tom"}

for ch in gc.stream({"messages": [input_messages]}, config, stream_mode="values"):
    ch["messages"][-1].pretty_print()

print("==============================================================================")
config = {"configurable": {"user_id": "user_1", "thread_id": "002"}}
input_messages = {"role": "user", "content": "你还记得我是谁吗？？"}
for ch in gc.stream({"messages": [input_messages]}, config, stream_mode="values"):
    ch["messages"][-1].pretty_print()

print("==============================================================================")
config = {"configurable": {"user_id": "user_2", "thread_id": "003"}}
input_messages = {"role": "user", "content": "你还记得我是谁吗？？"}
for ch in gc.stream({"messages": [input_messages]}, config, stream_mode="values"):
    ch["messages"][-1].pretty_print()
