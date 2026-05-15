from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from common.model import llm_openai


# =====================================================
# reducer
# =====================================================

def merge_memory(old: dict, new: dict):
    if old is None:
        old = {}

    if new is None:
        new = {}

    return {
        **old,
        **new
    }


# =====================================================
# State
# =====================================================

class MyAgentState(TypedDict):
    messages: Annotated[list, add_messages]

    memory_store: Annotated[
        Dict[str, Any],
        merge_memory
    ]


# =====================================================
# Read Memory Tool
# =====================================================

@tool
def read_memory(runtime: ToolRuntime[MyAgentState]) -> str:
    """
    读取记忆
    """

    memory = runtime.state.get("memory_store", {})

    if not memory:
        return "当前没有任何记忆"

    result = []

    for k, v in memory.items():
        result.append(f"{k}: {v}")

    return "\n".join(result)


# =====================================================
# Write Memory Tool
# =====================================================

@tool
def write_memory(
        key: str,
        value: str,
        runtime: ToolRuntime[MyAgentState]
):
    """
    写入记忆
    """

    current_memory = runtime.state.get(
        "memory_store",
        {}
    )

    new_memory = {
        **current_memory,
        key: value
    }

    # 重点：
    # 必须有 ToolMessage
    return Command(
        update={

            # 更新状态
            "memory_store": new_memory,

            # 更新消息流
            "messages": [
                ToolMessage(
                    content=f"记忆已写入: {key}={value}",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )


# =====================================================
# Create Agent
# =====================================================

def create_memory_agent():

    checkpointer = InMemorySaver()

    agent = create_agent(
        model=llm_openai,

        tools=[
            read_memory,
            write_memory
        ],

        state_schema=MyAgentState,

        checkpointer=checkpointer,

        system_prompt="""
你是一个拥有短期记忆的AI助手。

规则：

1. 用户要求记住内容：
必须调用 write_memory

2. 用户询问历史记忆：
必须调用 read_memory

3. 不允许伪造记忆
"""
    )

    return agent


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    agent = create_memory_agent()

    config = {
        "configurable": {
            "thread_id": "user_001"
        }
    }

    # =================================================
    # 写入记忆
    # =================================================

    print("=" * 60)
    print("用户：记住我叫小张，喜欢喝美式咖啡")

    result1 = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "记住我叫小张，喜欢喝美式咖啡"
                )
            ]
        },
        config=config
    )

    print("AI：")
    print(result1["messages"][-1].content)

    # =================================================
    # 读取记忆
    # =================================================

    print("\n" + "=" * 60)
    print("用户：我喜欢喝什么？")

    result2 = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "我喜欢喝什么？"
                )
            ]
        },
        config=config
    )

    print("AI：")
    print(result2["messages"][-1].content)

    # =================================================
    # 再次读取
    # =================================================

    print("\n" + "=" * 60)
    print("用户：我叫什么名字？")

    result3 = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "我叫什么名字？"
                )
            ]
        },
        config=config
    )

    print("AI：")
    print(result3["messages"][-1].content)

    # =================================================
    # 查看状态
    # =================================================

    print("\n" + "=" * 60)

    snapshot = agent.get_state(config)

    print("最终状态：")
    print(snapshot.values)

    print("\nmemory_store:")
    print(snapshot.values.get("memory_store"))