from typing import TypedDict, Dict, Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, SummarizationMiddleware
from langchain_core.messages import RemoveMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, state
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from common.model import mimo_openai, llm_openai


# 1. 定义工具
@tool
def add_nums(num1: int, num2: int) -> int:
    """计算两个数的和"""
    return num1 + num2


def memory_base_use():
    """ 短期记忆（正确版）"""
    memory = InMemorySaver()

    agent = create_agent(
        model=mimo_openai,
        tools=[add_nums],
        checkpointer=memory,  # 这里才有效！
        system_prompt="你只能使用工具来计算结果，请勿使用其他方式计算结果。"
    )

    # 3. 正确调用格式
    resp = agent.invoke(
        {
            "messages": [
                ("user", "计算299+347的结果！")  # 用元组格式，避免循环
            ]
        },
        config={"configurable": {"thread_id": "user_001"}}
    )

    # 4. 只打印最终答案（干净输出）
    final_answer = resp["messages"][-1].content
    print("\n最终结果：", final_answer)


def memory_custom_use():
    class CustomAgentState(MessagesState):
        """包含 messages 的状态（继承自 MessagesState），并添加自定义字段"""
        user_id: str
        preferences: Dict[str, Any]

    memory = InMemorySaver()

    agent = create_agent(
        model=mimo_openai,
        tools=[add_nums],
        state_schema=CustomAgentState,
        checkpointer=memory,
        system_prompt="你只能使用工具来计算结果，请勿使用其他方式计算结果。只掉一调用一次工具！"
    )

    resp = agent.invoke(
        {
            "messages": [
                ("user", "计算299+347的结果！")
            ],
            "user_id": "user_123",
            "preferences": {"theme": "dark"}
        },
        config={"configurable": {"thread_id": "1"}}
    )

    # 4. 只打印最终答案（干净输出）
    final_answer = resp["messages"][-1].content
    print("\n最终结果：", final_answer)


def memory_cut_use():
    memory = InMemorySaver()

    # 裁剪消息
    @before_model
    def cut_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        if len(messages) <= 3:
            return None
        print("======================================================")
        print(messages)
        print("======================================================")
        first_msg = messages[0]
        recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
        new_messages = [first_msg] + recent_messages
        return {
            "messases": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }

    agent = create_agent(
        model=mimo_openai,
        tools=[add_nums],
        middleware=[cut_messages],
        checkpointer=memory,
        system_prompt="你只能使用工具来计算结果，请勿使用其他方式计算结果。只掉一调用一次工具"
    )
    for i in range(5):
        resp = agent.invoke(
            {
                "messages": [
                    ("user", f"计算299+{i}的结果！")  # 用元组格式，避免循环
                ]
            },
            config={"configurable": {"thread_id": "user_001"}}
        )


def memory_del_use():
    memory = InMemorySaver()

    # 裁剪消息
    @before_model
    def delete_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state["messages"]
        if len(messages) > 2:
            # 这是删除所有消息
            # {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
            return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

    agent = create_agent(
        model=mimo_openai,
        tools=[add_nums],
        middleware=[delete_messages],
        checkpointer=memory,
        system_prompt="你只能使用工具来计算结果，请勿使用其他方式计算结果。只掉一调用一次工具"
    )
    for i in range(2):
        resp = agent.invoke(
            {
                "messages": [
                    ("user", f"计算299+{i}的结果！")  # 用元组格式，避免循环
                ]
            },
            config={"configurable": {"thread_id": "user_001"}}
        )


def memory_summary_use():
    memory = InMemorySaver()

    agent = create_agent(
        model=mimo_openai,
        tools=[add_nums],
        checkpointer=memory,
        middleware=[
            SummarizationMiddleware(
                model=llm_openai,
                trigger=("tokens", 3000),
                keep=("messages", 6),
            )
        ],
        system_prompt="你只能使用工具来计算结果，请勿使用其他方式计算结果。只掉一调用一次工具"
    )
    for i in range(5):
        resp = agent.invoke(
            {
                "messages": [
                    ("user", f"计算299+{i}的结果！")  # 用元组格式，避免循环
                ]
            },
            config={"configurable": {"thread_id": "user_001"}}
        )


if __name__ == "__main__":
    # memory_base_use()
    # memory_custom_use()
    # memory_cut_use()
    # memory_del_use()
    memory_summary_use()
