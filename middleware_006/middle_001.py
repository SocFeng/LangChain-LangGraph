from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.globals import set_verbose
from langchain_core.tools import tool

from common.model import mimo_openai
set_verbose(True)   # 开启详细日志

# 1. 配置模型 (以通义千问为例)

# 2. 设置一个简单的工具，用于模拟多轮对话
@tool
def answer_query(query: str) -> str:
    """回答用户问题。此工具仅用于演示对话累计效果。"""
    return f"已了解你的问题是：{query}"


# 3. 配置中间件：这是今天的核心嘉宾
middlewares = [
    SummarizationMiddleware(
        model=mimo_openai,  # 选用一个更便宜的模型来做总结
        trigger=("messages", 4),  # 重要! 当消息总数达到4条时，触发总结
        keep=("messages", 2),  # 重要! 总结完成后，只保留最近的2条原始消息
    ),
]

# 4. 创建智能体，并挂载中间件
agent = create_agent(model=mimo_openai, tools=[answer_query], middleware=middlewares)

# 在文件开头初始化历史列表
history = []


def send_message(user_input):
    global history
    # 将当前用户消息加入历史
    history.append({"role": "user", "content": user_input})

    print(f"\n👤 用户: {user_input} 简约的回答！")
    # 注意：这里传入的是完整历史
    result = agent.invoke({"messages": history})

    # 更新历史为返回的完整消息列表（这样摘要消息就会被保留）
    history = result["messages"]

    # 打印助手回答
    print(f"🤖 助手: {history[-1].content}")

    # 打印完整消息历史，以便观察摘要是否出现
    print("\n📜 当前所有消息（含摘要）：")
    for i, msg in enumerate(history):
        role = msg.__class__.__name__
        content_preview = msg.content
        print(f"  [{i}] {role}: {content_preview}")
    print("-" * 50)

if __name__ == "__main__":
    # 连续对话，模拟消息累积
    send_message("我是，哈机密")
    send_message("我家有两只猫")
    send_message("1+2+3 等于多少")
    send_message("现在的时间？")  # 注意: 第4条消息即将触发总结
