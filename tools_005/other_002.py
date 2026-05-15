import os
from datetime import datetime
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage  # ✅ 关键导入

from common.model import llm_openai


# ========== 工具 ==========
@tool
def get_current_date() -> str:
    """获取今天的日期"""
    return datetime.now().strftime("%Y-%m-%d")

@tool
def add_numbers(a: float, b: float) -> float:
    """计算两个数字的和"""
    return a + b

# ========== 回调处理器 ==========
class ToolCallPrinter(BaseCallbackHandler):
    def on_tool_start(self, serialized: dict, input_str: str, *, run_id, **kwargs):
        tool_name = serialized.get("name", "Unknown")
        print(f"\n🔧 [工具开始] {tool_name}")
        print(f"   输入: {input_str}")
        print(f"   run_id: {run_id}")

    def on_tool_end(self, output, *, run_id, **kwargs):
        if isinstance(output, ToolMessage):
            out = output.content
        else:
            out = str(output)
        print(f"   ✅ [工具结束] 输出: {out}")
        print("   " + "-" * 40)

    def on_tool_error(self, error, **kwargs):
        print(f"   ❌ [错误] {error}")

# ========== Agent ==========
agent = create_agent(
    model=llm_openai,
    tools=[get_current_date, add_numbers],
    system_prompt="你是一个助手，可以使用工具。"
)

# ========== 执行 ==========
if __name__ == "__main__":
    question = "今天是什么日子？另外，请计算 3.5 加 7.2 等于多少？"
    print(f"问题: {question}\n")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": [ToolCallPrinter()]}
    )
    print(f"\n🤖 最终回答:\n{result['messages'][-1].content}")