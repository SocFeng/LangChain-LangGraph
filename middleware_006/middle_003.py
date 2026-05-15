"""
complex_multi_middleware_demo.py
一个复杂中间件+多工具+多轮ReAct循环的完整示例。

业务场景：
用户要求：
1. 查询员工“张三”（身份证 11010119900307663X）的基本信息和绩效数据。
2. 计算加权绩效得分（绩效评分 * 0.7 + 完成率 * 0.3 * 100）。
3. 搜索行业平均薪资做对比。
4. 将分析报告发送至 admin@company.com。

中间件链（按顺序执行）：
- LoggingMiddleware: 记录每一步关键事件。
- PIIMiddleware: 对用户输入的身份证号进行脱敏。
- ToolRetryMiddleware: 工具调用失败时重试（例如模拟网络波动）。
- ToolCallLimitMiddleware: 限制整个任务中最多调用工具 10 次，防止无限循环。
- ModelFallbackMiddleware: 主模型（故意错的 base_url）自动切换到备用模型。

要求：
- 请替换代码中的 API Key 为你自己的有效 Key。
- 运行后会看到详细的中间件日志和工具调用流程。
"""

import os
import re
import time
import random
from typing import Any, Dict, List
from datetime import datetime

# LangChain 核心导入
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    PIIMiddleware,
    ToolRetryMiddleware,
    ToolCallLimitMiddleware,
    ModelFallbackMiddleware,
    RedactionRule, AgentMiddleware,
)
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from common.model import mimo_openai

# ------------------------------------------------------------
# 0. 配置 API Key 和环境（请修改）
# ------------------------------------------------------------
# 注意：请将下面的 your-api-key-here 替换为真实的 DashScope API Key
os.environ["DASHSCOPE_API_KEY"] = "your-api-key-here"

# ------------------------------------------------------------
# 1. 定义模型：主模型（故意配置错误）和 备用模型（正常）
# ------------------------------------------------------------
# 主模型：错误的 base_url（模拟服务故障）
primary_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v10",  # 错误路径，会返回404
    model="qwen-plus",
    temperature=0,
)

# 备用模型：正确的配置（使用 turbo 降低成本）
fallback_llm = mimo_openai


# ------------------------------------------------------------
# 2. 业务工具（模拟实际系统的数据查询和计算）
# ------------------------------------------------------------
@tool
def get_employee_info(id_card: str) -> str:
    """
    根据身份证号获取员工基本信息。
    注意：身份证属于敏感信息，中间件会自动脱敏。
    """
    # 模拟数据库查询
    print("===================",id_card)
    emp_db = {
        "11010119900307663X": {
            "name": "张三",
            "dept": "技术部",
            "position": "高级工程师",
            "hire_date": "2018-07-01"
        }
    }
    emp = emp_db.get(id_card)
    if not emp:
        return "未找到该身份证对应的员工信息"
    return f"员工：{emp['name']}，部门：{emp['dept']}，职位：{emp['position']}，入职时间：{emp['hire_date']}"


@tool
def get_performance_data(employee_name: str) -> str:
    """
    根据员工姓名获取当年绩效数据（评分和项目完成率）。
    """
    # 模拟绩效数据
    perf_db = {
        "张三": {"score": 88, "completion_rate": 0.92}
    }
    data = perf_db.get(employee_name)
    if not data:
        return "未找到该员工的绩效数据"
    return f"绩效评分：{data['score']}，项目完成率：{data['completion_rate'] * 100:.0f}%"


@tool
def calculate_weighted_score(score: float, completion_rate: float) -> str:
    """
    计算加权绩效得分。公式：score * 0.7 + completion_rate * 100 * 0.3
    参数均为数值。
    """
    weighted = score * 0.7 + completion_rate * 100 * 0.3
    return f"加权绩效得分：{weighted:.2f}分"


@tool
def search_industry_avg_salary() -> str:
    """
    搜索 IT 行业平均薪资（模拟网络请求，有时会失败触发重试）。
    """
    # 模拟 30% 概率失败，以演示 ToolRetryMiddleware
    if random.random() < 0.3:
        raise ConnectionError("行业薪资 API 暂时不可用，请重试")
    # 模拟搜索结果
    return "2025年IT行业平均月薪为 18500 元"


@tool
def send_report_email(to: str, subject: str, content: str) -> str:
    """
    发送分析报告到指定邮箱。
    """
    # 模拟邮件发送，实际可接入 SMTP
    print(f"\n[模拟邮件] 收件人: {to}")
    print(f"[模拟邮件] 主题: {subject}")
    print(f"[模拟邮件] 内容:\n{content[:200]}...")
    return f"报告已成功发送至 {to}"


# ------------------------------------------------------------
# 3. 自定义回调：详细记录每个工具调用和模型调用
# ------------------------------------------------------------
class VerboseCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.step = 0

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        self.step += 1
        tool_name = serialized.get("name", "unknown")
        print(f"\n🔧 [步骤 {self.step}] 调用工具: {tool_name}")
        print(f"   输入: {input_str}")

    def on_tool_end(self, output, **kwargs):
        # output 可能是 ToolMessage 或字符串
        out_str = output.content if hasattr(output, "content") else str(output)
        print(f"   ✅ 工具返回: {out_str[:150]}...")

    def on_tool_error(self, error, **kwargs):
        print(f"   ❌ 工具错误: {error}")


class LoggingMiddleware(AgentMiddleware):
    """基于 AgentMiddleware 的日志中间件"""

    def before_model(self, state: AgentState, runtime: Runtime) -> None:
        print(f"\n[Log] 即将调用模型，当前消息数: {len(state['messages'])}")
        return None  # 返回 None 表示继续执行


class MyLoggingHandler(BaseCallbackHandler):
    """一个简单的自定义日志回调处理器"""

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        """当LLM开始调用时"""
        model_name = serialized.get("kwargs", {}).get("model", "unknown")
        print(f"[日志] 🤖 开始调用模型: {model_name}")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        """当工具开始执行时"""
        tool_name = serialized.get("name", "unknown")
        print(f"[日志] 🔧 开始调用工具: {tool_name}, 参数: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """当工具执行结束时"""
        # output 可能是 ToolMessage 对象，也可能是普通字符串，做个兼容处理
        output_str = output.content if hasattr(output, 'content') else str(output)
        print(f"[日志] ✅ 工具执行完成, 结果: {output_str[:100]}...")


# ------------------------------------------------------------
# 4. 配置中间件链（顺序至关重要！）
# ------------------------------------------------------------
middlewares = [

    LoggingMiddleware(),
    # 4.2 PII 中间件：脱敏身份证号（内置模式检测+自定义规则）
    # PIIMiddleware(
    #     pii_type="id_card",  # 自定义一个类型名称
    #     # strategy="redact",  # 设置处理策略为替换
    #     detector=r"\b\d{17}[\dXx]\b",  # 身份证号的正则表达式
    #     apply_to_input=True,  # 应用到用户输入
    #     apply_to_output=False,  # 是否应用到模型输出
    # ),

    # 4.3 工具重试中间件：对失败的工具最多重试2次，指数退避
    ToolRetryMiddleware(max_retries=2, backoff_factor=1.0),

    # 4.4 工具调用限流中间件：整个任务最多调用工具8次（防止无限循环）
    ToolCallLimitMiddleware(run_limit=8),

    # 4.5 模型 fallback 中间件：主模型失败时自动切换到备用模型
    ModelFallbackMiddleware(fallback_llm),
]

# ------------------------------------------------------------
# 5. 创建 Agent（主模型传入错误的那个，触发 fallback）
# ------------------------------------------------------------
agent = create_agent(
    model=primary_llm,  # 主模型（故意配置错误的 base_url）
    tools=[
        get_employee_info,
        get_performance_data,
        calculate_weighted_score,
        search_industry_avg_salary,
        send_report_email,
    ],
    system_prompt="""
    你是一个专业的数据分析助手。用户会提出绩效分析需求。

    请严格按照以下步骤执行：
    1. 先调用 get_employee_info 获取员工基本信息（参数为身份证号）。
    2. 从返回的信息中提取员工姓名，然后调用 get_performance_data 获取绩效数据。
    3. 从绩效数据中提取评分和完成率，调用 calculate_weighted_score 计算加权得分。
    4. 调用 search_industry_avg_salary 获取行业平均薪资。
    5. 将以上所有信息整合成一份结构清晰的报告，然后调用 send_report_email 发送给指定邮箱。

    注意：每一步都要等待工具返回结果后再进行下一步。不要一次性调用多个工具。
    """,
    middleware=middlewares,
)

# ------------------------------------------------------------
# 6. 主程序：执行复杂任务
# ------------------------------------------------------------
if __name__ == "__main__":
    # 用户任务（包含敏感身份证号，PII 中间件会脱敏）
    user_task = """
    请帮我分析员工绩效：
    身份证号：11010119900307663X，姓名张三。

    请完成以下子任务：
    1. 获取员工基本信息。
    2. 获取绩效数据（绩效评分和项目完成率）。
    3. 计算加权绩效得分（评分占70%，完成率占30%）。
    4. 搜索行业平均薪资。
    5. 将最终分析报告发送到邮箱 admin@company.com。

    报告要包含：员工姓名、部门、绩效评分、完成率、加权得分、与行业平均薪资的对比结论。
    """

    print("=" * 70)
    print("开始执行复杂智能体任务")
    print("=" * 70)
    print("用户任务:\n", user_task)
    print("\n中间件链已就绪：Logging → PII → ToolRetry → ToolCallLimit → ModelFallback")
    print("注意：主模型 base_url 错误，会自动 fallback 到备用模型。")
    print("=" * 70)

    # 自定义回调（用于更详细的输出）
    config = {"callbacks": [VerboseCallbackHandler()]}

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_task}]},
            config=config,
        )
        print("\n" + "=" * 70)
        print("✅ 任务执行完成")
        print("最终回答摘要:")
        final_content = result["messages"][-1].content
        print(final_content[:500] + "..." if len(final_content) > 500 else final_content)
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ 智能体执行失败: {e}")
        import traceback

        traceback.print_exc()
