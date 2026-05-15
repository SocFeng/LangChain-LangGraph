import os
import json
import random
from datetime import datetime
from typing import Annotated, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import ToolMessage
from langchain_core.callbacks import BaseCallbackHandler

from common.model import llm_openai


# ========== 1. 配置大模型（以通义千问为例）==========

# ========== 2. 定义工具 ==========
# 注意：工具间存在依赖关系，Agent 会自动规划执行顺序

@tool
def get_employee_info(employee_id: str) -> dict:
    """
    根据员工ID获取员工基本信息。
    参数：employee_id - 员工工号
    返回：包含姓名、部门、入职时间、当前薪资的字典
    """
    # 模拟数据库查询
    employees = {
        "10001": {"name": "张三", "dept": "技术部", "hire_date": "2020-03-15", "salary": 18000},
        "10002": {"name": "李四", "dept": "产品部", "hire_date": "2021-07-01", "salary": 22000},
    }
    return employees.get(employee_id, {"name": "未找到", "dept": "未知", "hire_date": "未知", "salary": 0})


@tool
def performance_query(employee_id: str, year: int = 2025) -> dict:
    """
    获取员工绩效数据。
    参数：
        employee_id - 员工工号
        year - 年份（默认2025）
    返回：绩效等级、项目完成率、年度得分
    """
    import random
    base_scores = {
        "10001": {"grade": "A", "completion_rate": 0.95, "score": 92},
        "10002": {"grade": "B+", "completion_rate": 0.87, "score": 84},
    }
    return base_scores.get(employee_id, {"grade": "C", "completion_rate": 0.75, "score": 70})


@tool
def analyze_performance(employee_id: str) -> str:
    """
    综合分析员工绩效，结合基本信息与绩效数据。
    注意：本工具会调用 get_employee_info 和 performance_query 获取依赖数据。
    """
    emp_info = get_employee_info.invoke({"employee_id": employee_id})
    perf = performance_query.invoke({"employee_id": employee_id})

    analysis = (
        f"员工 {emp_info['name']}（{emp_info['dept']}，入职 {emp_info['hire_date']}），"
        f"2025年度绩效评级 {perf['grade']}，项目完成率 {perf['completion_rate'] * 100:.0f}%，"
        f"综合得分 {perf['score']}。建议：{'表现优秀，予以表彰' if perf['grade'] == 'A' else '继续加强业务能力'}。"
    )
    return analysis


@tool
def calculate_stats(expression: str) -> str:
    """
    执行 Python 数学表达式进行统计计算。
    参数：expression - 合法的Python表达式，如 "sum([1,2,3])" 或 "95*0.8"
    返回：计算结果
    """
    import math
    allowed_context = {"__builtins__": {}, "sum": sum, "len": len, "max": max, "min": min, "math": math}
    try:
        result = eval(expression, allowed_context, {})
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def web_search(query: str) -> str:
    """
    模拟网络搜索行业动态和竞品信息。
    参数：query - 搜索关键词
    返回：包含搜索结果的摘要信息
    """
    # 模拟搜索结果（实际可替换为真实的搜索API）
    mock_db = {
        "it行业平均薪资": "2025年IT行业平均薪资为18500元/月",
        "绩效评估 最新标准": "2025年绩效评估新标准聚焦OKR量化指标",
    }
    for key, val in mock_db.items():
        if key in query.lower():
            return val
    return f"搜索'{query}'的3条结果：[1] 相关资讯A [2] 相关资讯B [3] 相关资讯C（此为模拟数据）"


@tool
def send_message(channel: str, title: str, content: str) -> str:
    """
    发送分析报告到指定渠道。
    参数：
        channel - 发送平台（如wechat/dingtalk）
        title - 报告标题
        content - 报告正文
    返回：发送状态
    """
    # 模拟发送成功
    return f"✅ 已将「{title}」发送到{channel}渠道，内容长度：{len(content)}字符"


# ========== 3. 自定义回调处理器（详细打印调用流程）==========
class DetailedCallbackHandler(BaseCallbackHandler):
    """打印完整的工具链调用信息，包括依赖关系和步骤编号"""

    def __init__(self):
        self.step_counter = 0
        self.runid_to_tool = {}  # 记录 run_id -> (step, tool_name)

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id, **kwargs):
        self.step_counter += 1
        tool_name = serialized.get("name", "Unknown")
        self.runid_to_tool[run_id] = (self.step_counter, tool_name)

        print(f"\n{'=' * 50}")
        print(f"[步骤 {self.step_counter}] 开始调用工具: {tool_name}")
        print(f"输入参数: {input_str}")
        print(f"[调用链] 当前工具可能依赖之前已执行的结果")

    def on_tool_end(self, output, *, run_id, **kwargs):
        # 根据 run_id 获取之前存储的工具信息
        step, tool_name = self.runid_to_tool.get(run_id, (self.step_counter, "Unknown"))

        # 提取输出内容（兼容 ToolMessage 类型）
        if hasattr(output, 'content'):  # ToolMessage 或类似对象
            output_str = output.content
        else:
            output_str = str(output)

        print(f"\n[步骤 {step}] ✓ {tool_name} 执行完成")
        print(f"输出结果: {output_str[:200]}{'...' if len(output_str) > 200 else ''}")
        print(f"{'=' * 50}")

    def on_tool_error(self, error: BaseException, *, run_id, **kwargs):
        step, tool_name = self.runid_to_tool.get(run_id, (self.step_counter, "Unknown"))
        print(f"\n[步骤 {step}] ✗ {tool_name} 执行出错: {error}")


# ========== 4. 创建 Agent ==========
agent = create_agent(
    model=llm_openai,
    tools=[
        get_employee_info,
        performance_query,
        analyze_performance,
        calculate_stats,
        web_search,
        send_message
    ],
    system_prompt="""
    你是一名专业的数据分析师助手。用户会提出数据查询、分析和汇报请求。

    请遵循以下原则：
    1. 当需要分析某员工绩效时，先调用 get_employee_info 和 performance_query 获取数据，再调用 analyze_performance 做综合评估。
    2. 如果需要行业对比数据，可以调用 web_search。
    3. 若需要统计数据（如平均值、百分比），使用 calculate_stats 而非依赖大模型。
    4. 最终生成完整分析报告后，调用 send_message 发送结果。

    请按照由基础到综合的顺序调用工具，不要同时发起过多无依赖关系的调用。
    """
)

# ========== 5. 执行复杂任务 ==========
if __name__ == "__main__":
    # 用户提出一个包含多步依赖的任务
    task = """
    请帮我对员工ID=10001进行绩效分析，并完成以下任务：

    1. 获取该员工的基本信息和绩效数据。
    2. 分析该员工的综合绩效水平。
    3. 计算一个加权指标：绩效得分占比70% + 项目完成率占比30% = 最终绩效分。
    4. 查询IT行业平均薪资，与该员工当前薪资做对比。
    5. 将以上所有分析内容整理成清晰的报告，通过 send_message 工具发送到 wechat 系统。
    """

    print("🚀 开始执行智能体任务...\n")
    print("用户任务:", task)
    print("\n" + "🔄🔄🔄 Agent 工具调用流程 🔄🔄🔄")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"callbacks": [DetailedCallbackHandler()]}
    )

    print("\n" + "=" * 60)
    print("✅ 任务执行完成")
    print("\n📄 最终输出报告：")
    print(result["messages"][-1].content)