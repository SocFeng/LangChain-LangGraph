from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# 你的模型（必须加 model_kwargs={"tool_choice":"auto"}）
from common.model import llm_openai


# ======================
# 工具（我帮你写了标准入参出参描述）
# ======================
@tool
def search_tianqi(city: str) -> dict:
    """
    查询指定城市明天的天气
    Args:
        city: 要查询的城市名称（例如：北京、上海）
    Returns:
        字典格式，包含天气状态tq 和 温度wd
    """
    return {"tq": "晴天", "wd": 45}


@tool
def search_todo(weather: str, temperature: int) -> str:
    """
    根据天气和温度，判断是否适合出门
    Args:
        weather: 天气状况（晴天/雨天/雪天）
        temperature: 温度，数字
    Returns:
        字符串：可以出门 或 不适合出门
    """
    if temperature < 20:
        return "可以出门"
    return "不适合出门"


@tool(name_or_callable="add_num")
def add(n: int, m: int) -> int:
    """计算两个数的和"""
    return n + m


@tool(name_or_callable="mul_num")
def mul(n: int, m: int) -> int:
    """计算两个数的乘积"""
    return n * m


# ======================
# 标准提示词
# ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，必须严格按照步骤调用工具，一步一步来。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ======================
# 创建 Agent
# ======================
agent = create_tool_calling_agent(
    llm=llm_openai,
    tools=[mul, add, search_tianqi, search_todo],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[mul, add, search_tianqi, search_todo],
    verbose=True,
    handle_parsing_errors=True  # 🔥 必加，防止国产模型解析报错
)

# ======================
# 调用测试
# ======================
if __name__ == '__main__':
    resp = agent_executor.invoke({
        "input": "4 * 4 * (3 + 1) 等于多少？"
    })
    # resp = agent_executor.invoke({
    #     "input": "明天上海的天气怎么样？能不能出门？请一步一步调用工具！"
    # })
    print("\n结果：", resp["output"])
