from langchain_core.tools import tool
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from common.model import mimo_openai

# ======================
# 1. 定义工具
# ======================
@tool
def get_weather(location: str) -> str:
    """获取指定城市的天气信息"""
    if location == "北京":
        return "多云，温度 25°C"
    elif location == "上海":
        return "雷阵雨，温度 23°C"
    elif location == "广州":
        return "阴，温度 27°C"
    return "晴转多云，温度 28°C"

tools = [get_weather]

# ======================
# 2. MongoDB 持久化记忆
# ======================
MONGODB_URI = "mongodb://localhost:27017/"

# 创建 MongoDB 记忆存储器
with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:

    # ======================
    # 3. 创建 Agent（正确写法）
    # ======================
    graph = create_agent(
        mimo_openai,
        tools=tools,
        checkpointer=checkpointer  # 开启持久化记忆
    )

    # ======================
    # 4. 执行对话
    # ======================
    config = {"configurable": {"thread_id": "agent_001"}}

    response = graph.invoke(
        {
            "messages": [
                HumanMessage(content="北京今天天气怎么样？")
            ]
        },
        config=config
    )

    # ======================
    # 5. 美化输出
    # ======================
    print("\n" + "="*60)
    print("🤖 最终回答：")
    print(response["messages"][-1].content)
    print("="*60)