from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pymongo import MongoClient

from common.model import mimo_openai

# ==========================
# 1. 配置
# ==========================
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "langgraph_memory"
COLLECTION = "checkpoints"

# 摘要模板
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "请把下面的对话历史总结成一段简短摘要，保留关键信息：\n{history}"),
])


# ==========================
# 2. 工具（可选）
# ==========================
def get_weather(location: str):
    return f"{location} 天气晴，25°C"


# ==========================
# 3. 核心节点：聊天 + 裁剪 + 摘要
# ==========================
def chat_node(state: MessagesState):
    messages = state["messages"]

    # ----------------------
    # 🔥 第一步：裁剪消息（只保留最新 5 条）
    # ----------------------
    trimmed_messages = trim_messages(
        messages,
        max_tokens=5,  # 保留最新5条
        token_counter=len,  # 简单按条数计算
        allow_partial=False,
    )

    # ----------------------
    # 🔥 第二步：如果消息过长 → 生成摘要
    # ----------------------
    if len(messages) > 6:
        print("\n📌 消息过长，正在生成摘要...")

        # 把旧消息转成文本
        history_text = "\n".join([
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}：{m.content}"
            for m in messages[:-4]
        ])

        # 生成摘要
        summary_chain = SUMMARY_PROMPT | mimo_openai
        summary = summary_chain.invoke({"history": history_text}).content
        print(f"✅ 对话摘要：{summary}")

        # 新消息 = 摘要 + 最新4条
        new_messages = [
                           SystemMessage(content=f"对话摘要：{summary}")
                       ] + messages[-4:]

        # 调用模型
        response = mimo_openai.invoke(new_messages)
    else:
        # 消息不长 → 直接回答
        response = mimo_openai.invoke(trimmed_messages)

    return {"messages": response}


# ==========================
# 4. 构建图
# ==========================
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# 持久化
with MongoDBSaver.from_conn_string(MONGODB_URI, db_name=DB_NAME, collection_name=COLLECTION) as cp:
    graph = builder.compile(checkpointer=cp)

    # ==========================
    # 5. 测试：连续对话 → 自动裁剪+摘要
    # ==========================
    config = {"configurable": {"thread_id": "test_summary_01"}}

    # 第一轮
    print("=" * 70)
    graph.invoke({
        "messages": [HumanMessage("你好，我叫小明，我喜欢打篮球")]
    }, config=config)

    # 第二轮
    graph.invoke({
        "messages": [HumanMessage("我今年20岁，在上海读书")]
    }, config=config)

    # 第三轮
    graph.invoke({
        "messages": [HumanMessage("我最喜欢吃火锅")]
    }, config=config)

    # 第四轮
    graph.invoke({
        "messages": [HumanMessage("我最喜欢的运动是跑步")]
    }, config=config)

    # 第五轮 → 触发消息裁剪 + 摘要
    print("\n" + "=" * 70)
    res = graph.invoke({
        "messages": [HumanMessage("你还记得我的信息吗？")]
    }, config=config)

    # ==========================
    # 输出最终结果
    # ==========================
    print("\n" + "=" * 70)
    print("🤖 最终回答：")
    print(res["messages"][-1].content)
    print("=" * 70)

# ==========================
# 🔥 验证 MongoDB 数据
# ==========================
client = MongoClient(MONGODB_URI)
coll = client[DB_NAME][COLLECTION]
count = coll.count_documents({"configurable.thread_id": "test_summary_01"})
print(f"\n✅ MongoDB 中已存储对话条数：{count}")