# 短期记忆和长期记忆

## 短期记忆使用 dev
"""
    memory_checkpointer = InMemorySaver()

    # 创建 Agent 时启用 checkpointer
    agent = create_agent(
        model="your-llm",
        tools=your_tools,
        checkpointer=memory_checkpointer  # <- 关键参数
    )
"""

## 短期记忆 生产(进行持久化)
"""
    from langgraph.checkpoint.postgres import PostgresSaver
    
    DB_URI = "postgresql://user:pass@localhost/dbname"
    checkpointer = PostgresSaver.from_conn_string(DB_URI)
    await checkpointer.setup()  # 首次使用需初始化数据库表
    
    agent = create_agent(model=your_model, checkpointer=checkpointer)
"""


# 长期记忆

"""
from langgraph.store.memory import InMemoryStore

memory_store = InMemoryStore()

# …… 与用户对话，了解到他偏的好后 ……
user_id = "user_123"
# 将学习到的信息，存入用户的专属“抽屉”（命名空间）里[reference:19]
await memory_store.aput(
    ["users", user_id, "preferences"],
    {"language": "简体中文", "interests": ["AI", "LangChain"]}
)

# 在后续任意对话的任意环节，都可以随时取出这份记忆
preferences = await memory_store.aget(["users", user_id, "preferences"])
print(preferences)  # 输出：{'language': '简体中文', 'interests': ['AI', 'LangChain']}
"""
