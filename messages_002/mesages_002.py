from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from common.model import llm_openai

# langchain中的提示数据也叫消息

# 1 元组的形式
# messages = [
#     SystemMessage("将用户提供的话翻译成德语，不要废话！"),
#     HumanMessage("你爱我我爱你，蜜雪冰城甜蜜蜜！"),
#     AIMessage("这是AI回复的消息！：：")
# ]

# 2 字典的形式

# messages = [
#     {"role": "system", "content": "将用户提供的话翻译成德语，不要废话！"},
#     {"role": "user", "content": "你爱我我爱你，蜜雪冰城甜蜜蜜！"},
#     {"role": "assistant", "content": "这是AI回复的消息！：："}
# ]

# 3 元数据的形式
# 相当于添加额外的数据，可以做链路追踪！
messages = [
    SystemMessage(content="将用户提供的话翻译成德语，不要废话！", id="123456789", name="system"),
    HumanMessage(content="你爱我我爱你，蜜雪冰城甜蜜蜜！", id="0000000", name="user"),
    AIMessage(content="这是AI回复的消息！：：", id="987654321", name="assistant")
]


resp = llm_openai.invoke(messages)
print(resp)
