"""
提示词模板
"""
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
# 创建一个模板
chatPrompt = ChatPromptTemplate([
    ("system", "将用户输入的中文翻译成{language}，直接翻译不要废话！"),
    ("user", "{text}"),
])

# 模板填充数据
pt = chatPrompt.format(text="你爱我我爱你，蜜雪冰城甜蜜蜜！", language="法语")
print("=============提示词模板==================")
print(pt)
print("=======================================")

# 创建模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model=os.getenv("DASHSCOPE_TEXT_MODEL"),
)
resp = llm.invoke(pt)
print(resp)
print(resp.content)
