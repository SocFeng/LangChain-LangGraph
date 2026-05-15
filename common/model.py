import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 创建大模型对象
llm_openai = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model=os.getenv("DASHSCOPE_TEXT_MODEL"),
    model_kwargs={"tool_choice": "auto"}
)
mimo_openai = ChatOpenAI(
    api_key=os.getenv("MIMO_API_KEY"),
    base_url=os.getenv("MIMO_BASE_URL"),
    model=os.getenv("MIMO_TEXT_MODEL"),
    model_kwargs={"tool_choice": "auto"},
    extra_body={"thinking": {"type": "disabled"}}
)
