import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# 方式 1 通过封装的api，但是有地方需要注意
""" 

llm = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 这里必须是 dashscope_api_key -- 这里使用纯文本模型不能使用多模态模型
    model="qwen3.6-max-preview",
)

print(llm.invoke("不需要思考，简单干脆的，直接回答你是谁？").content)
"""

# 方式 2 通过openai的规范协议

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model=os.getenv("DASHSCOPE_TEXT_MODEL"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)
resp = llm.invoke("不需要思考，简单干脆的，直接回答你是谁？")
print(resp.content)


# 方式 3 模型动态切换 --- 到某个条件，切换模型
