import os

from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.llms import Tongyi

# 导入环境变量
load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")
model_name = os.getenv("DASHSCOPE_TEXT_MODEL")


def main():
    # 创建模型对象 -- openai方式创建
    # llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
    # resp = llm.invoke("不需要思考，简单干脆的，直接回答你是谁？")
    # print(resp.content)

    # 创建模型对象 -- 使用千问自己的sdk创建)  最新版本的包，直接返回回答的内容
    # 自己的sdk创建不用加地址了 --- 当然也可以加
    llm = Tongyi(api_key=api_key, model_name=model_name)
    resp = llm.invoke("不需要思考，简单干脆的，直接回答你是谁？")
    print(resp)


if __name__ == "__main__":
    main()
