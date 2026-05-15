# 创建一个模板
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.model import llm_openai

# LCEL == LangChain （language Chain）
# chain 就是一个链条，将一个固定的步骤，串联起来，按照步骤执行，并且Chain的各个节点数据能够流动


# 创建提示词模板
prompt = ChatPromptTemplate([
    ("system", "将用户输入的中文翻译成{language}，直接翻译不要废话！"),
    ("user", "{text} 返回的数据结构是{dataType}类型！"),
])

# prompt = chatPrompt.format(text="你爱我我爱你，蜜雪冰城甜蜜蜜！", language="法语", dataType="string")

# 创建输出解释器
out_parse = StrOutputParser()

# 创建大模型
llms = llm_openai

# 创建一个简单的链 不涉及数据的流转
# 将这些步骤封装起来成 一个chain ，通过chain来执行这个步骤
# ps: prompt -> llms -> out_parse
_chain = prompt | llms | out_parse

resp = _chain.invoke({"text": "你爱我我爱你，蜜雪冰城甜蜜蜜！", "language": "法语", "dataType": "string"})

print(resp)
"""
Tu m'aimes, je t'aime, Mixue Bingcheng est doux comme le miel !
"""
