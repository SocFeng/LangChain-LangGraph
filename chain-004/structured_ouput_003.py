from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from common.model import llm_openai  # 你的模型

# 1. 定义输出结构
class ContactInfo(BaseModel):
    name: str
    email: str
    age: int
    phone: str

# 2. 解析器
parser = PydanticOutputParser(pydantic_object=ContactInfo)

# 3. 提示词
prompt = ChatPromptTemplate.from_messages([
    ("user", "从文本提取信息，按要求返回JSON格式\n{format_instructions}\n文本:{text}")
]).partial(format_instructions=parser.get_format_instructions())

# 4. 构造链（完全不用 Agent！）
chain = prompt | llm_openai | parser

# 5. 调用
resp = chain.invoke({
    "text": "从: tyom 15137617301 28 2418028741@qq.com 提取信息"
})

# 6. 输出
print("✅ 提取结果：")
print(resp)