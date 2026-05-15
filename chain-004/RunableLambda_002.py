# 普通的函数是无法，再当成chain中的一环使用的，需要特殊处理
import time
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, chain

from common.model import llm_openai


def get_len(text: str) -> int:
    """
    获取每个字符串的长度
    :param text:
    :return:
    """
    print("这里是长度：", text)
    time.sleep(1)
    return len(text)


def mul(str1: str, str2: str) -> int:
    """
    计算两个字符串的长度 乘积
    :param str1:
    :param str2:
    :return:
    """
    print("这里是计算长度的乘积：")
    return get_len(str1) * get_len(str2)


@chain
def mul_len(d: dict):
    """
    计算字典的两个元素 t1,t2 长度的乘积
    :param d:{"t1":"xxx","t2":"xxx"}
    :return: 10
    """
    print("这里是mul_len入口,原数据是：", d)
    return mul(d["t1"], d["t2"])


# 创建提示词模板
template = ChatPromptTemplate.from_template("{a} + {b}的值是多少 {str}返回？")
# 获取这个itemgetter("name")({"name": "xxx"}) 获取字典某个key的内容
# print(itemgetter("name")({"name": "xxx"}))


# 创建一个模板模型chain
chain_pt_llm = template | llm_openai

# 创建一个输出格式化器
str_out = StrOutputParser()
# 创建一个chain
_chain = (
        {
            "a": itemgetter("name") | RunnableLambda(get_len),
            "b": {"t1": itemgetter("name"), "t2": itemgetter("re_name")} | mul_len,
            "str": itemgetter("str")
        }
        | chain_pt_llm
        | str_out
)

resp = _chain.invoke({
    "name": "你的名字是什么",
    "re_name": "我的名字是tom",
    "str": "字符串"
})
print(resp)
# 分析执行过程
"""
1. _chain.invoke  
    传入输入参数：data_001 -->{"name": "你的名字是什么", "re_name": "我的名字是tom", "str": "字符串"}
2. _chain 构建数据字典
    {a:xxx,b:xxx,str:xxx} 同时获取执行 
    2.1 "a": itemgetter("name") | RunnableLambda(get_len)
        获取到data_001的name字段"你的名字是什么" --->  RunnableLambda(get_len) ---> 包装好的可执行函数，返回一个长度[get_len("你的名字是什么")]
    2.2 {"t1": itemgetter("name"), "t2": itemgetter("re_name")} | mul_len
        都见一个新的字典 {'t1': '你的名字是什么', 't2': '我的名字是tom'} ---> mul_len(可执行的函数) --> 返回结果
    2.3 "str": itemgetter("str") 
        获取执行阐述
    {"a": 7, "b": 56, "str": "字符串"}
3. chain_pt_llm
    在2中构建的字典数据传入这个chain中     template | llm_openai
    3.1 template
        将上一步构建的字典，写入的模板中
        "7 + 56 的值是多少 字符串返回？"
    3.2 llm_openai
        大模型执行格式化好的模板，并执行模板返回结构
         "63"
4. str_out
    格式化输出内容
    `"63"`

    
"""
"""
这里是长度： 你的名字是什么
这里是mul_len入口,原数据是： {'t1': '你的名字是什么', 't2': '我的名字是tom'}
这里是计算长度的乘积：
这里是长度： 你的名字是什么
这里是长度： 我的名字是tom
`"63"`
"""
