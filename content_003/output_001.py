import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.model import llm_openai

load_dotenv()
# 创建一个模板
chatPrompt = ChatPromptTemplate([
    ("system", "将用户输入的中文翻译成{language}，直接翻译不要废话！"),
    ("user", "{text} 返回的数据结构是{dataType}类型！"),
])

# 模板填充数据

print("====================================字符串输出========================")
pt = chatPrompt.format(text="你爱我我爱你，蜜雪冰城甜蜜蜜！", language="法语", dataType="string")

resp = llm_openai.invoke(pt)
print("(str)这是标准输出完整输出：\n\t", resp)

# 创建输出解释器
strOutput = StrOutputParser()
strOut = strOutput.invoke(resp)
print("(str)这是输出解释器格式化后的输出：", strOut)

print("====================================json输出========================")
ptJson = chatPrompt.format(text="你爱我我爱你，蜜雪冰城甜蜜蜜！", language="法语", dataType="json")

respJson = llm_openai.invoke(ptJson)
print("(json)这是标准输出完整输出：\n\t", respJson)

# 创建输出解释器
jsonOutput = JsonOutputParser()
jsonOut = jsonOutput.invoke(respJson)
print("(json)这是输出解释器格式化后的输出：", jsonOut)

"""

====================================字符串输出========================
(str)这是标准输出完整输出：
	 content="Tu m'aimes, je t'aime, Mixue Bingcheng, doux et sucré !" additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 1857, 'prompt_tokens': 48, 'total_tokens': 1905, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 1832, 'rejected_prediction_tokens': None, 'text_tokens': 1857}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': None, 'text_tokens': 48}}, 'model_provider': 'openai', 'model_name': 'qwen3.6-plus', 'system_fingerprint': None, 'id': 'chatcmpl-e5092e29-249e-9ab6-a872-c17d267fa356', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019e1666-e107-77f3-bb98-2a1b977120b6-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 48, 'output_tokens': 1857, 'total_tokens': 1905, 'input_token_details': {}, 'output_token_details': {'reasoning': 1832}}
(str)这是输出解释器格式化后的输出： Tu m'aimes, je t'aime, Mixue Bingcheng, doux et sucré !
====================================json输出========================
(json)这是标准输出完整输出：
	 content='{"translation": "Tu m\'aimes, je t\'aime, Mixue Bingcheng est si doux !"}' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 2139, 'prompt_tokens': 48, 'total_tokens': 2187, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 2111, 'rejected_prediction_tokens': None, 'text_tokens': 2139}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': None, 'text_tokens': 48}}, 'model_provider': 'openai', 'model_name': 'qwen3.6-plus', 'system_fingerprint': None, 'id': 'chatcmpl-71564b55-3504-9838-bbd2-d24becf07968', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019e1667-6b5a-7ff0-b68e-bc8a6c5321ab-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 48, 'output_tokens': 2139, 'total_tokens': 2187, 'input_token_details': {}, 'output_token_details': {'reasoning': 2111}}
(json)这是输出解释器格式化后的输出： {'translation': "Tu m'aimes, je t'aime, Mixue Bingcheng est si doux !"}
"""
