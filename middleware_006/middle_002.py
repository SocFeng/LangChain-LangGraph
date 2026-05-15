"""
model_fallback_demo.py
演示 ModelFallbackMiddleware 的正确用法：
- 主模型使用错误的 base_url（模拟故障）
- 自动切换到备用模型（正常的 mim_openai）
- 记录切换过程
"""

import os
import traceback
import warnings
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.globals import set_verbose
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 抑制版本警告（可选）
set_verbose(True)  # 开启详细日志

warnings.filterwarnings("ignore", category=UserWarning)

# 假设你的 common.model 中有 mimo_openai 实例
# 如果无法导入，可以手动定义一个备用模型
try:
    from common.model import mimo_openai
except ImportError:
    # 如果导入失败，创建一个备用模型（请替换为真实 API Key）
    mimo_openai = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY", "your-api-key"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0,
    )

# ------------------------------------------------------------
# 1. 主模型：故意配置错误的 base_url（路径为 v10，正常应为 v1）
# ------------------------------------------------------------
primary_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 正确的 API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v10",  # 错误 URL
    model="qwen-plus",
    temperature=0,
)

# ------------------------------------------------------------
# 2. 备用模型：使用正确的配置
# ------------------------------------------------------------
fallback_llm = mimo_openai  # 假设它是一个配置正确的 ChatOpenAI 实例

# ------------------------------------------------------------
# 3. 配置中间件（正确用法：用位置参数传入备用模型）
# ------------------------------------------------------------
fallback_middleware = ModelFallbackMiddleware(
    fallback_llm,  # 第一个备用模型
    # 可以继续传入第二个、第三个备用模型，例如：
    # fallback_llm2, fallback_llm3
)


# ------------------------------------------------------------
# 4. 简单工具
# ------------------------------------------------------------
@tool
def get_current_time() -> str:
    """获取当前服务器时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------
# 5. 创建智能体
# ------------------------------------------------------------
agent = create_agent(
    model=primary_llm,  # 主模型（故意配置错误）
    tools=[get_current_time],
    system_prompt="你是一个时间助手，可以调用工具获取当前时间。",
    middleware=[fallback_middleware],
)

# ------------------------------------------------------------
# 6. 可选：更多日志（通过回调）
# ------------------------------------------------------------
from langchain_core.callbacks import BaseCallbackHandler


class FallbackLoggingHandler(BaseCallbackHandler):
    def __init__(self):
        # 创建一个字典，用于存储每次调用对应的模型名称
        self.run_id_to_model = {}

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        # 1. 从 serialized 字典中稳健地提取模型名称
        model_name = (
            serialized.get('kwargs', {}).get('model_name') or
            serialized.get('id', [''])[-1] or  # 提取类名
            "unknown_model"
        )

        # 2. 获取当前调用的唯一ID，并将模型名存起来
        run_id = kwargs.get("run_id")
        if run_id:
            self.run_id_to_model[run_id] = model_name

        # 3. 打印开始信息
        print(f"[回调] 🔄 开始尝试使用模型: {model_name}")

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        # 1. 根据 run_id 获取之前存储的模型名称
        run_id = kwargs.get("run_id")
        model_name = self.run_id_to_model.pop(run_id, "unknown_model")

        # 2. 获取详细的错误信息
        error_type = type(error).__name__
        error_msg = str(error)
        # 获取完整的错误堆栈，有助于调试
        error_trace = traceback.format_exc()

        # 3. 打印失败信息，并包含模型名和错误详情
        print(f"\n[回调] ❌ 模型调用失败: {model_name}")
        print(f"         错误类型: {error_type}")
        print(f"         错误详情: {error_msg}")
        # 如果你需要更详细的堆栈，可以取消下面一行的注释
        # print(f"         错误堆栈:\n{error_trace}")
        print("-" * 50)


# ------------------------------------------------------------
# 7. 运行测试
# ------------------------------------------------------------
if __name__ == "__main__":
    config = {"callbacks": [FallbackLoggingHandler()]}
    print("=" * 60)
    print("主模型使用了错误的 base_url（v10），预期会自动切换到备用模型")
    print("=" * 60)

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "现在几点了？"}]},
            config=config,
        )
        print("\n✅ 最终回答:")
        print(result["messages"][-1].content)
    except Exception as e:
        print(f"\n❌ 所有模型均失败: {e}")
