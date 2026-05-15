# LangChain-LangGraph 学习项目

这是一个基于 **LangChain** 和 **LangGraph** 的 Python 学习项目，通过 8 个模块、30+ 个示例文件，系统学习大语言模型应用开发的核心概念和实践方法。

## 项目概述

本项目演示了如何使用 LangChain 和 LangGraph 框架构建 AI 应用，包括：
- 大模型调用与管理
- 提示词工程与消息处理
- 链式调用与数据流转
- 工具调用与 Agent 构建
- 中间件机制
- 记忆系统（短期/长期）
- LangGraph 图编排与流程控制

## 技术栈

- **Python 3.10+**
- **LangChain** - LLM 应用开发框架
- **LangGraph** - 状态图编排框架
- **通义千问 (Tongyi/Qwen)** - 阿里云大模型 API
- **MIMO** - 小米大模型 API
- **MongoDB** - 持久化存储（可选）

## 项目结构

```
LangChain-LangGraph/
├── common/                    # 公共模块
│   └── model.py              # 模型配置（千问、MIMO）
│
├── models_001/                # 模块1：模型基础
│   ├── 001_hello_langchain.py    # LangChain 入门 - 基础模型调用
│   └── 002_model.py              # 多种方式创建模型（OpenAI SDK / 千问 SDK）
│
├── messages_002/              # 模块2：消息与提示词
│   ├── prompt_template_001.py    # 提示词模板（ChatPromptTemplate）
│   ├── mesages_002.py            # 消息类型（SystemMessage / HumanMessage / AIMessage）
│   └── dynamic_prompt_003.py     # 动态提示词（根据上下文切换角色）
│
├── content_003/               # 模块3：输出解析
│   └── output_001.py             # 输出解析器（StrOutputParser / JsonOutputParser）
│
├── chain-004/                 # 模块4：链式调用（LCEL）
│   ├── chain_001.py              # 基础链：prompt | llm | parser
│   ├── RunableLambda_002.py      # RunnableLambda - 自定义函数融入链
│   └── structured_ouput_003.py   # 结构化输出（PydanticOutputParser）
│
├── tools_005/                 # 模块5：工具与Agent
│   ├── tools_001.py              # 基础工具调用（@tool 装饰器）
│   ├── other_002.py              # 工具链式调用（天气查询 → 出门判断）
│   └── next_agent.py             # 复杂 Agent（多工具协作 + 回调追踪）
│
├── middleware_006/            # 模块6：中间件
│   ├── middle_001.py             # SummarizationMiddleware（消息摘要压缩）
│   ├── middle_002.py             # ModelFallbackMiddleware（模型降级）
│   └── middle_003.py             # 复杂中间件链（PII脱敏/重试/限流/降级）
│
├── memory_007/                # 模块7：记忆系统
│   ├── memory_001.py             # 记忆基础概念（短期/长期记忆说明）
│   ├── memory_short_002.py       # 短期记忆实践（InMemorySaver / 消息裁剪 / 摘要）
│   └── memory_short_read_write_003.py  # 自定义记忆读写（ToolRuntime + Command）
│
├── langgraph_008/             # 模块8：LangGraph 图编排
│   ├── hello_langgraph_001.py          # LangGraph 入门（状态图基础）
│   ├── langgraph_serial_002.py         # 串行流程（节点顺序执行）
│   ├── langgraph_branch_003.py         # 并行分支（多节点同时执行）
│   ├── langgraph_conditions_004.py     # 条件分支（add_conditional_edges）
│   ├── langgraph_loop_005.py           # 循环流程（状态驱动的循环）
│   ├── langchain_dynamic_model_006.py  # 动态模型切换
│   ├── langgrap_test_007.py            # LangGraph 高级特性（回放/分叉/时间旅行）
│   ├── langgrap_test_008.py            # 并行笑话生成（Send 并行分发）
│   ├── langchain_memory_demo_009.py    # LangGraph 记忆演示（基础）
│   ├── langchain_memory_demo_010.py    # LangGraph 记忆演示（跨对话共享）
│   ├── langchain_memory_mongodb_011.py # MongoDB 持久化记忆
│   ├── langchain_memory_todo_012.py    # Todo 记忆管理
│   ├── langgraph_human_waiting_input_013.py  # 人机交互（interrupt 等待人工输入）
│   ├── langgraph_runtime_014.py        # Runtime 配置
│   ├── langgraph_output_stream_015.py  # 流式输出（updates/values/debug/messages）
│   └── langgraph_codeing_demo_016.py   # 完整代码生成 Demo
│
├── .env                       # 环境变量配置
└── README.md
```

## 各模块详解

### 1. 模型基础 (models_001)
- 学习 LangChain 的基本模型调用方式
- 支持通义千问 (Tongyi) 和 OpenAI 兼容协议
- 演示通过 `ChatOpenAI` 和 `ChatTongyi` 两种方式调用千问模型

### 2. 消息与提示词 (messages_002)
- **提示词模板**：使用 `ChatPromptTemplate` 创建可复用的模板
- **消息类型**：`SystemMessage`、`HumanMessage`、`AIMessage` 三种消息格式
- **动态提示词**：根据用户角色（初学者/专家）动态调整系统提示词

### 3. 输出解析 (content_003)
- `StrOutputParser`：提取纯文本输出
- `JsonOutputParser`：解析 JSON 格式输出
- `PydanticOutputParser`：结构化输出到 Pydantic 模型

### 4. 链式调用 (chain-004)
- **LCEL (LangChain Expression Language)**：使用 `|` 运算符串联链
- **RunnableLambda**：将自定义函数融入链中
- **数据流转**：链中各节点数据的自动传递和转换

### 5. 工具与Agent (tools_005)
- **@tool 装饰器**：快速定义可调用工具
- **Agent 模式**：让大模型自主决定何时调用工具
- **回调系统**：通过 `BaseCallbackHandler` 追踪工具调用过程
- **复杂场景**：多工具协作、工具间依赖、错误处理

### 6. 中间件 (middleware_006)
- **SummarizationMiddleware**：自动压缩长对话，降低 token 消耗
- **ModelFallbackMiddleware**：主模型失败时自动切换备用模型
- **PIIMiddleware**：敏感信息（身份证号等）自动脱敏
- **ToolRetryMiddleware**：工具调用失败自动重试
- **ToolCallLimitMiddleware**：限制工具调用次数，防止无限循环

### 7. 记忆系统 (memory_007)
- **短期记忆**：`InMemorySaver` 基于线程 ID 的会话记忆
- **长期记忆**：`InMemoryStore` 跨对话的用户记忆
- **记忆管理**：消息裁剪、摘要压缩、自动清理
- **自定义记忆**：通过工具实现记忆的读写操作

### 8. LangGraph 图编排 (langgraph_008)
- **基础图**：`StateGraph` 状态图定义
- **流程控制**：串行、并行、条件分支、循环
- **人机交互**：`interrupt` 暂停等待人工输入
- **时间旅行**：回放历史状态、修改状态后重新执行
- **持久化**：`MemorySaver` / `MongoDBSaver` 检查点存储
- **流式输出**：`updates` / `values` / `debug` / `messages` 四种模式

## 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-openai langchain-community langgraph python-dotenv
pip install pymongo  # 如需 MongoDB 持久化
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# 通义千问 API
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_TEXT_MODEL=qwen3.6-plus

# MIMO API（小米大模型）
MIMO_API_KEY=your_mimo_api_key
MIMO_BASE_URL=your_mimo_base_url
MIMO_TEXT_MODEL=your_mimo_model
```

### 3. 运行示例

```bash
# 运行模型基础示例
python models_001/001_hello_langchain.py

# 运行 Agent 示例
python tools_005/tools_001.py

# 运行 LangGraph 示例
python langgraph_008/hello_langgraph_001.py
```

## 学习路径建议

1. **入门**：`models_001` → `messages_002` → `content_003`
2. **进阶**：`chain-004` → `tools_005` → `middleware_006`
3. **高级**：`memory_007` → `langgraph_008`

## 核心概念速查

| 概念 | 说明 | 示例文件 |
|------|------|---------|
| LCEL | LangChain 表达式语言，用 `|` 串联链 | `chain_001.py` |
| Agent | 大模型自主决策调用工具 | `tools_001.py` |
| Middleware | 在 Agent 执行前后插入逻辑 | `middle_001.py` |
| Checkpoint | 保存/恢复执行状态 | `memory_002.py` |
| StateGraph | LangGraph 的状态图编排 | `hello_langgraph_001.py` |
| interrupt | 暂停流程等待人工输入 | `langgraph_human_waiting_input_013.py` |
| Send | 并行分发任务到多个节点 | `langgrap_test_008.py` |

## 注意事项

- 本项目使用通义千问和小米 MIMO 的 API，请确保 API Key 配置正确
- 部分功能需要 MongoDB 支持（如持久化记忆），请先启动 MongoDB 服务
- `.env` 文件包含敏感信息，请勿提交到版本控制系统
- 代码注释主要使用中文，便于中文开发者理解

## 许可证

本项目仅供学习参考使用。
