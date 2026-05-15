from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


def func_human_input_base():
    checkpointer = MemorySaver()

    class State(TypedDict):
        input: str

    def step1(state: State):
        print("这是第一步11111")

    def step2(state: State):
        # 这里等待数据输入
        print("这是第二步22222,等待输入数据！")
        input_data = interrupt("在这里输入数据！")  # 等待输入，输入完后重新执行整个函数
        return {"input": input_data}

    def step3(state: State):
        print("这是第三步33333")

    gp = StateGraph(State)
    gp.add_node("step1", step1)
    gp.add_node("step2", step2)
    gp.add_node("step3", step3)
    gp.add_edge(START, "step1")
    gp.add_edge("step1", "step2")
    gp.add_edge("step2", "step3")
    gp.add_edge("step3", END)
    gc = gp.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "demo_1"}}

    print("============任务开始=============")
    gc.invoke({}, config=config)

    print("===========开始输入数据==========")

    gc.invoke(Command(resume="我现在输入数据了，接下来继续执行"), config=config)


def func_human_input_todo():
    # --------------------------
    # 1. 定义状态
    # --------------------------
    class AgentState(TypedDict):
        user_query: str  # 用户问题
        llm_reason: str  # 大模型判断是否需要调工具
        human_decision: str  # 人工决策：同意/拒绝
        tool_result: str  # 工具返回结果
        final_answer: str  # 最终大模型回答

    checkpointer = MemorySaver()

    # --------------------------
    # 2. 定义工具
    # --------------------------
    @tool
    def get_weather(location: str) -> str:
        """获取城市天气"""
        if location == "北京":
            return "北京：多云，25℃"
        elif location == "上海":
            return "上海：雷阵雨，23℃"
        return "未知城市天气"

    # --------------------------
    # 3. 节点1：大模型分析用户问题
    # --------------------------
    def llm_analyze(state: AgentState):
        # 模拟大模型：识别用户要查天气，需要调用工具
        user_q = state["user_query"]
        reason = f"分析：用户问题【{user_q}】需要调用天气工具查询"
        return {"llm_reason": reason}

    # --------------------------
    # 4. 节点2：人工输入/确认（关键人机交互）
    # --------------------------
    def human_confirm(state: AgentState):
        print("\n🤖 大模型分析结果：", state["llm_reason"])
        # 暂停，等待人工输入：同意 / 拒绝
        decision = interrupt("请人工决定是否调用天气工具？请输入：同意 / 拒绝")
        return {"human_decision": decision}

    # --------------------------
    # 5. 节点3：执行工具
    # --------------------------
    def call_tool_node(state: AgentState):
        # 固定查北京天气做演示
        res = get_weather.invoke("北京")
        return {"tool_result": res}

    # --------------------------
    # 6. 节点4：大模型生成最终回答
    # --------------------------
    def llm_final_answer(state: AgentState):
        if state["human_decision"] == "同意":
            ans = f"已调用工具查询完毕：{state['tool_result']}"
        else:
            ans = "已人工拒绝调用工具，结束对话"
        return {"final_answer": ans}

    # --------------------------
    # 7. 构建流程图
    # --------------------------
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("llm_analyze", llm_analyze)
    builder.add_node("human_confirm", human_confirm)
    builder.add_node("call_tool_node", call_tool_node)
    builder.add_node("llm_final_answer", llm_final_answer)

    # 流程连线
    builder.add_edge(START, "llm_analyze")
    builder.add_edge("llm_analyze", "human_confirm")

    # 条件分支：根据人工决策走不同分支
    def route_by_human(state: AgentState) -> str:
        if state["human_decision"] == "同意":
            return "call_tool_node"
        return "llm_final_answer"

    builder.add_conditional_edges("human_confirm", route_by_human)
    builder.add_edge("call_tool_node", "llm_final_answer")
    builder.add_edge("llm_final_answer", END)

    # 编译
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "agent_human_tool_001"}}

    print("===== 第一轮执行：大模型分析，等待人工确认 =====")
    graph.invoke({"user_query": "帮我查北京今天天气"}, config=config)

    # 第二步：人工输入决策，用Command恢复执行
    print("\n===== 人工输入：同意调用工具 =====")
    graph.invoke(Command(resume="同意"), config=config)

    # 查看最终结果
    final_state = graph.get_state(config).values
    print("\n===== 最终回答 =====")
    print(final_state["final_answer"])

def func_human_judge():
    # ==========================
    # 1. 复杂状态定义
    # ==========================
    class WorkflowState(TypedDict):
        user_query: str  # 用户问题
        llm_think: str  # 大模型思考
        tool_name: str  # 要调用的工具名
        tool_args: dict  # 工具参数
        human_review: str  # 人工审核结果
        tool_result: str  # 工具返回
        final_answer: str  # 最终回答

    checkpointer = MemorySaver()

    # ==========================
    # 2. 定义工具（模拟天气）
    # ==========================
    @tool
    def get_weather(location: str) -> str:
        """查询天气"""
        if location == "北京":
            return "北京：晴天 25°C"
        elif location == "上海":
            return "上海：雨天 23°C"
        return f"{location}：晴天 24°C"

    # ==========================
    # 3. 节点1：大模型思考
    # ==========================
    def llm_think_node(state: WorkflowState):
        print("=" * 60)
        print("🤖 LLM 思考中...")
        user_q = state["user_query"]

        # 模拟LLM判断：需要调用天气工具
        return {
            "llm_think": f"用户问题『{user_q}』需要查询天气",
            "tool_name": "get_weather",
            "tool_args": {"location": "北京"}  # LLM自动识别的参数
        }

    # ==========================
    # 4. 节点2：【核心审查节点】
    # ==========================
    def human_review_node(state: WorkflowState):
        print("\n" + "=" * 60)
        print("🔴 进入【人工审查节点】，流程已暂停")
        print(f"工具：{state['tool_name']}")
        print(f"参数：{state['tool_args']}")
        print("\n【操作指令】：")
        print("  通过 → 直接调用")
        print("  驳回 → 不调用")
        print("  修改:上海 → 手动改参数")

        # 中断，等待人工输入
        review_cmd = interrupt("请输入审核指令")
        return {"human_review": review_cmd}

    # ==========================
    # 5. 节点3：执行工具
    # ==========================
    def exec_tool_node(state: WorkflowState):
        print("\n" + "=" * 60)
        print("⚡ 执行工具调用...")

        args = state["tool_args"]
        result = get_weather.invoke(args)
        return {"tool_result": result}

    # ==========================
    # 6. 节点4：大模型生成最终答案
    # ==========================
    def llm_final_node(state: WorkflowState):
        review = state["human_review"]

        if review == "驳回":
            answer = "❌ 人工驳回，无法调用工具"
        else:
            answer = f"✅ 查询结果：{state['tool_result']}"

        return {"final_answer": answer}

    # ==========================
    # 7. 路由：根据人工审核分支
    # ==========================
    def review_router(state: WorkflowState):
        cmd = state["human_review"]

        if cmd == "通过":
            return "exec_tool_node"
        elif cmd == "驳回":
            return "llm_final_node"
        elif cmd.startswith("修改:"):
            # 提取城市
            city = cmd.replace("修改:", "").strip()
            state["tool_args"]["location"] = city
            return "exec_tool_node"
        else:
            return "llm_final_node"

    # ==========================
    # 8. 构建流程图
    # ==========================
    builder = StateGraph(WorkflowState)

    builder.add_node("llm_think_node", llm_think_node)
    builder.add_node("human_review_node", human_review_node)
    builder.add_node("exec_tool_node", exec_tool_node)
    builder.add_node("llm_final_node", llm_final_node)

    builder.add_edge(START, "llm_think_node")
    builder.add_edge("llm_think_node", "human_review_node")

    # 条件分支
    builder.add_conditional_edges(
        "human_review_node",
        review_router
    )

    builder.add_edge("exec_tool_node", "llm_final_node")
    builder.add_edge("llm_final_node", END)

    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "complex_review"}}

    # 第一步：执行到审查节点暂停
    graph.invoke({
        "user_query": "今天天气怎么样"
    }, config=config)

    # =====================
    # 人工选择操作（3选1）
    # =====================
    # 1. 通过
    # graph.invoke(Command(resume="通过"), config=config)

    # 2. 驳回
    # graph.invoke(Command(resume="驳回"), config=config)

    # 3. 修改参数（改成查询上海）
    graph.invoke(Command(resume="修改:上海"), config=config)

    # 输出最终结果
    final = graph.get_state(config).values
    print("\n" + "=" * 60)
    print("🎯 最终结果：", final["final_answer"])




if __name__ == "__main__":
    # func_human_input_base()
    # func_human_input_todo()
    func_human_judge()
