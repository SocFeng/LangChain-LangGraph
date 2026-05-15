from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

def func_replay():
    """回放"""

    class State(TypedDict):
        step: int
        message: str

    # ==============================
    # 流程节点（打印方式不变）
    # ==============================
    def step1(state: State):
        print("✅ 执行步骤 1：开始")
        return {"step": 1, "message": "步骤1完成"}

    def step2(state: State):
        print("✅ 执行步骤 2：处理中")
        return {"step": 2, "message": "步骤2完成"}

    def step3(state: State):
        print("✅ 执行步骤 3：完成")
        return {"step": 3, "message": "步骤3完成"}

    def step4(state: State):
        print("✅ 执行步骤 4：完成")
        return {"step": 4, "message": "步骤4完成"}

    def step5(state: State):
        print("✅ 执行步骤 5：完成")
        return {"step": 5, "message": "步骤5完成"}

    # ==============================
    # 构建流程图
    # ==============================
    builder = StateGraph(State)
    builder.add_node("step1", step1)
    builder.add_node("step2", step2)
    builder.add_node("step3", step3)
    builder.add_node("step4", step4)
    builder.add_node("step5", step5)

    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", "step3")
    builder.add_edge("step3", "step4")
    builder.add_edge("step4", "step5")
    builder.add_edge("step5", END)

    graph = builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "test123"}}

    # ==============================
    # 1. 先正常运行一遍（完整执行）
    # ==============================
    print("=" * 60)
    print("正常运行流程")
    print("=" * 60)
    graph.invoke({"step": 0, "message": "开始"}, config=config)

    # ==============================
    # 2. 获取历史检查点
    # ==============================
    history = list(memory.list(config))
    print(f"\n总历史数量：{len(history)}")

    # 找到 step4 执行完后的那个检查点（即 step 值为 4 的）
    target_checkpoint =  history[-4]  # ✅ 这是 step4 结束后的点！
    # for checkpoint_tuple in history:
    #     # 检查点内部存储的状态值
    #     if checkpoint_tuple.checkpoint["channel_values"].get("step") == 3:
    #         target_checkpoint = checkpoint_tuple
    #         break
    #
    # if target_checkpoint is None:
    #     print("未找到 step4 之后的检查点")
    #     return


    # 关键：正确获取 checkpoint_id（是 'id' 不是 'v'）
    checkpoint_id = target_checkpoint.checkpoint["id"]

    print(f"\n目标检查点 ID: {checkpoint_id} ")

    # ==============================
    # 3. 从 step4 之后恢复执行（只执行 step5）
    # ==============================
    print("\n" * 2)
    print("=" * 60)

    # 新 config：加上 checkpoint_id
    resume_config = {"configurable": {
        "thread_id": "test123",
        "checkpoint_id": checkpoint_id
    }}

    # 恢复执行，流式输出（打印方式不变）
    for event in graph.stream(None, config=resume_config):
        print("▶️ 回放结果:", event)



def func_branch():
    """分叉"""
    memory = MemorySaver()

    class WorkState(TypedDict):
        score: int
        name: str
        flag: str
        msg: str

    # 节点定义（不变）
    def node1_start(state: WorkState):
        print("🟢 节点1：任务开始")
        return {"msg": "节点1完成"}

    def node2_init(state: WorkState):
        print("🟡 节点2：初始化参数")
        return {"score": 60, "name": "考试评级", "msg": "节点2完成"}

    def node3_check(state: WorkState):
        print("⚙️  节点3：分数校验")
        return {"msg": "节点3完成"}

    def node4_fail(state: WorkState):
        print("🔴 节点4：不及格流程")
        return {"flag": "fail", "msg": "节点4完成"}

    def node5_pass(state: WorkState):
        print("🔵 节点5：及格流程")
        return {"flag": "pass", "msg": "节点5完成"}

    def node6_good(state: WorkState):
        print("🟣 节点6：评级-良好")
        return {"msg": "节点6完成"}

    def node7_excel(state: WorkState):
        print("🟠 节点7：评级-优秀 ✅ 新分支")
        return {"msg": "节点7完成"}

    def node8_record(state: WorkState):
        print("📒 节点8：录入档案")
        return {"msg": "节点8完成"}

    def node9_end(state: WorkState):
        print("🏁 节点9：结束")
        print(f"✅ 最终结果：分数={state['score']} | 等级={state['flag']}")
        return {"msg": "结束"}

    # 路由
    def route_pass_fail(state: WorkState) -> str:
        if state["score"] < 60:
            return "node4_fail"
        return "node5_pass"

    def route_good_excel(state: WorkState) -> str:
        print(f"🔍 当前路由判断分数：{state['score']}")
        if state["score"] <= 70:
            return "node6_good"
        return "node7_excel"

    # 构建图
    builder = StateGraph(WorkState)
    builder.add_node("node1_start", node1_start)
    builder.add_node("node2_init", node2_init)
    builder.add_node("node3_check", node3_check)
    builder.add_node("node4_fail", node4_fail)
    builder.add_node("node5_pass", node5_pass)
    builder.add_node("node6_good", node6_good)
    builder.add_node("node7_excel", node7_excel)
    builder.add_node("node8_record", node8_record)
    builder.add_node("node9_end", node9_end)

    builder.add_edge(START, "node1_start")
    builder.add_edge("node1_start", "node2_init")
    builder.add_edge("node2_init", "node3_check")
    builder.add_conditional_edges("node3_check", route_pass_fail)
    builder.add_conditional_edges("node5_pass", route_good_excel)
    builder.add_edge("node4_fail", "node8_record")
    builder.add_edge("node6_good", "node8_record")
    builder.add_edge("node7_excel", "node8_record")
    builder.add_edge("node8_record", "node9_end")
    builder.add_edge("node9_end", END)

    graph = builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "demo"}}

    # ================= 第一次运行（score=60 → 良好） =================
    print("=" * 70)
    print("第一次运行：score=60 → 良好分支")
    print("=" * 70)
    graph.invoke({}, config=config)

    # ================= 时间旅行：回到 node2 之后，修改 score=80 =================
    print("\n\n" + "=" * 70)
    print("⏪ 时间旅行：修改 score=80 → 强制走【优秀分支】")
    print("=" * 70)

    # 1. 获取所有历史状态（从旧到新）
    states = list(graph.get_state_history(config))

    target_state = None
    for s in states:
        if s.values.get("msg") == "节点5完成":
            target_state = s
            break

    # 3. 基于该 checkpoint 修改分数 → 会生成新的 checkpoint
    new_config = graph.update_state(target_state.config, {"score": 80}, as_node="node5_pass")

    # 4. 从新配置继续执行（只会从 node3 开始）
    print("🚀 从修改后的状态继续执行...")
    for _ in graph.stream(None, new_config):
        pass


if __name__ == "__main__":
    # func_replay()
    func_branch()
