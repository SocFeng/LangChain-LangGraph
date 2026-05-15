import operator
from typing import Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


# ======================
# 1. 状态必须用 add_messages / operator.add 才能追加列表！
# ======================
class State(TypedDict):
    nodes: Annotated[list[str], operator.add]  # 🔥 修复：必须加 operator.add


# ======================
# 2. 节点函数：绝对不能用 append！
# ======================
def a(state: State):
    current_node = "A"
    print(f"将 {current_node} 节点添加到列表：{state['nodes']}")
    return {"nodes": [current_node]}  # ✅ 返回新列表，LangGraph 会自动追加

def b(state: State):
    current_node = "B"
    print(f"将 {current_node} 节点添加到列表：{state['nodes']}")
    return {"nodes": [current_node]}

def c(state: State):
    current_node = "C"
    print(f"将 {current_node} 节点添加到列表：{state['nodes']}")
    return {"nodes": [current_node]}

def d(state: State):
    current_node = "D"
    print(f"将 {current_node} 节点添加到列表：{state['nodes']}")
    return {"nodes": [current_node]}


# ======================
# 3. 创建图
# ======================
g = StateGraph(State)

g.add_node("a", a)
g.add_node("b", b)
g.add_node("c", c)
g.add_node("d", d)

# 边

g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("a", "c")
g.add_edge("b", "d")
g.add_edge("c", "d")
g.add_edge("d", END)  # 🔥 修复：不是 add_node，是 add_edge！

# 编译
gc = g.compile()

# 运行
resp = gc.invoke({"nodes": []})
print("\n✅ 最终结果：", resp)