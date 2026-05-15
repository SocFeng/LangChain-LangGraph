# 创建数据结构
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


class State(TypedDict):
    name: str
    age: int


def step_one_for_name(state: State):
    print(state)
    df_name = "张"
    return {
        "name": df_name,
    }
def step_two_for_name(state: State):
    first_name = state["name"]
    second_name = f"{first_name} 三三"
    return {
        "name": second_name,
    }


def step_three_for_age(state: State):
    if state["name"] == "张三三":
        return {
            "age": 18,
        }

    return {
        "age": 20,
    }


# 创建一个图
gp = StateGraph(State)

# 添加节点
gp.add_node("step_one_for_name", step_one_for_name)
gp.add_node("step_two_for_name", step_two_for_name)
gp.add_node("step_three_for_age", step_three_for_age)

# 设计边的信息
gp.add_edge(START, "step_one_for_name")
gp.add_edge("step_one_for_name", "step_two_for_name")
gp.add_edge("step_two_for_name", "step_three_for_age")
gp.add_edge("step_three_for_age", END)

gpb = gp.compile()
resp = gpb.invoke({"hi":"你好"})

print(resp)
