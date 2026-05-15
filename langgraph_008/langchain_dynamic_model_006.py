import operator
from typing import Sequence

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict, Annotated

from common.model import llm_openai, mimo_openai


class State(TypedDict):
    messages: Annotated[Sequence[BaseModel], operator.add]


models = {
    "qwen": llm_openai,
    "mimo": mimo_openai
}


def change_model(state: State, config: RunnableConfig):
    model_name = config["configurable"].get("model", "qwen")
    model = models[model_name]
    resp = model.invoke(state["messages"])
    print(resp)
    return {"messages": [resp]}


# 创建图
g = StateGraph(State)
g.add_node("change_model", change_model)
g.add_edge(START, "change_model")
g.add_edge("change_model", END)
gp = g.compile()
# config = {"configurable": {"model": "mimo"}}
# config = {"configurable": {"model": "mimo"}}
gp.invoke({"messages": [HumanMessage(content="你是那个模型？直接回答！")]})
