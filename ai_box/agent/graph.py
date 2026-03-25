from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from agent.nodes import (
    planner_node,
    rag_node,
    search_node,
    math_node,
    llm_node,
    date_node,
    code_node,
)


class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str


builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("rag",     rag_node)
builder.add_node("search",  search_node)
builder.add_node("math",    math_node)
builder.add_node("llm",     llm_node)
builder.add_node("date",    date_node)
builder.add_node("code",    code_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "planner",
    lambda state: state["next"],
    {
        "rag":    "rag",
        "search": "search",
        "math":   "math",
        "llm":    "llm",
        "date":   "date",
        "code":   "code",
    }
)

# Every tool node ends the graph
for node in ["rag", "search", "math", "llm", "date", "code"]:
    builder.add_edge(node, END)

graph = builder.compile()