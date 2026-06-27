from langgraph.graph import StateGraph, END
from src.edges import decide_to_end, decide_to_write
from src.nodes import (
    critic_node,
    formatter_node,
    grade_node,
    planner_node,
    retrieve_node,
    writer_node,
)
from src.state import AgentState

graph_builder = StateGraph(AgentState)

graph_builder.add_node("planner",   planner_node)
graph_builder.add_node("retrieve",  retrieve_node)
graph_builder.add_node("grade",     grade_node)
graph_builder.add_node("writer",    writer_node)
graph_builder.add_node("critic",    critic_node)
graph_builder.add_node("formatter", formatter_node)  

graph_builder.set_entry_point("planner")

graph_builder.add_edge("planner",  "retrieve")
graph_builder.add_edge("retrieve", "grade")
graph_builder.add_edge("writer",   "critic")
graph_builder.add_edge("formatter", END)              

graph_builder.add_conditional_edges(
    "grade",
    decide_to_write,
    {
        "re_plan":          "planner",
        "proceed_to_write": "writer",
    },
)

graph_builder.add_conditional_edges(
    "critic",
    decide_to_end,
    {
        "proceed_to_write": "writer",
        "end": "formatter",
    },
)

app = graph_builder.compile()