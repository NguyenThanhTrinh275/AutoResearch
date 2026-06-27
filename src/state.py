from typing import Annotated, TypedDict, List
import operator


class AgentState(TypedDict):
    # Core input
    topic: str

    # Search tracking
    search_queries: List[str]
    previous_queries: Annotated[List[str], operator.add]   # Accumulates across re-plan loops

    # Document pipeline
    documents: Annotated[List[dict], operator.add]          # Raw docs (deduped by retrieve_node)
    filtered_documents: List[dict]                          # Graded/approved docs

    # Writing pipeline
    draft: str
    critique: str
    critique_history: Annotated[List[str], operator.add]   # All past critiques for recurring-issue context
    is_perfect: bool

    # Loop counters
    loop_count: int         # Re-plan iterations
    loop_draft_count: int   # Writer revision iterations

    # Final output
    final_report: str       # Formatted report produced by formatter_node
