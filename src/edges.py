import logging

from src.state import AgentState

logger = logging.getLogger(__name__)

# Thresholds — can be overridden via environment variables (Phase 5 config)
import os
MIN_CONTENT_LENGTH = int(os.getenv("MIN_CONTENT_LENGTH", 5000))
MAX_REPLAN_LOOPS   = int(os.getenv("MAX_REPLAN_LOOPS", 3))
MAX_DRAFT_LOOPS    = int(os.getenv("MAX_DRAFT_LOOPS", 5))


def decide_to_write(state: AgentState) -> str:
    """Route after grade_node: re-plan if content is insufficient, else proceed to writer."""
    filtered_docs = state.get("filtered_documents", [])
    loop_count    = state.get("loop_count", 0)

    total_length = sum(len(doc.get("content", "")) for doc in filtered_docs)

    if total_length < MIN_CONTENT_LENGTH and loop_count < MAX_REPLAN_LOOPS:
        logger.info(
            "[ROUTE] Insufficient content (%d chars, threshold=%d). "
            "Re-planning with new queries (loop %d/%d).",
            total_length, MIN_CONTENT_LENGTH, loop_count, MAX_REPLAN_LOOPS
        )
        return "re_plan"

    if loop_count >= MAX_REPLAN_LOOPS:
        logger.warning(
            "[ROUTE] Re-plan limit reached (%d). Proceeding to writer with available content.",
            loop_count
        )
    else:
        logger.info(
            "[ROUTE] Content quality OK (%d chars). Proceeding to writer.", total_length
        )
    return "proceed_to_write"


def decide_to_end(state: AgentState) -> str:
    """Route after critic_node: format+end if approved, else send back to writer."""
    if state["is_perfect"]:
        logger.info("[ROUTE] Critic approved the draft. Moving to formatter.")
        return "end"

    if state.get("loop_draft_count", 0) >= MAX_DRAFT_LOOPS:
        logger.warning(
            "[ROUTE] Draft revision limit reached (%d). Moving to formatter with best draft.",
            MAX_DRAFT_LOOPS
        )
        return "end"

    logger.info("[ROUTE] Draft needs revision. Sending back to writer.")
    return "proceed_to_write"
