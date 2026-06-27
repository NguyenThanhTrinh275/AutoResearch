"""
nodes.py — All LangGraph node functions for the AutoResearch pipeline.

Phase 1 fixes applied:
  - Error handling in retrieve_node and grade_node
  - Output persistence (drafts saved after every writer cycle)

Phase 2 improvements:
  - planner_node: aware of previous_queries to avoid duplicate searches on re-plan
  - grade_node: parallel execution via ThreadPoolExecutor (was sequential)
  - writer_llm: upgraded to gpt-4o-mini (was qwen2.5:7b) for consistent quality
  - writer_node: uses accumulated critique_history for context on recurring issues
  - critic_node: appends feedback to critique_history
  - formatter_node: new — adds metadata header, ToC, citations, and saves final report

"""

import concurrent.futures
import logging
import os
from datetime import datetime

from langchain_openai import ChatOpenAI

from src.schemas import CriticSchema, DocumentGrade, PlannerOutput
from src.state import AgentState
from src.tools import search_web #, search_web_fallback

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# from langchain_ollama import ChatOllama
# writer_llm = ChatOllama(model="qwen2.5:7b", temperature=0.5)
writer_llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

structured_planner = llm.with_structured_output(PlannerOutput)
structure_grade     = llm.with_structured_output(DocumentGrade)
structured_critic   = llm.with_structured_output(CriticSchema)


def planner_node(state: AgentState) -> dict:
    topic = state["topic"]
    current_loop = state.get("loop_count", 0)
    previous_queries = state.get("previous_queries", [])

    logger.info("=== PLANNER NODE | loop=%d | prev_queries=%d ===",
                current_loop, len(previous_queries))

    system_prompt = (
        "You are a research planning expert. "
        "Your task is to analyze the target topic and generate 2 to 3 of the best "
        "Google search queries to gather information and literature. "
        "The queries should be concise, cover different perspectives, and avoid repetition."
    )

    avoid_str = ""
    if previous_queries:
        avoid_str = "\n\nAvoid repeating these already-used queries:\n" + \
                    "\n".join(f"  - {q}" for q in previous_queries)

    user_prompt = f"Target Topic: {topic}{avoid_str}"

    response = structured_planner.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])

    logger.info("Generated queries: %s", response.queries)

    return {
        "search_queries":   response.queries,
        "previous_queries": response.queries,
        "loop_count":       current_loop + 1,
    }


def retrieve_node(state: AgentState) -> dict:
    seen_urls: set[str] = set()
    content: list[dict] = []

    for query in state["search_queries"]:
        response = search_web(query)
        raw_results = response.get("results", [])
        # try:
        #     response = search_web(query)
        #     raw_results = response.get("results", [])
        # except Exception as exc:
        #     logger.warning("[Tavily] Failed for '%s': %s — trying fallback...", query, exc)
        #     raw_results = search_web_fallback(query)

        for doc in raw_results:
            url = doc.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                content.append(doc)
            elif url in seen_urls:
                logger.debug("[Dedup] Skipped duplicate URL: %s", url)

    logger.info("Retrieved %d unique documents from %d queries.",
                len(content), len(state["search_queries"]))
    return {"documents": content}


_GRADE_SYSTEM_PROMPT = (
    "You are a document reviewer. "
    "Your task is to assess whether the provided text contains useful information "
    "or is directly relevant to the research topic. "
    "Be objective; only select 'yes' if it genuinely helps in writing the report."
)


def _grade_single_doc(doc: dict, topic: str) -> dict | None:
    """Grade one document; returns the doc if relevant, None otherwise."""
    user_prompt = f"Topic: {topic}\n\nDocument: {doc.get('content', '')}"
    try:
        result = structure_grade.invoke([
            {"role": "system", "content": _GRADE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ])
        if result.binary_score.lower().strip() == "yes":
            logger.info("  [APPROVED] %s", doc.get("title", "No Title"))
            return doc
        else:
            logger.info("  [REJECTED] %s", doc.get("title", "No Title"))
            return None
    except Exception as exc:
        logger.warning("  [GRADE ERROR] '%s': %s — skipping doc.",
                       doc.get("title", "No Title"), exc)
        return None


def grade_node(state: AgentState) -> dict:
    """Grade all documents in parallel (ThreadPoolExecutor) instead of sequentially."""
    raw_docs = state["documents"]
    topic    = state["topic"]

    logger.info("=== GRADE NODE | grading %d docs in parallel ===", len(raw_docs))

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures  = [executor.submit(_grade_single_doc, doc, topic) for doc in raw_docs]
        results  = [f.result() for f in concurrent.futures.as_completed(futures)]

    clean_docs = [r for r in results if r is not None]
    logger.info("Grading complete: kept %d / %d documents.", len(clean_docs), len(raw_docs))
    return {"filtered_documents": clean_docs}



def writer_node(state: AgentState) -> dict:
    """Write or revise the research draft. Uses critique_history for recurring-issue context."""
    topic            = state["topic"]
    docs             = state["filtered_documents"]
    current_draft    = state.get("draft", "")
    critique         = state.get("critique", "")
    critique_history = state.get("critique_history", [])
    loop_draft_count = state.get("loop_draft_count", 0)

    context_str = "\n\n".join([
        f"Source Title: {d.get('title', 'N/A')}\n"
        f"URL: {d.get('url', 'N/A')}\n"
        f"Content: {d.get('content', '')}"
        for d in docs
    ])

    if current_draft and critique:
        # Revision mode: surface prior critique history for recurring issues
        history_block = ""
        prior_critiques = critique_history[:-1]  # exclude the latest (already shown)
        if prior_critiques:
            history_block = (
                "\n\nPrior Critique History (watch for recurring issues):\n" +
                "\n---\n".join(prior_critiques)
            )

        system_prompt = (
            "You are a professional copyeditor and research specialist. Your task is to review the "
            "existing draft report, compare it against the peer-reviewer's critique, and systematically "
            "revise the text. Integrate any missing information from the provided source documents, fix "
            "inaccuracies, and refine the overall flow. "
            "Return the updated report in clean Markdown syntax."
        )
        user_prompt = (
            f"Target Topic: {topic}\n\n"
            f"Current Draft:\n{current_draft}\n\n"
            f"Reviewer's Critique:\n{critique}{history_block}\n\n"
            f"Original Source Documents for Reference:\n{context_str}"
        )

    else:
        # First-draft mode
        system_prompt = (
            "You are an expert research scientist and technical writer. Your task is to synthesize the "
            "provided reference documents into a comprehensive, deeply analytical, and structured research "
            "report on the given topic. Avoid superficial summaries; provide deep insights based strictly "
            "on the facts. "
            "The final output must be formatted entirely in clean Markdown syntax "
            "(using #, ##, ### headers, bullet points, and bold text for key concepts)."
        )
        user_prompt = f"Target Topic: {topic}\n\nReference Documents:\n{context_str}"

    response = writer_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])

    draft_version = loop_draft_count + 1
    logger.info("=== WRITER NODE | draft v%d | length=%d chars ===",
                draft_version, len(response.content))

    # Persist intermediate draft so no work is lost on crash
    _save_draft(response.content, version=draft_version)

    return {
        "draft":           response.content,
        "loop_draft_count": draft_version,
    }


def _save_draft(content: str, version: int) -> None:
    """Save an intermediate draft to the output/ directory."""
    os.makedirs("output", exist_ok=True)
    path = f"output/draft_v{version}.md"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("[✓] Draft saved → %s", path)
    except Exception as exc:
        logger.warning("Could not save draft: %s", exc)


def critic_node(state: AgentState) -> dict:
    """Peer-review the draft for hallucinations and omissions against source docs."""
    logger.info("=== CRITIC NODE ===")

    topic = state["topic"]
    draft = state["draft"]
    docs  = state["filtered_documents"]

    context_str = "\n\n".join([f"Source Content: {d.get('content', '')}" for d in docs])

    system_prompt = (
        "You are a rigorous academic peer-reviewer and expert fact-checker. Your task is to critically "
        "evaluate the provided draft report against the original source documents (context) to ensure "
        "maximum factual accuracy and completeness.\n\n"
        "Check for two critical issues:\n"
        "1. Hallucinations: Does the draft contain claims that CANNOT be verified from the source docs?\n"
        "2. Omissions: Did the draft miss vital insights present in the source docs?\n\n"
        "Be strict. Set 'is_perfect' to False for any issue; True only if fully accurate and complete."
    )
    user_prompt = (
        f"Target Topic: {topic}\n\n"
        f"Draft Report:\n{draft}\n\n"
        f"Original Source Documents:\n{context_str}"
    )

    response = structured_critic.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])

    if response.is_perfect:
        logger.info("-> [CRITIQUE]: PASSED — draft is accurate and complete.")
    else:
        logger.info("-> [CRITIQUE]: FAILED — needs revision.\n  Feedback: %s", response.feedback)

    return {
        "is_perfect":      response.is_perfect,
        "critique":        response.feedback,
        "critique_history": [response.feedback] if response.feedback else [],  # Accumulates
    }


def formatter_node(state: AgentState) -> dict:
    """Polish the approved draft into a final report with metadata, ToC, and citations."""
    logger.info("=== FORMATTER NODE ===")

    topic            = state["topic"]
    draft            = state["draft"]
    docs             = state["filtered_documents"]
    loop_count       = state.get("loop_count", 0)
    loop_draft_count = state.get("loop_draft_count", 0)

    # 1. Metadata header
    metadata_header = (
        f"---\n"
        f"**Topic**: {topic}  \n"
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
        f"**Sources**: {len(docs)} documents  \n"
        f"**Research loops**: {loop_count} | **Revision loops**: {loop_draft_count}\n\n"
        f"---\n\n"
    )

    # 2. Auto-generate Table of Contents from ## / ### headings
    toc_lines: list[str] = []
    for line in draft.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            anchor  = _to_anchor(heading)
            toc_lines.append(f"  - [{heading}](#{anchor})")
        elif line.startswith("### "):
            heading = line[4:].strip()
            anchor  = _to_anchor(heading)
            toc_lines.append(f"    - [{heading}](#{anchor})")

    toc_section = ""
    if toc_lines:
        toc_section = "## 📑 Table of Contents\n\n" + "\n".join(toc_lines) + "\n\n---\n\n"

    # 3. References section (numbered, with clickable links)
    ref_lines = [
        f"{i}. [{d.get('title', 'No Title')}]({d.get('url', '#')})"
        for i, d in enumerate(docs, 1)
    ]
    references_section = "\n\n---\n\n## 📚 References\n\n" + "\n".join(ref_lines)

    final_report = metadata_header + toc_section + draft + references_section

    # 4. Save final report
    os.makedirs("output", exist_ok=True)
    safe_topic = "".join(
        c if c.isalnum() or c in " _-" else "_" for c in topic
    )[:40].strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"output/report_{safe_topic}_{timestamp}.md"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_report)
        logger.info("[✓] Final report saved → %s", filename)
    except Exception as exc:
        logger.warning("Could not save final report: %s", exc)

    return {"final_report": final_report}


def _to_anchor(heading: str) -> str:
    """Convert a heading string to a GitHub Markdown anchor."""
    return (
        heading.lower()
        .replace(" ", "-")
        .replace("(", "").replace(")", "")
        .replace(",", "").replace(":", "")
        .replace("'", "").replace('"', "")
    )
