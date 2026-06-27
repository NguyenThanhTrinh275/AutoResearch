"""
main.py — AutoResearch entry point.

Usage:
    python main.py                        # interactive topic prompt
    python main.py --topic "Your Topic"   # CLI argument
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Load .env before importing any langchain/src modules
load_dotenv()

# ---------------------------------------------------------------------------
# Logging — configured once here; all modules inherit via logging.getLogger()
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console
    ],
)
logger = logging.getLogger("main")

# Import graph after dotenv + logging are set up
from src.graph import app  # noqa: E402


def get_topic() -> str:
    """Resolve the research topic from CLI arg or interactive prompt."""
    parser = argparse.ArgumentParser(
        description="AutoResearch — AI-powered autonomous research tool"
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default=None,
        help="The research topic (e.g. 'Graph Neural Networks in community detection')",
    )
    args = parser.parse_args()

    if args.topic:
        return args.topic.strip()

    # Interactive fallback
    print("\n" + "=" * 60)
    print("   🔬  AutoResearch — AI Research Assistant")
    print("=" * 60)
    topic = input("Enter research topic: ").strip()
    if not topic:
        logger.error("No topic provided. Exiting.")
        sys.exit(1)
    return topic


def main() -> None:
    topic = get_topic()

    print("\n" + "=" * 60)
    print("🚀  STARTING AutoResearch")
    print(f"📌  Topic: {topic}")
    print("=" * 60 + "\n")

    initial_state = {
        "topic":             topic,
        "search_queries":    [],
        "previous_queries":  [],   # Accumulates across re-plan loops
        "documents":         [],
        "filtered_documents": [],
        "draft":             "",
        "critique":          "",
        "critique_history":  [],   # Accumulates for writer context
        "is_perfect":        False,
        "loop_count":        0,
        "loop_draft_count":  0,
        "final_report":      "",
    }

    final_report = ""

    try:
        for output in app.stream(initial_state, config={"recursion_limit": 20}):
            for node_name, state_update in output.items():
                print(f"\n✅  [{node_name.upper()}] completed")

                if "search_queries" in state_update:
                    print(f"   🎯  Queries: {state_update['search_queries']}")

                if "documents" in state_update:
                    print(f"   🌐  Retrieved {len(state_update['documents'])} unique docs")

                if "filtered_documents" in state_update:
                    print(f"   🛡️   Kept {len(state_update['filtered_documents'])} docs after grading")

                if "draft" in state_update:
                    length = len(state_update["draft"])
                    version = state_update.get("loop_draft_count", "?")
                    print(f"   📝  Draft v{version} written ({length} chars)")

                if "is_perfect" in state_update:
                    verdict = "✔ PASSED" if state_update["is_perfect"] else "✘ FAILED"
                    print(f"   🔍  Critic verdict: {verdict}")
                    if not state_update["is_perfect"] and state_update.get("critique"):
                        print(f"       Feedback: {state_update['critique'][:200]}...")

                if "final_report" in state_update and state_update["final_report"]:
                    final_report = state_update["final_report"]

    except Exception as exc:
        logger.exception("Graph execution failed: %s", exc)
        sys.exit(1)

    # Display final report
    print("\n" + "=" * 60)
    print("🎉  RESEARCH COMPLETE — FINAL REPORT")
    print("=" * 60 + "\n")

    if final_report:
        print(final_report)
    else:
        logger.warning("No final report was generated. Check graph execution logs.")

    # Remind where files were saved
    output_dir = os.path.abspath("output")
    print(f"\n📁  All files saved to: {output_dir}")


if __name__ == "__main__":
    main()