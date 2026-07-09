"""Q2Q Agent Memory System - Entry Point

Configuration loading logic:
1. .env → environment variables (API keys, model paths, storage paths)
2. CLI args → runtime parameters (alpha, top_k, storage backend, language, etc.)
3. Defaults → for any parameter not explicitly provided

Default mode: interactive Q&A (memorize + query in a streaming loop)
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Q2QConfig
from src.utils.logger import setup_logger
from agent import Q2QAgent


def apply_cli_overrides(config: Q2QConfig, args: argparse.Namespace) -> Q2QConfig:
    if args.log_level:
        config.log.level = args.log_level
    if args.storage:
        config.storage.backend = args.storage
    if args.language:
        config.language = args.language
    if args.alpha is not None:
        config.retrieval.alpha = args.alpha
    if args.top_k is not None:
        config.retrieval.top_k_per_sub = args.top_k
    if args.top_n is not None:
        config.retrieval.top_n = args.top_n
    if args.num_fake_queries is not None:
        config.retrieval.num_fake_queries = args.num_fake_queries
    return config


def cmd_interactive(agent: Q2QAgent, args: argparse.Namespace) -> None:
    """Interactive streaming Q&A mode."""
    print("=" * 60)
    print("  Q2Q Agent Memory System - Interactive Mode")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /memorize  - Enter memorize mode (input session text, end with empty line)")
    print("  /query     - Enter query mode (ask a question)")
    print("  /stats     - Show memory statistics")
    print("  /clear     - Clear all memories")
    print("  /quit      - Exit")
    print()
    print(f"Config: alpha={agent.config.retrieval.alpha}, "
          f"top_k={agent.config.retrieval.top_k_per_sub}, "
          f"top_n={agent.config.retrieval.top_n}, "
          f"num_fq={agent.config.retrieval.num_fake_queries}")
    print(f"Storage: {agent.config.storage.backend} | "
          f"Embedding: {agent.config.embedding.model_name}")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit" or user_input == "/exit":
            print("Bye!")
            break

        elif user_input == "/stats":
            stats = agent.get_stats()
            print(json.dumps(stats, indent=2, ensure_ascii=False))

        elif user_input == "/clear":
            count = agent.memory_store.count()
            confirm = input(f"  Confirm clear {count} memories? (y/N): ").strip().lower()
            if confirm == "y":
                agent.memory_store.clear()
                print(f"  Cleared {count} memories.")
            else:
                print("  Cancelled.")

        elif user_input == "/memorize":
            print("  [Enter session text, end with an empty line]")
            lines = []
            while True:
                try:
                    line = input("  ... ")
                except (EOFError, KeyboardInterrupt):
                    break
                if line == "":
                    break
                lines.append(line)

            if lines:
                text = "\n".join(lines)
                print(f"  Memorizing ({len(text)} chars)...")
                entry = agent.memorize(text)
                print(f"  Done! memory_id={entry.memory_id}")
                print(f"  Fake queries ({len(entry.fake_queries)}):")
                for fq in entry.fake_queries:
                    print(f"    - {fq.text}")
            else:
                print("  Empty input, skipped.")

        elif user_input.startswith("/query "):
            query_text = user_input[7:].strip()
            if query_text:
                _run_query(agent, query_text)
            else:
                print("  Usage: /query <your question>")

        elif user_input == "/query":
            q = input("  Query: ").strip()
            if q:
                _run_query(agent, q)

        else:
            # Default: treat as a query
            _run_query(agent, user_input)


def _run_query(agent: Q2QAgent, query_text: str) -> None:
    """Execute query and print results in streaming fashion."""
    print(f"\n  Querying: {query_text}")
    print("  " + "-" * 50)

    result = agent.query(raw_query=query_text, return_answer=True)

    # Sub-queries
    print(f"  Sub-queries ({len(result['sub_queries'])}):")
    for sq in result["sub_queries"]:
        print(f"    - {sq}")

    # Retrieval results
    print(f"\n  Retrieved memories ({result['num_results']}):")
    for r in result["results"][:5]:
        print(f"    [{r['final_score']:.4f}] q2q={r['score_q2q']:.4f} q2c={r['score_q2c']:.4f}")
        fq_preview = r['matched_fake_queries'][0][:80] if r.get('matched_fake_queries') else ""
        print(f"      FQ: {fq_preview}")
        print(f"      Preview: {r['content_preview'][:80]}...")

    # Answer
    if result.get("answer"):
        print(f"\n  {'=' * 50}")
        print(f"  Answer:\n")
        print(f"  {result['answer']}")

    print()


def cmd_memorize(agent: Q2QAgent, args: argparse.Namespace) -> None:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Error: provide --text or --file")
        sys.exit(1)

    entry = agent.memorize(text)
    print(f"\nMemory created: {entry.memory_id}")
    print(f"  Fake queries ({len(entry.fake_queries)}):")
    for fq in entry.fake_queries:
        print(f"    - {fq.text}")
    print(f"  Content chunks: {len(entry.content_embeddings)}")


def cmd_query(agent: Q2QAgent, args: argparse.Namespace) -> None:
    result = agent.query(
        raw_query=args.query,
        history=args.history or "",
        return_answer=not args.no_answer,
    )

    print(f"\nQuery: {result['raw_query']}")
    print(f"Sub-queries: {result['sub_queries']}")
    print(f"Results: {result['num_results']}")
    for r in result["results"]:
        print(f"\n  [{r['final_score']:.4f}] {r['memory_id']}")
        print(f"    Q2Q: {r['score_q2q']:.4f} | Q2C: {r['score_q2c']:.4f}")
        print(f"    Matched FQs: {r['matched_fake_queries'][:3]}")
        print(f"    Version Chains: {r.get('version_chains', 0)}")
        print(f"    Preview: {r['content_preview'][:100]}...")

    if result.get("answer"):
        print(f"\n--- Answer ---\n{result['answer']}")


def cmd_stats(agent: Q2QAgent, _args: argparse.Namespace) -> None:
    stats = agent.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def cmd_clear(agent: Q2QAgent, _args: argparse.Namespace) -> None:
    count = agent.memory_store.count()
    agent.memory_store.clear()
    print(f"Cleared {count} memory entries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2Q Agent Memory System")

    # Environment config
    parser.add_argument("--env", type=str, default=None, help="Path to .env file")

    # Runtime parameters (override defaults)
    parser.add_argument("--log-level", type=str, default=None, help="Log level (DEBUG/INFO/WARNING)")
    parser.add_argument("--storage", type=str, default=None, help="Storage backend (chromadb/json)")
    parser.add_argument("--language", type=str, default=None, help="Prompt language (zh/en)")
    parser.add_argument("--alpha", type=float, default=None, help="Q2Q weight alpha (0-1)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K per sub-query")
    parser.add_argument("--top-n", type=int, default=None, help="Top-N final results")
    parser.add_argument("--num-fake-queries", type=int, default=None, help="Fake queries per memory")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # interactive (default)
    subparsers.add_parser("interactive", help="Interactive Q&A mode (default)")

    # memorize
    p_mem = subparsers.add_parser("memorize", help="Store a session as memory")
    p_mem.add_argument("--text", type=str, help="Session text to memorize")
    p_mem.add_argument("--file", type=str, help="File containing session text")

    # query
    p_query = subparsers.add_parser("query", help="Query memories (single shot)")
    p_query.add_argument("query", type=str, help="Query string")
    p_query.add_argument("--history", type=str, default="", help="Conversation history")
    p_query.add_argument("--no-answer", action="store_true", help="Skip answer generation")

    # stats
    subparsers.add_parser("stats", help="Show statistics")

    # clear
    subparsers.add_parser("clear", help="Clear all memories")

    args = parser.parse_args()

    # Step 1: Load env config
    config = Q2QConfig.from_env(args.env)

    # Step 2: Apply CLI overrides
    config = apply_cli_overrides(config, args)

    # Step 3: Setup logging
    setup_logger("q2q", config.log.level, config.log.log_dir)

    # Step 4: Init agent
    agent = Q2QAgent(config)

    # Step 5: Dispatch command (default: interactive)
    if not args.command or args.command == "interactive":
        cmd_interactive(agent, args)
    else:
        commands = {
            "memorize": cmd_memorize,
            "query": cmd_query,
            "stats": cmd_stats,
            "clear": cmd_clear,
        }
        commands[args.command](agent, args)


if __name__ == "__main__":
    main()
