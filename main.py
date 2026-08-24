"""Q2Q Agent Memory System - Entry Point"""

import argparse
import asyncio
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
    if args.prop_mode:
        config.preprocessing.proposition_mode = args.prop_mode
    return config


async def cmd_interactive(agent: Q2QAgent, args: argparse.Namespace) -> None:
    print("=" * 60)
    print("  Q2Q Agent Memory System - Interactive Mode")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /memorize  - Enter memorize mode")
    print("  /query     - Enter query mode")
    print("  /stats     - Show memory statistics")
    print("  /clear     - Clear all memories")
    print("  /quit      - Exit")
    print()
    print(f"Config: alpha={agent.config.retrieval.alpha}, "
          f"top_k={agent.config.retrieval.top_k_per_sub}, "
          f"top_n={agent.config.retrieval.top_n}, "
          f"num_fq={agent.config.retrieval.num_fake_queries}")
    print(f"Storage: {agent.config.storage.backend} | "
          f"Embedding: {agent.config.embedding.model_name} | "
          f"Prop: {agent.config.preprocessing.proposition_mode}")
    print("=" * 60)
    print()

    loop = asyncio.get_event_loop()

    while True:
        try:
            user_input = await loop.run_in_executor(None, lambda: input(">>> ").strip())
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            print("Bye!")
            break

        elif user_input == "/stats":
            stats = agent.get_stats()
            count = await agent.memory_store.count()
            stats["memory_count"] = count
            print(json.dumps(stats, indent=2, ensure_ascii=False))

        elif user_input == "/clear":
            count = await agent.memory_store.count()
            confirm = await loop.run_in_executor(
                None, lambda: input(f"  Confirm clear {count} memories? (y/N): ").strip().lower()
            )
            if confirm == "y":
                await agent.memory_store.clear()
                print(f"  Cleared {count} memories.")
            else:
                print("  Cancelled.")

        elif user_input == "/memorize":
            print("  [Enter session text, end with an empty line]")
            lines = []
            while True:
                try:
                    line = await loop.run_in_executor(None, lambda: input("  ... "))
                except (EOFError, KeyboardInterrupt):
                    break
                if line == "":
                    break
                lines.append(line)

            if lines:
                text = "\n".join(lines)
                print(f"  Memorizing ({len(text)} chars)...")
                entry = await agent.memorize(text)
                print(f"  Done! session_id={entry.session_id}")
                print(f"  Propositions: {len(entry.propositions)}")
                print(f"  Fake queries: {len(entry.fake_queries)}")
                for fq in entry.fake_queries[:5]:
                    print(f"    - {fq.text} -> {fq.answer_source}")
            else:
                print("  Empty input, skipped.")

        elif user_input.startswith("/query "):
            query_text = user_input[7:].strip()
            if query_text:
                await _run_query(agent, query_text)

        elif user_input == "/query":
            q = await loop.run_in_executor(None, lambda: input("  Query: ").strip())
            if q:
                await _run_query(agent, q)

        else:
            await _run_query(agent, user_input)


async def _run_query(agent: Q2QAgent, query_text: str) -> None:
    print(f"\n  Querying: {query_text}")
    print("  " + "-" * 50)

    result = await agent.query(raw_query=query_text, return_answer=True)

    print(f"  Sub-queries ({len(result['sub_queries'])}):")
    for sq in result["sub_queries"]:
        print(f"    - {sq}")

    print(f"\n  Matched FQs: {result['num_fqs']}")
    for fq in result["matched_fqs"][:3]:
        print(f"    FQ: {fq['text'][:80]}")

    print(f"\n  Top Propositions: {result['num_propositions']}")
    for p in result["top_propositions"][:5]:
        print(f"    [{p['prop_id']}] {p['text'][:80]}")

    if result.get("answer"):
        print(f"\n  {'=' * 50}")
        print(f"  Answer:\n  {result['answer']}")
    print()


async def cmd_memorize(agent: Q2QAgent, args: argparse.Namespace) -> None:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Error: provide --text or --file")
        sys.exit(1)

    entry = await agent.memorize(text)
    print(f"\nMemory created: {entry.session_id}")
    print(f"  Propositions: {len(entry.propositions)}")
    print(f"  Fake queries: {len(entry.fake_queries)}")
    for fq in entry.fake_queries[:5]:
        print(f"    - {fq.text} -> {fq.answer_source}")


async def cmd_query(agent: Q2QAgent, args: argparse.Namespace) -> None:
    result = await agent.query(
        raw_query=args.query,
        history=args.history or "",
        return_answer=not args.no_answer,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


async def cmd_stats(agent: Q2QAgent, _args: argparse.Namespace) -> None:
    stats = agent.get_stats()
    count = await agent.memory_store.count()
    stats["memory_count"] = count
    print(json.dumps(stats, indent=2, ensure_ascii=False))


async def cmd_clear(agent: Q2QAgent, _args: argparse.Namespace) -> None:
    count = await agent.memory_store.count()
    await agent.memory_store.clear()
    print(f"Cleared {count} memory entries")


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Q2Q Agent Memory System")

    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--num-fake-queries", type=int, default=None)
    parser.add_argument("--prop-mode", type=str, default=None, choices=["llm", "nlp"])

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("interactive")

    p_mem = subparsers.add_parser("memorize")
    p_mem.add_argument("--text", type=str)
    p_mem.add_argument("--file", type=str)

    p_query = subparsers.add_parser("query")
    p_query.add_argument("query", type=str)
    p_query.add_argument("--history", type=str, default="")
    p_query.add_argument("--no-answer", action="store_true")

    subparsers.add_parser("stats")
    subparsers.add_parser("clear")

    args = parser.parse_args()

    config = Q2QConfig.from_env(args.env)
    config = apply_cli_overrides(config, args)
    setup_logger("q2q", config.log.level, config.log.log_dir)

    agent = Q2QAgent(config)

    if not args.command or args.command == "interactive":
        await cmd_interactive(agent, args)
    else:
        commands = {
            "memorize": cmd_memorize,
            "query": cmd_query,
            "stats": cmd_stats,
            "clear": cmd_clear,
        }
        await commands[args.command](agent, args)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
