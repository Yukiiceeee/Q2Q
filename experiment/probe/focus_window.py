"""Check that relevance-guided windowing reaches facts a prefix cut discards.

Drives Answerer._focus_to_tokens with real AMA-Bench WEB questions and asks the
only question that matters for recall scoring: does the element id quoted by the
gold answer survive into the budgeted context? Compares against the prefix cut
the answerer used before. No LLM calls -- the raw question stands in for its
sub-queries so only the windowing is measured.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import Q2QConfig
from src.embedding.factory import create_embedding_provider
from src.retrieval.answerer import Answerer
from src.schemas.models import SubQuery

AMA = "<DATA_DIR>/AMA-Hub/dataset/test/open_end_qa_set.jsonl"
BUDGET = int(os.getenv("PROBE_BUDGET", "10000"))
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ID = re.compile(r"\[\d+\]")


def trajectory_to_text(trajectory: list[dict]) -> str:
    lines = []
    for turn in trajectory:
        lines.append(f"Step {turn.get('turn_idx', 0)}:")
        lines.append(f"Action: {turn.get('action', '')}")
        lines.append(f"Observation: {turn.get('observation', '')}")
        lines.append("")
    return "\n".join(lines)


class _NoLLM:
    model = "gpt-4o-mini"


async def main() -> None:
    config = Q2QConfig.from_env()
    provider = create_embedding_provider(
        provider=config.embedding.provider,
        model_name=config.embedding.model_name,
        device=config.embedding.device,
    )
    answerer = Answerer(
        llm_client=_NoLLM(),
        language="en",
        max_context_tokens=BUDGET,
        embedding_provider=provider,
    )

    episodes = [json.loads(line) for line in open(AMA, encoding="utf-8")]
    web = [e for e in episodes if e.get("domain") == "WEB"]
    web.sort(key=lambda e: e["episode_id"])
    picked = web[:N_EP]

    prefix_hits = focus_hits = total = 0
    for episode in picked:
        text = trajectory_to_text(episode["trajectory"])
        prefix = answerer._truncate_to_tokens(text, BUDGET)
        n_tok = answerer._count_tokens(text)
        for qa in episode["qa_pairs"]:
            ids = {i for i in ID.findall(str(qa.get("answer", ""))) if i in text}
            if not ids:
                continue
            sq = SubQuery(index=0, text=qa["question"])
            sq.embedding = (await provider.embed_batch([qa["question"]]))[0]
            focused = await answerer._focus_to_tokens(text, BUDGET, [sq])
            p = sum(i in prefix for i in ids)
            f = sum(i in focused for i in ids)
            prefix_hits += p
            focus_hits += f
            total += len(ids)
            if f != p:
                print(f"  ep{episode['episode_id']} q: ids={len(ids)} prefix={p} focus={f}")
        print(f"ep {episode['episode_id']}: {n_tok} tokens -> budget {BUDGET}")

    print()
    print(f"gold element-ids present in context ({total} total):")
    print(f"  prefix cut        {prefix_hits:3d}  {prefix_hits/max(total,1):.1%}")
    print(f"  focused window    {focus_hits:3d}  {focus_hits/max(total,1):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
