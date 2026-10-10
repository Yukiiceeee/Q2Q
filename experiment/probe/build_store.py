"""Memorize a dataset slice into a persistent Q2Q store, once.

The evaluation adapter builds its ChromaDB store inside a per-process temp
directory (``methods/q2q_agent.py``: ``storage.chromadb_path`` is joined onto
``mkdtemp``, and the collection is named ``q2q_<pid>``), so the store dies
with the process and every run -- including every parameter sweep -- pays a
full re-memorize. Memorization dominates: for LoCoMo it is minutes of LLM
calls and hundreds of thousands of tokens, against a retrieval phase that is
milliseconds of local embedding work.

This script does the memorize half once and keeps the result:

    python -m experiment.probe.build_store --dataset locomo --out data/probe/locomo_small

What it writes, all under the output directory:

* ``chromadb/``            -- the persistent store, collection ``q2q_probe``
* ``manifest.json``        -- one record per chunk: session ids, original
                              text, compressed text, store id, token counts
* ``queries.json``         -- every evaluated query with its gold answer,
                              gold evidence ``dia_id``s, category, and (for
                              retrieval scoring) the chunk each evidence turn
                              lives in
* ``meta.json``            -- dataset, config, and LLM usage summary

Retrieval probes then reopen the same directory. No LLM call is made here
beyond memorization itself, and the stored text is what the real harness
would have stored, because the chunking is mirrored in ``locomo_data``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import Q2QAgent  # noqa: E402
from src.config import Q2QConfig  # noqa: E402
from src.utils.llm_client import get_usage_tracker  # noqa: E402
from experiment.probe import locomo_data  # noqa: E402

logger = logging.getLogger("probe.build")

DEFAULT_EMBEDDING = "<MODELS_DIR>/bge-small-en-v1.5"
DEFAULT_COMPRESSOR = "<MODELS_DIR>/Qwen2.5-0.5B-Instruct"
DEFAULT_NLP_MODEL = "<MODELS_DIR>/flan-t5-large"

# locomo_small.yaml: evaluation.memory_chunk_size
LOCOMO_CHUNK_SIZE = 10240


def build_config(args: argparse.Namespace, store_dir: Path) -> Q2QConfig:
    config = Q2QConfig()
    config.llm.provider = args.provider
    config.llm.model = args.model
    config.llm.temperature = args.temperature
    config.llm.max_tokens = args.max_tokens
    config.llm.api_key = os.getenv("OPENAI_API_KEY", "")
    config.llm.base_url = os.getenv("OPENAI_BASE_URL", "")

    config.embedding.provider = "local"
    config.embedding.model_name = args.embedding_model
    config.embedding.device = args.device

    config.preprocessing.proposition_mode = "llm"
    config.preprocessing.compressor_model = args.compressor_model
    config.preprocessing.compressor_device = args.device
    config.preprocessing.perplexity_threshold = args.perplexity_threshold
    config.preprocessing.nlp_inference_model = args.nlp_model

    config.retrieval.num_fake_queries = args.num_fake_queries
    config.retrieval.binding_verify_threshold = 0.8
    config.retrieval.fq_style = args.fq_style
    config.retrieval.fq_role = args.fq_role

    config.graph.evolve_threshold = args.evolve_threshold
    config.graph.diverge_low = args.diverge_low
    config.graph.max_parents = args.max_parents
    config.graph.similarity_top_k = args.similarity_top_k
    config.graph.maintenance_interval = args.maintenance_interval
    config.graph.merge_overlap_threshold = args.merge_overlap_threshold
    config.graph.max_chain_length = args.max_chain_length
    config.graph.chain_sample_interval = args.chain_sample_interval
    config.graph.max_chain_nodes = args.max_chain_nodes

    config.storage.backend = "chromadb"
    config.storage.chromadb_path = str(store_dir / "chromadb")
    config.storage.chromadb_collection = args.collection

    config.language = args.language
    config.log.level = "WARNING"
    return config


async def memorize_chunks(
    agent: Q2QAgent, chunks: list[locomo_data.Chunk]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        started = time.time()
        entry = await agent.memorize(chunk.text)
        chunk.q2q_session_id = entry.session_id
        record = {
            "chunk_index": chunk.chunk_index,
            "session_ids": chunk.session_ids,
            "q2q_session_id": entry.session_id,
            "input_chars": len(chunk.text),
            "compressed_chars": len(entry.session_text or ""),
            "num_propositions": len(entry.propositions or []),
            "num_fake_queries": len(entry.fake_queries or []),
            "elapsed_s": round(time.time() - started, 2),
            "text": chunk.text,
            "compressed_text": entry.session_text or "",
        }
        records.append(record)
        logger.warning(
            "chunk %d/%d: sessions=%s %d->%d chars, %d props, %d fqs, %.1fs",
            chunk.chunk_index + 1,
            len(chunks),
            chunk.session_ids,
            record["input_chars"],
            record["compressed_chars"],
            record["num_propositions"],
            record["num_fake_queries"],
            record["elapsed_s"],
        )
    return records


def write_queries(
    path: Path,
    sample: locomo_data.Sample,
    chunk_records: list[dict[str, Any]],
) -> None:
    """Persist each query with everything a retrieval probe needs.

    ``evidence_chunks`` is the key field: it maps gold ``dia_id``s through
    ``dia_id -> session_id -> chunk_index``, giving a *literal retrieval
    target* that needs no LLM to score.
    """
    turn_index = sample.turn_index()
    session_of_turn = sample.session_of_turn()
    chunk_of_session: dict[int, int] = {}
    for record in chunk_records:
        for session_id in record["session_ids"]:
            chunk_of_session[session_id] = record["chunk_index"]

    queries = []
    for query in sample.queries:
        evidence_chunks = sorted(
            {
                chunk_of_session[session_of_turn[dia_id]]
                for dia_id in query.evidence
                if dia_id in session_of_turn
                and session_of_turn[dia_id] in chunk_of_session
            }
        )
        queries.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "category": query.category,
                "category_name": query.category_name,
                "answer": query.answer,
                "evidence": query.evidence,
                "evidence_chunks": evidence_chunks,
                "evidence_texts": [
                    turn_index[dia_id].render()
                    for dia_id in query.evidence
                    if dia_id in turn_index
                ],
            }
        )

    path.write_text(
        json.dumps(
            {"sample_id": sample.sample_id, "queries": queries},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="locomo", choices=["locomo"])
    parser.add_argument("--out", required=True, help="output directory for the store")
    parser.add_argument("--data-file", default=locomo_data.DEFAULT_DATA_FILE)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--sample-ids", default="", help="comma-separated sample ids")
    parser.add_argument("--chunk-size", type=int, default=LOCOMO_CHUNK_SIZE)

    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=40960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default="en")

    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--compressor-model", default=DEFAULT_COMPRESSOR)
    parser.add_argument("--nlp-model", default=DEFAULT_NLP_MODEL)
    parser.add_argument("--perplexity-threshold", type=float, default=2.5)

    parser.add_argument("--num-fake-queries", type=int, default=15)
    parser.add_argument("--fq-style", default="formal")
    parser.add_argument("--fq-role", default="user")

    parser.add_argument("--evolve-threshold", type=float, default=0.6)
    parser.add_argument("--diverge-low", type=float, default=0.2)
    parser.add_argument("--max-parents", type=int, default=3)
    parser.add_argument("--similarity-top-k", type=int, default=5)
    parser.add_argument("--maintenance-interval", type=int, default=10)
    parser.add_argument("--merge-overlap-threshold", type=float, default=0.9)
    parser.add_argument("--max-chain-length", type=int, default=5)
    parser.add_argument("--chain-sample-interval", type=int, default=2)
    parser.add_argument("--max-chain-nodes", type=int, default=60)
    parser.add_argument("--collection", default="q2q_probe")
    parser.add_argument("--reset", action="store_true", help="delete an existing store first")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
    )

    out_dir = Path(args.out).resolve()
    if args.reset and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [s for s in args.sample_ids.split(",") if s] or None
    samples = locomo_data.load_samples(
        args.data_file, max_samples=args.max_samples, sample_ids=sample_ids
    )
    if not samples:
        logger.error("no samples loaded")
        return 1
    sample = samples[0]

    chunks = locomo_data.split_into_chunks(sample.sessions, args.chunk_size)
    logger.warning(
        "sample=%s sessions=%d queries=%d chunks=%d",
        sample.sample_id,
        len(sample.sessions),
        len(sample.queries),
        len(chunks),
    )

    config = build_config(args, out_dir)
    agent = Q2QAgent(config)
    usage = get_usage_tracker()
    usage.set_phase("memorize")

    started = time.time()
    records = asyncio.run(memorize_chunks(agent, chunks))
    elapsed = time.time() - started

    (out_dir / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_queries(out_dir / "queries.json", sample, records)

    stats = usage.get_stats()
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "data_file": args.data_file,
                "sample_id": sample.sample_id,
                "chunk_size": args.chunk_size,
                "collection": args.collection,
                "num_chunks": len(chunks),
                "num_queries": len(sample.queries),
                "elapsed_s": round(elapsed, 1),
                "llm_usage": stats,
                "memorize_params": {
                    "num_fake_queries": args.num_fake_queries,
                    "perplexity_threshold": args.perplexity_threshold,
                    "fq_style": args.fq_style,
                    "fq_role": args.fq_role,
                    "evolve_threshold": args.evolve_threshold,
                    "diverge_low": args.diverge_low,
                    "max_parents": args.max_parents,
                    "similarity_top_k": args.similarity_top_k,
                    "maintenance_interval": args.maintenance_interval,
                    "merge_overlap_threshold": args.merge_overlap_threshold,
                    "max_chain_length": args.max_chain_length,
                    "chain_sample_interval": args.chain_sample_interval,
                    "max_chain_nodes": args.max_chain_nodes,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_fqs = sum(r["num_fake_queries"] for r in records)
    total_props = sum(r["num_propositions"] for r in records)
    total_in = sum(r["input_chars"] for r in records)
    total_out = sum(r["compressed_chars"] for r in records)
    logger.warning(
        "done: %d fqs, %d props, %d->%d chars (%.1f%%), %.1fs",
        total_fqs,
        total_props,
        total_in,
        total_out,
        100.0 * total_out / max(total_in, 1),
        elapsed,
    )
    print(f"store: {out_dir}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"llm_usage: {json.dumps(stats, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
