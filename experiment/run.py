"""Q2Q Motivation Analysis Experiment - Main Entry Point.

Usage:
    python -m experiment.run --config configs/base.yaml --dataset configs/locomo.yaml --step generate
    python -m experiment.run --config configs/base.yaml --dataset configs/locomo.yaml --step embed
    python -m experiment.run --config configs/base.yaml --dataset configs/locomo.yaml --step exp1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

LOG_DIR = Path(PROJECT_ROOT) / "logs"


def setup_logging(level: str = "INFO", dataset_name: str = "", step: str = ""):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"exp_{dataset_name}_{step}_{timestamp}.log"
    log_path = LOG_DIR / log_filename

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File handler (detailed)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(fh)

    # Console handler (concise)
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(ch)

    logging.info(f"Log file: {log_path}")
    return log_path


async def step_generate(base_config: dict, dataset_config: dict) -> None:
    """Generate fake queries and indirect queries for all sessions/queries."""
    from experiment.src.data_loader import load_dataset
    from experiment.generation import ExperimentFQGenerator, IndirectQueryGenerator, NoteGenerator
    from src.utils.llm_client import create_llm_client

    logger = logging.getLogger(__name__)

    loader = load_dataset(dataset_config["dataset"])
    dataset_name = dataset_config["dataset"]["name"]
    logger.info(f"Dataset loaded: {loader.summary()}")

    llm_cfg = base_config["llm"]
    llm_client = create_llm_client(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0.7),
        max_tokens=llm_cfg.get("max_tokens", 2000),
        api_key=llm_cfg.get("api_key") or None,
        base_url=llm_cfg.get("base_url") or None,
    )
    logger.info(f"LLM client: provider={llm_cfg['provider']}, model={llm_cfg['model']}")

    gen_cfg = base_config.get("generation", {})
    output_dir = dataset_config.get("generation", {}).get(
        "output_dir", gen_cfg.get("output_dir", "experiment/generation/outputs")
    )

    # --- Phase A: Generate fake queries ---
    print(f"\n{'='*60}")
    print(f"  Phase A: Generating Fake Queries ({dataset_name})")
    print(f"{'='*60}\n")

    fq_gen = ExperimentFQGenerator(
        llm_client=llm_client,
        num_queries=gen_cfg.get("num_fake_queries", 10),
        language=gen_cfg.get("language", "en"),
        output_dir=output_dir,
        max_concurrent=gen_cfg.get("max_concurrent", 5),
        save_interval=gen_cfg.get("save_interval", 10),
    )

    answer_sids = set()
    for q in loader.queries:
        answer_sids.update(q.evidence_session_ids)

    sessions = loader.get_sessions_as_dicts()
    target_sessions = [s for s in sessions if s["session_id"] in answer_sids]
    logger.info(f"Target sessions (answer sessions): {len(target_sessions)}")

    t0 = time.time()
    await fq_gen.generate_for_dataset(target_sessions, dataset_name)
    logger.info(f"FQ generation completed in {time.time()-t0:.1f}s")

    # --- Phase B: Generate indirect queries ---
    print(f"\n{'='*60}")
    print(f"  Phase B: Generating Indirect Queries ({dataset_name})")
    print(f"{'='*60}\n")

    indirect_gen = IndirectQueryGenerator(
        llm_client=llm_client,
        language=gen_cfg.get("language", "en"),
        output_dir=output_dir,
        max_concurrent=gen_cfg.get("max_concurrent", 5),
        save_interval=gen_cfg.get("save_interval", 20),
    )

    # Build query list with context (session text from evidence)
    queries_with_context = []
    for q in loader.queries:
        context = ""
        for sid in q.evidence_session_ids:
            sess = loader.get_session_by_id(sid)
            if sess:
                context = sess.text[:2000]
                break
        queries_with_context.append({
            "query_id": q.query_id,
            "text": q.text,
            "answer": q.answer,
            "context": context,
        })

    logger.info(f"Target queries for indirect generation: {len(queries_with_context)}")
    t0 = time.time()
    await indirect_gen.generate_for_queries(queries_with_context, dataset_name)
    logger.info(f"Indirect query generation completed in {time.time()-t0:.1f}s")

    # --- Phase C: Generate memory note variants ---
    from experiment.generation.prompts.note_prompt import NOTE_VARIANT_STYLES

    for style in NOTE_VARIANT_STYLES:
        print(f"\n{'='*60}")
        print(f"  Phase C: Generating {style.title()}s ({dataset_name})")
        print(f"{'='*60}\n")

        note_gen = NoteGenerator(
            llm_client=llm_client,
            language=gen_cfg.get("language", "en"),
            output_dir=output_dir,
            max_concurrent=gen_cfg.get("max_concurrent", 5),
            save_interval=gen_cfg.get("save_interval", 10),
        )

        t0 = time.time()
        await note_gen.generate_for_dataset(target_sessions, dataset_name, style=style)
        logger.info(f"{style.title()} generation completed in {time.time()-t0:.1f}s")

    print("\n  Generation complete.\n")


async def step_embed(base_config: dict, dataset_config: dict) -> None:
    """Compute and store embeddings for all generated artifacts."""
    from experiment.src.data_loader import load_dataset
    from experiment.embedding import create_embedding_provider
    from experiment.store import EmbeddingStore

    logger = logging.getLogger(__name__)

    loader = load_dataset(dataset_config["dataset"])
    dataset_name = dataset_config["dataset"]["name"]
    logger.info(f"Dataset loaded: {loader.summary()}")

    emb_cfg = base_config["embedding"]
    provider = create_embedding_provider(
        provider=emb_cfg["provider"],
        model_name=emb_cfg["model_name"],
        device=emb_cfg.get("device", "cpu"),
        dimension=emb_cfg.get("dimension", 0),
        max_seq_length=emb_cfg.get("max_seq_length", 512),
        batch_size=emb_cfg.get("batch_size", 32),
        api_key=emb_cfg.get("api_key", ""),
        base_url=emb_cfg.get("base_url", ""),
    )
    logger.info(f"Embedding provider: {emb_cfg['provider']}, model={emb_cfg['model_name']}")

    store_path = dataset_config.get("store", {}).get(
        "base_dir", f"experiment/store/{dataset_name}"
    )
    store = EmbeddingStore(store_path)
    logger.info(f"Store path: {store_path}")

    gen_output_dir = dataset_config.get("generation", {}).get(
        "output_dir", "experiment/generation/outputs"
    )

    print(f"\n{'='*60}")
    print(f"  Computing Embeddings ({dataset_name})")
    print(f"{'='*60}\n")

    # 1. Embed sessions (content chunks)
    answer_sids = set()
    for q in loader.queries:
        answer_sids.update(q.evidence_session_ids)

    sessions_to_embed = [
        s for s in loader.sessions
        if s.session_id in answer_sids and not store.has_session(s.session_id)
    ]
    logger.info(f"Sessions to embed: {len(sessions_to_embed)}")
    t0 = time.time()
    for i, sess in enumerate(sessions_to_embed):
        chunks, chunk_meta = await provider.embed_session_turns(sess.turns)
        store.save_session_content(sess.session_id, chunks)
        for m in chunk_meta:
            m["session_id"] = sess.session_id
        store.save_chunk_metadata(sess.session_id, chunk_meta)
        if (i + 1) % 20 == 0:
            logger.info(f"  Session embeddings: {i+1}/{len(sessions_to_embed)}")
    logger.info(f"  Session embeddings done ({time.time()-t0:.1f}s)")

    # 2. Embed fake queries
    fq_path = Path(gen_output_dir) / f"{dataset_name}_fake_queries.json"
    if fq_path.exists():
        with open(fq_path, "r", encoding="utf-8") as f:
            fq_data = json.load(f)
        fq_to_embed = [
            (sid, texts) for sid, texts in fq_data.items()
            if not store.has_fake_queries(sid) and texts
        ]
        logger.info(f"Fake queries to embed: {len(fq_to_embed)} sessions")
        t0 = time.time()
        for i, (sid, texts) in enumerate(fq_to_embed):
            embeddings = await provider.embed_batch(texts)
            store.save_fake_queries(sid, embeddings, texts)
            if (i + 1) % 20 == 0:
                logger.info(f"  Fake query embeddings: {i+1}/{len(fq_to_embed)}")
        logger.info(f"  Fake query embeddings done ({time.time()-t0:.1f}s)")
    else:
        logger.warning(f"No fake queries at {fq_path}. Run --step generate first.")

    # 3. Embed true queries
    queries_to_embed = [
        q for q in loader.queries if not store.has_true_query(q.query_id)
    ]
    if queries_to_embed:
        logger.info(f"True queries to embed: {len(queries_to_embed)}")
        t0 = time.time()
        texts = [q.text for q in queries_to_embed]
        embeddings = await provider.embed_batch(texts)
        for q, emb in zip(queries_to_embed, embeddings):
            store.save_true_query(q.query_id, emb)
        logger.info(f"  True query embeddings done ({time.time()-t0:.1f}s)")

    # 4. Embed indirect queries
    iq_path = Path(gen_output_dir) / f"{dataset_name}_indirect_queries.json"
    if iq_path.exists():
        with open(iq_path, "r", encoding="utf-8") as f:
            iq_data = json.load(f)
        iq_to_embed = [
            (qid, data) for qid, data in iq_data.items()
            if not store.has_paraphrases(qid)
        ]
        logger.info(f"Indirect queries to embed: {len(iq_to_embed)}")
        t0 = time.time()
        for i, (qid, data) in enumerate(iq_to_embed):
            indirect = data.get("indirect_queries", {})
            styles = list(indirect.keys())
            texts = [indirect[s] for s in styles if indirect[s]]
            valid_styles = [s for s in styles if indirect[s]]
            if texts:
                embeddings = await provider.embed_batch(texts)
                store.save_paraphrases(qid, embeddings, valid_styles)
            if (i + 1) % 50 == 0:
                logger.info(f"  Indirect query embeddings: {i+1}/{len(iq_to_embed)}")
        logger.info(f"  Indirect query embeddings done ({time.time()-t0:.1f}s)")
    else:
        logger.info(f"No indirect queries at {iq_path} (optional for Exp1).")

    # 5. Embed memory note variants (propositions / notes / reflections)
    variant_map = {
        "proposition": "propositions",
        "note": "notes",
        "reflection": "reflections",
    }
    for style, store_name in variant_map.items():
        vpath = Path(gen_output_dir) / f"{dataset_name}_{style}s.json"
        if not vpath.exists():
            logger.info(f"No {style}s at {vpath}, skipping.")
            continue

        with open(vpath, "r", encoding="utf-8") as f:
            variant_data = json.load(f)

        to_embed = [
            (sid, items)
            for sid, items in variant_data.items()
            if not store.has_variant(store_name, sid) and items
        ]
        logger.info(f"{style.title()}s to embed: {len(to_embed)} sessions")
        t0 = time.time()
        for i, (sid, items) in enumerate(to_embed):
            if style == "note":
                texts = [
                    f"{item['title']}. {item.get('key_insight', '')} {item.get('content', '')}"
                    for item in items
                    if isinstance(item, dict) and item.get("title")
                ]
            else:
                texts = [str(item).strip() for item in items if str(item).strip()]
            if texts:
                embeddings = await provider.embed_batch(texts)
                store.save_variant(store_name, sid, embeddings, texts)
            if (i + 1) % 20 == 0:
                logger.info(f"  {style.title()} embeddings: {i+1}/{len(to_embed)}")
        logger.info(f"  {style.title()} embeddings done ({time.time()-t0:.1f}s)")

    store.close()
    print("\n  Embedding step complete.\n")


async def step_exp1(base_config: dict, dataset_config: dict, args) -> None:
    """Run Experiment 1: Spatial Distribution Consistency."""
    from experiment.src.exp1_alignment import Exp1Alignment

    exp = Exp1Alignment(args.config, args.dataset)
    await exp.run()
    exp.store.close()


async def step_exp2(base_config: dict, dataset_config: dict, args) -> None:
    """Run Experiment 2: Robustness Analysis."""
    from experiment.src.exp2_robustness import Exp2Robustness

    exp = Exp2Robustness(args.config, args.dataset)
    await exp.run()
    exp.store.close()


async def step_exp3(base_config: dict, dataset_config: dict, args) -> None:
    """Run Experiment 3: Temporal Drift Verification."""
    from experiment.src.exp3_temporal_drift import Exp3TemporalDrift

    exp = Exp3TemporalDrift(args.config, args.dataset)
    await exp.run()
    exp.store.close()


async def async_main():
    parser = argparse.ArgumentParser(description="Q2Q Motivation Analysis Experiments")
    parser.add_argument(
        "--config", type=str,
        default="experiment/configs/base.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to dataset config YAML (locomo.yaml or longmemeval.yaml)",
    )
    parser.add_argument(
        "--step", type=str, required=True,
        choices=["generate", "embed", "exp1", "exp2", "exp3", "exp4", "all"],
        help="Which step to run",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()

    # Load .env
    env_path = Path(PROJECT_ROOT) / ".env"
    if env_path.exists():
        load_dotenv(str(env_path), override=True)

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)

    dataset_name = dataset_config.get("dataset", {}).get("name", "unknown")
    log_path = setup_logging(args.log_level, dataset_name, args.step)

    # Merge env vars
    llm = base_config.get("llm", {})
    if not llm.get("api_key"):
        llm["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    if not llm.get("base_url"):
        llm["base_url"] = os.environ.get("OPENAI_BASE_URL", "")
    emb = base_config.get("embedding", {})
    if not emb.get("model_name"):
        emb["model_name"] = os.environ.get("DEFAULT_EMBEDDING_MODEL", "")

    logging.info(f"Starting step '{args.step}' on dataset '{dataset_name}'")
    logging.info(f"Config: {args.config}")
    logging.info(f"Dataset config: {args.dataset}")

    t_start = time.time()

    if args.step == "generate":
        await step_generate(base_config, dataset_config)
    elif args.step == "embed":
        await step_embed(base_config, dataset_config)
    elif args.step == "exp1":
        await step_exp1(base_config, dataset_config, args)
    elif args.step == "exp2":
        await step_exp2(base_config, dataset_config, args)
    elif args.step == "exp3":
        await step_exp3(base_config, dataset_config, args)
    elif args.step == "all":
        await step_generate(base_config, dataset_config)
        await step_embed(base_config, dataset_config)
        await step_exp1(base_config, dataset_config, args)
        await step_exp2(base_config, dataset_config, args)
        await step_exp3(base_config, dataset_config, args)
    else:
        print(f"Step '{args.step}' not yet implemented.")

    elapsed = time.time() - t_start
    logging.info(f"Step '{args.step}' completed in {elapsed:.1f}s")
    logging.info(f"Full log: {log_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
