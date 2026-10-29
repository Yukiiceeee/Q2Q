# Q2Q: Query-to-Query Agent Memory System

An agent memory system built on anchor-query generation and dual-path joint
retrieval.

This repository provides the method implementation. Its experiments are run
under the [MedMemoryBench](https://github.com/AQ-MedAI/MedMemoryBench)
evaluation framework rather than an entry point of its own; see
[Evaluation](#evaluation) and [EVALUATION.md](EVALUATION.md).

## Approach

- **Memory construction.** For each session turn, generate anchor queries (also
  referred to as fake queries in the code) and index them as `query → memory`.
- **Question answering.** Decompose the user query into sub-queries, then match
  them against both the anchor queries (Q2Q) and the raw content (Q2C).

## Installation

Verified on Python 3.10, which is also the version `requirements-lock.txt` was
resolved against. Use a versioned interpreter when creating the environment:
many systems ship only `python3` or `python3.10` and have no bare `python` on
`PATH`, so the command below fails confusingly if copied verbatim.

```bash
python3.10 -m venv .venv && source .venv/bin/activate

# Core runtime
pip install -r requirements.txt

# Additionally, to reproduce the experiments under experiment/
pip install -r requirements-experiment.txt

# Configure
cp .env.example .env
```

To reproduce the reported numbers rather than just run the system, install the
exact versions the results were produced with:

```bash
pip install -r requirements-lock.txt
```

`requirements.txt` declares version floors — the oldest release whose API this
code relies on. `requirements-lock.txt` declares the versions actually used, so
that a major release of `transformers`, `numpy`, or `chromadb` cannot silently
change behaviour. Two extras are commented out in `requirements.txt` because
they are imported lazily and the default configuration does not need them:
`anthropic` (only for `LLM_PROVIDER=anthropic`) and `sentencepiece` (only for
`PROPOSITION_MODE=nlp`, whose `T5Tokenizer` fails to load without it).

Note that `tiktoken` downloads its BPE encoding on first use and caches it, so
the first run needs network access. Set `TIKTOKEN_CACHE_DIR` to control where
that cache lives.

## Usage

`examples/session.txt` is a short synthetic clinical dialogue, included so that
the commands below run against a fresh clone. It carries no license
restriction. Real corpora are not distributed here; see
[DATASETS.md](DATASETS.md).

Global flags precede the subcommand. The sample is English while `LANGUAGE`
defaults to `zh`, so pass `--language en` or set `LANGUAGE=en` in `.env`:
Q2Q generates its own anchor queries and propositions, and leaving the default
in place produces Chinese anchors over English evidence.

```bash
# Store memory
python main.py --language en memorize --file examples/session.txt

# Retrieve
python main.py --language en query "How well is the patient's blood glucose controlled?"

# Interactive session
python main.py --language en interactive

# Statistics
python main.py stats

# Clear memory
python main.py clear
```

## Model weights

No weights are distributed with this repository. The default configuration
needs two checkpoints on disk. Fetch them into `MODELS_DIR` keeping the
directory names intact, since those names are what the configs resolve:

```bash
mkdir -p models && cd models
git lfs install
git clone https://huggingface.co/BAAI/bge-large-en-v1.5        # DEFAULT_EMBEDDING_MODEL
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct    # COMPRESSOR_MODEL
```

The embedding checkpoint backs `EMBEDDING_PROVIDER=local`, and
`Qwen2.5-0.5B-Instruct` backs the perplexity compression that
`MemoryConstructor` runs on every session, so both are needed before the first
`memorize`. A third, `flan-t5-large`, is needed only for
`PROPOSITION_MODE=nlp` (`NLP_INFERENCE_MODEL`, which also requires
`sentencepiece`); the default `llm` mode does not load it.

Setting `EMBEDDING_PROVIDER=openai` with an API model id in
`DEFAULT_EMBEDDING_MODEL` removes the embedding checkpoint from this list but
not the compressor.

Match the embedding checkpoint to the language of the corpus. Q2Q indexes
anchor queries it generates itself, so an English checkpoint over Chinese text
degrades retrieval sharply; `bge-small-zh-v1.5` is the Chinese counterpart
used for the reported Chinese results.

## Asset path convention

No absolute path is hardcoded anywhere in this repository. Two environment
variables define the roots for models and datasets; both default to locations
inside the project and can be pointed at a shared cache instead. Config files
only ever carry names or paths relative to these roots, and all resolution is
centralised in `src/paths.py`.

| Variable | Purpose | Default |
|---|---|---|
| `MODELS_DIR` | Root for local model weights | `<project_root>/models` |
| `DATA_DIR` | Root for datasets | `<project_root>/data` |

For example, `model_name: "bge-large-en-v1.5"` in
`experiment/configs/base.yaml` resolves to `$MODELS_DIR/bge-large-en-v1.5`. A
value that is already absolute is passed through unchanged.

## Datasets

**No dataset is distributed with this repository.** LoCoMo is licensed
CC BY-NC 4.0, and its NonCommercial restriction propagates to anything derived
from it (anchor queries, propositions, embeddings), which is why
`experiment/store/` is gitignored as a local build artifact. See
[DATASETS.md](DATASETS.md) for how to obtain each corpus, the paths the configs
expect, and citation entries.

## Configuration

Configured through `.env`; see `.env.example` for the complete list.

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM backend (openai/anthropic/azure) | openai |
| `DEFAULT_LLM_MODEL` | LLM model | gpt-4o-mini |
| `OPENAI_API_KEY` | API key for LLM and embeddings | - |
| `OPENAI_BASE_URL` | API endpoint, for compatible gateways | - |
| `MODELS_DIR` | Root for local model weights | `<project_root>/models` |
| `DATA_DIR` | Root for datasets | `<project_root>/data` |
| `EMBEDDING_PROVIDER` | Embedding mode (local/openai) | local |
| `DEFAULT_EMBEDDING_MODEL` | Embedding model name, relative to `MODELS_DIR` | - |
| `EMBEDDING_DEVICE` | Embedding device | cpu |
| `PROPOSITION_MODE` | Proposition extraction mode (llm/nlp) | llm |
| `COMPRESSOR_MODEL` | Perplexity compression model | Qwen2.5-0.5B-Instruct |
| `NLP_INFERENCE_MODEL` | Inference model for `nlp` mode | flan-t5-large |
| `STORAGE_CHROMADB_PATH` | ChromaDB persistence directory | data/chromadb |
| `LANGUAGE` | Prompt language for construction and answering (zh/en) | zh |
| `FQ_STYLE` | Anchor-query phrasing (formal/colloquial/keyword/elliptical) | formal |
| `FQ_ROLE` | Anchor-query perspective (user/otherside) | user |
| `Q2Q_ANCHOR_WEIGHT` | Anchor-query weighting coefficient | - |
| `LOG_LEVEL` | Log level | INFO |

A `-` means the code carries no default and the value must be supplied.
`.env.example` ships a working value for each of them.

> `LANGUAGE` has to match the language of the corpus being memorized, because
> Q2Q generates its own anchor queries and propositions rather than indexing
> the source text alone. The default is `zh`; `main.py --language` overrides it
> per invocation, and `experiment/configs/base.yaml` sets `en` for the
> experiments, so the reported results are unaffected by this default.

> Scene routing (dialogue / trajectory / document) is decided automatically at
> memory-construction time by `src/preprocessing/scene_detector.py` from the
> structure of the text, and needs no configuration.
>
> Note that the two retrieval paths are merged directly: there is no α
> weighting between Q2Q and Q2C.

## Project layout

```
Q2Q/
├── agent.py                 # Top-level orchestration (memorize/query)
├── main.py                  # CLI entry point
├── src/
│   ├── config.py            # Global configuration
│   ├── paths.py             # Model/dataset path resolution, project-relative
│   ├── schemas/             # Data models
│   ├── prompts/             # Prompt templates, incl. scene-routed variants
│   ├── embedding/           # Embedding computation (local/API)
│   ├── memory/              # Memory construction (anchor queries + DAG)
│   ├── retrieval/           # Retrieval (decomposition + dual path)
│   ├── storage/             # Storage layer (ChromaDB)
│   └── utils/               # LLM client, logging, tokenizer
├── experiment/              # Motivation analysis (generation/embedding/store/runs)
├── examples/                # Sample input for the commands in Usage
├── scripts/                 # Launch scripts
├── models/                  # Local model weights (not tracked)
├── data/                    # Datasets (not tracked)
├── logs/                    # Log output
└── outputs/                 # Evaluation output
```

## Evaluation

**The experiments in this repository are run under the MedMemoryBench
evaluation framework.** This repository provides the method implementation and
deliberately ships no evaluation entry point: every reported number was
produced by the MedMemoryBench harness driving `agent.py` through a thin
adapter (`methods/q2q_agent.py` there), launched as

```bash
bash scripts/run_eval.sh q2q_gpt-5.1 locomo   # from the MedMemoryBench root
```

The harness owns the datasets, the metrics, the answer judging, and the
per-method hyperparameter configs. Keeping it out of this repository is what
makes Q2Q comparable to the other methods it is reported against: they are all
measured by the same evaluator, with no method-specific evaluation code in the
loop.

[EVALUATION.md](EVALUATION.md) documents how that adapter is written, the five
integration hazards specific to Q2Q, the method config behind the reported
numbers, and what to check in a finished run. Read it before evaluating a new
Q2Q variant.

## Experiment artifacts

`store/`, `results/`, and `figs/` under `experiment/` are all run output and are
not distributed with the repository.

Experiments are driven by `experiment/run.py`, which requires both a dataset
config and a step. The steps are ordered: `generate` writes the anchor and
indirect queries, `embed` turns them into vectors, and the `expN` steps read
those vectors.

```bash
python experiment/run.py --dataset experiment/configs/locomo.yaml --step generate
python experiment/run.py --dataset experiment/configs/locomo.yaml --step embed
python experiment/run.py --dataset experiment/configs/locomo.yaml --step exp1
python experiment/run.py --dataset experiment/configs/locomo.yaml --step all   # all of the above

open experiment/web/index.html    # view results
```

## License

The code is released under the [MIT License](LICENSE). Third-party datasets
retain their original licenses, detailed in [DATASETS.md](DATASETS.md); MIT does
not, and cannot, cover those corpora.
