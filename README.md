# Q2Q: Query-to-Query Agent Memory System

An agent memory system built on anchor-query generation and dual-path joint retrieval.

## Approach

- **Memory construction.** For each session turn, generate anchor queries (also
  referred to as fake queries in the code) and index them as `query → memory`.
- **Question answering.** Decompose the user query into sub-queries, then match
  them against both the anchor queries (Q2Q) and the raw content (Q2C).

## Installation

Verified on Python 3.10.

```bash
python -m venv .venv && source .venv/bin/activate

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

```bash
# Store memory
python main.py memorize --file data/session.txt

# Retrieve
python main.py query "How well is the patient's blood glucose controlled?"

# Interactive session
python main.py interactive

# Statistics
python main.py stats

# Clear memory
python main.py clear
```

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
| `FQ_STYLE` | Anchor-query phrasing (formal/colloquial/keyword/elliptical) | formal |
| `FQ_ROLE` | Anchor-query perspective (user/otherside) | user |
| `Q2Q_ANCHOR_WEIGHT` | Anchor-query weighting coefficient | - |
| `LOG_LEVEL` | Log level | INFO |

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
├── scripts/                 # Launch scripts
├── models/                  # Local model weights (not tracked)
├── data/                    # Datasets (not tracked)
├── logs/                    # Log output
└── outputs/                 # Evaluation output
```

> Evaluation is driven by an external harness (MedMemoryBench
> `scripts/run_eval.sh`); this repository no longer ships an evaluation entry
> point.

## Experiment artifacts

`store/`, `results/`, and `figs/` under `experiment/` are all run output and are
not distributed with the repository:

```bash
python experiment/run.py          # regenerate
open experiment/web/index.html    # view results
```

## License

The code is released under the [MIT License](LICENSE). Third-party datasets
retain their original licenses, detailed in [DATASETS.md](DATASETS.md); MIT does
not, and cannot, cover those corpora.
