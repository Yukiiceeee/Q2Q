# Evaluating Q2Q on MedMemoryBench

This repository ships the Q2Q method: memory construction, anchor-query
generation, the temporal graph, and dual-path retrieval. It does not ship an
evaluation entry point. Every number reported for Q2Q was produced by the
MedMemoryBench harness driving this code through a thin adapter, and the
motivation experiments under `experiment/` replay the runs that harness writes.

This document describes how that adapter is written, so that the reported
results can be reproduced and so that a new variant of Q2Q can be evaluated
without reverse-engineering the integration.

The reference implementation is `methods/q2q_agent.py` in
[MedMemoryBench](https://github.com/AQ-MedAI/MedMemoryBench). Read it alongside
this document: what follows explains the shape and the non-obvious constraints
rather than restating the file.

## Division of labour

```
Q2Q (this repository)            MedMemoryBench (harness)
---------------------            ------------------------
agent.py       Q2QAgent          methods/q2q_agent.py   adapter
src/memory/    construction      src/agent.py           method registry
src/retrieval/ dual path         configs/method_config/ hyperparameters
src/storage/   ChromaDB          configs/dataset_config/ corpus + language
                                 benchmarks/*/evaluator.py  metrics
                                 scripts/run_eval.sh    launcher
```

The adapter is the only place the two meet. Nothing in this repository imports
the harness, and the harness touches only the public surface of `Q2QAgent`.

## The four touchpoints

Adding or changing a method in the harness means editing exactly four places.

1. `methods/<name>_agent.py` — the adapter class, subclassing `BaseAgent`.
2. `src/agent.py` — one entry in `SUPPORTED_METHODS` mapping a method key to
   `(module, class)`, plus a branch in the parameter builder that forwards
   config knobs to the constructor.
3. `configs/method_config/<name>_<model>.yaml` — the hyperparameters.
4. `scripts/run_eval.sh <method_config> <dataset>` — the launcher.

Registry resolution is first-substring-wins, so a longer key must be listed
before a shorter one that is a prefix of it. `q2q_v1` and `q2q_abl` precede
`q2q` for this reason; a key added as `q2q_something` without that ordering
silently resolves to the full method instead.

## The contract

`BaseAgent` (in `methods/base.py`) requires two methods.

```python
def memorize(self, text: str, **kwargs) -> MemoryBuildResult: ...
def query(self, question: str, system_message: str | None = None, **kwargs) -> AgentResponse: ...
```

`MemoryBuildResult` reports what was stored: `success`, `memory_entries`,
`chunk_count`, `time_cost`, and a free-form `extra`. `AgentResponse` reports
the answer: `output`, `retrieved_count`, `retrieved_memories`, and `extra`.
The evaluator scores `output` and uses everything else for diagnostics, so the
counters have to describe what actually reached the prompt rather than what
retrieval returned as a pool.

`BaseAgent` also provides `reset()`, called between personas, and
`truncate_docs_to_context()` for methods that need simple context packing.

## Minimal adapter

The skeleton below is the complete shape, with the Q2Q-specific hazards marked.
Each is explained in the next section.

```python
import os, sys, asyncio, tempfile, shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseAgent, MemoryBuildResult, AgentResponse

_q2q_kept_refs: Dict[str, Any] = {}
_Q2QAgent = None
_Q2QConfig = None


def _load_q2q_modules(q2q_root: str) -> None:
    """Import Q2Q under a temporarily swapped-out ``src`` namespace.

    Hazard 1. Both repositories have a top-level ``src`` package. Importing
    Q2Q with the harness's ``src`` resident binds Q2Q's modules to the wrong
    package and fails in ways that look like method bugs.
    """
    global _Q2QAgent, _Q2QConfig
    if _Q2QAgent is not None:
        return

    saved_src = {}
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            saved_src[key] = sys.modules.pop(key)

    sys.path.insert(0, q2q_root)
    try:
        import agent as q2q_agent_mod
        _Q2QAgent = q2q_agent_mod.Q2QAgent
        _Q2QConfig = q2q_agent_mod.Q2QConfig
    finally:
        # Keep hard references so the modules stay alive after being unloaded,
        # then restore the harness's own ``src``.
        for key in list(sys.modules):
            if key == "src" or key.startswith("src.") or key == "agent":
                _q2q_kept_refs[key] = sys.modules.pop(key)
        sys.modules.update(saved_src)
        if q2q_root in sys.path:
            sys.path.remove(q2q_root)


class MyQ2QAgent(BaseAgent):
    METHOD_TYPE = "agentic_memory"

    def __init__(self, q2q_project_path: str = "", language: str = "zh",
                 num_fake_queries: int = 15, top_n: int = 5,
                 embedding_device: str = "cpu", **kwargs):
        super().__init__(**kwargs)

        root = Path(q2q_project_path)
        if not root.is_absolute():
            root = (Path(__file__).resolve().parent.parent / root).resolve()
        _load_q2q_modules(str(root))

        self._language = language          # Hazard 3: follows the dataset
        self._num_fake_queries = num_fake_queries
        self._top_n = top_n
        self._embedding_device = embedding_device
        self._temp_dir = None
        self._create_q2q_agent()

    # -- event loop ------------------------------------------------------
    # Hazard 2. Q2Q is async, the harness is synchronous. One persistent loop
    # per adapter instance; asyncio.run() would close the loop and invalidate
    # the ChromaDB client and the embedding model held across calls.
    def _get_event_loop(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def _run_async(self, coro):
        return self._get_event_loop().run_until_complete(coro)

    # -- construction ----------------------------------------------------
    def _build_q2q_config(self, storage_path: str):
        config = _Q2QConfig()
        config.llm.model = self.model
        config.llm.api_key = os.environ.get("OPENAI_API_KEY", "")
        config.llm.base_url = os.environ.get("OPENAI_BASE_URL", "")
        config.language = self._language
        config.retrieval.num_fake_queries = self._num_fake_queries
        config.retrieval.top_n = self._top_n
        config.embedding.device = self._embedding_device
        config.storage.chromadb_path = os.path.join(storage_path, "chromadb")
        # Hazard 4: one collection per process, or parallel workers collide.
        config.storage.chromadb_collection = f"q2q_{os.getpid()}"
        config.log.level = "WARNING"
        return config

    def _create_q2q_agent(self):
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = tempfile.mkdtemp(prefix="q2q_bench_")
        self._q2q_agent = _Q2QAgent(self._build_q2q_config(self._temp_dir))

    def reset(self) -> None:
        super().reset()
        self._create_q2q_agent()

    def __del__(self):
        if getattr(self, "_temp_dir", None) and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    # -- contract --------------------------------------------------------
    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        import time
        t0 = time.time()
        # Public API only. Reaching into memory_constructor or memory_store
        # here is what makes an adapter rot the next time Q2Q is refactored.
        # memorize() returns one MemoryEntry per call, not a list.
        entry = self._run_async(self._q2q_agent.memorize(text))
        memory_entries = [
            {"event": fq.text,
             "session_id": entry.session_id,
             "fake_query_id": fq.query_id,
             "answer_source": fq.answer_source}
            for fq in (entry.fake_queries or [])
        ]
        return MemoryBuildResult(
            success=True,
            method="q2q",
            action="memorize",
            input_content=text,
            memory_entries=memory_entries,
            chunk_count=len(entry.fake_queries or []),
            time_cost=time.time() - t0,
        )

    def query(self, question: str, system_message: Optional[str] = None,
              **kwargs) -> AgentResponse:
        import time
        t0 = time.time()
        # Hazard 5: strip the evaluator's answer-format instructions before
        # using the text as a retrieval signal.
        raw = self._extract_raw_question(question)
        agent = self._q2q_agent
        sub_queries = self._run_async(agent.query_decomposer.decompose(raw))
        result = self._run_async(agent.dual_retriever.retrieve(sub_queries))

        context = self._pack_context(result, question)   # elided, see below
        response = self._run_async(agent.llm_client.chat([
            {"role": "system", "content": system_message or ""},
            {"role": "user", "content": f"{context}\n\n{question}"},
        ]))
        return AgentResponse(
            output=response.content,
            query_time=time.time() - t0,
            retrieved_count=len(result.matched_fqs),
            retrieved_memories=[{"content": s} for s in result.direct_session_texts],
            extra={"method": "q2q",
                   "sub_queries": [sq.text for sq in sub_queries]},
        )
```

## The five hazards

**1. The `src` namespace collision.** Both repositories expose a top-level
`src` package. The adapter must pop every `src*` entry out of `sys.modules`,
put the Q2Q root on `sys.path`, import, then unload Q2Q's modules while
keeping hard references to them and restore the harness's `src`. Skipping the
hard references lets the garbage collector reclaim modules whose classes are
still instantiated, which surfaces much later as attribute errors on objects
that worked a moment earlier.

**2. One event loop per adapter, not per call.** `Q2QAgent` is async and the
harness is synchronous. `asyncio.run()` creates and closes a loop on every
call, which invalidates the ChromaDB client, the loaded embedding model, and
any open connection the agent holds between `memorize` and `query`. Create one
loop and reuse it.

**3. Language follows the data, not the method.** Q2Q generates its own
prompts, so the language of those prompts has to match the language of the
text being memorized. The harness therefore defaults this parameter to
`dataset_config.language` rather than pinning it in the method config, and the
method config deliberately leaves it unset. The local embedding checkpoint
follows the same rule: an English checkpoint scored R@1 0.034 and MRR 0.056 on
Chinese text where the Chinese checkpoint scored 0.134 and 0.172, so the
adapter selects the checkpoint from the language instead of hardcoding one.
Pin `language` in the method config only to run a method cross-lingually on
purpose.

**4. Per-process storage isolation.** Storage goes in a fresh
`tempfile.mkdtemp()` per agent, and the ChromaDB collection name carries the
process id. Without both, parallel workers write into one collection and every
persona sees every other persona's memories, which inflates accuracy in a way
that is invisible in the logs.

Caches for expensive artifacts, on the other hand, belong at module level so
they survive `reset()`. The embedding provider and the perplexity compressor
cost seconds to load and are stateless with respect to the stored memories;
reloading them per persona dominates the wall clock of a full run. The
reference adapter keeps module-level caches keyed by model path and device,
and injects the cached objects into the freshly constructed agent.

**5. The question is not the prompt.** The evaluator hands the adapter a
rendered QA prompt: a question label, the question, then answer-format
instructions addressed to the answer model. Feeding all of it to the query
decomposer pollutes retrieval with instruction text. Extract the question for
retrieval, and pass the full rendered prompt to the answer call.

A sixth point is not a hazard but a reporting requirement. When the evaluator
packs several sessions into one `memorize` call, the adapter should derive the
real date span from the payload and pass it as metadata, otherwise the
temporal graph timestamps every event with the first date it sees. Translate
Q2Q's retry-exhaustion exception into the harness's equivalent as well, so an
exhausted API budget is recorded as an infrastructure failure rather than
scored as a wrong answer.

## Packing the context

`_pack_context` is elided above because it is where an adapter most easily
biases the measurement. Dual-path retrieval returns more than fits in the
window, so something has to choose what reaches the prompt, and a fixed
percentage split between evidence types wastes budget on a small retrieval and
truncates arbitrarily on a large one.

The reference adapter claims budget in order of information density per token
rather than by a fixed split. Propositions go first: they carry the distilled
anchors that precise-recall questions hinge on, at a fraction of the tokens
per fact. Direct Q2Q session texts follow as primary evidence at full
fidelity, then DAG chain texts and Q2C texts expand into whatever remains.
Letting session texts fill first starved the proposition tier to zero bytes on
every long-session query.

Budget priority and presentation order are independent. Full-fidelity evidence
leads the prompt and distilled facts follow it as anchors, regardless of which
claimed budget first.

Report what was admitted, not what was retrieved. Keep the pool counts and the
injected counts as separate fields in `extra`: a top-k sweep varies the pool,
but the injected counts are what the answer model actually saw, and conflating
them makes a budget-bound run look like a retrieval-bound one.

## Method config

Hyperparameters live in `configs/method_config/`. The file below is the
configuration behind the reported Q2Q numbers.

```yaml
method_name: "q2q"
method_type: "agentic_memory"

model:
  name: "gpt-5.1"
  provider: "openai"
  temperature: 0.3
  max_completion_tokens: 40960

agent_params:
  q2q_project_path: "../Agent_Memory/Q2Q"

  # Retrieval
  top_k_per_sub: 20
  top_n: 5
  top_k_q2c: 3
  num_fake_queries: 15
  chain_depth: 3
  prop_top_k_per_fq: 3
  binding_verify_threshold: 0.8
  max_context_tokens: 64000

  # Preprocessing
  proposition_mode: llm
  compressor_model: models/Qwen2.5-0.5B-Instruct
  compressor_device: cpu
  perplexity_threshold: 2.5
  nlp_inference_model: models/flan-t5-base

  # Graph
  evolve_threshold: 0.6
  diverge_low: 0.2
  diverge_high: 0.6
  max_parents: 3
  max_chain_length: 5
  max_chain_nodes: 60

  storage_backend: chromadb
  embedding_device: cpu
  # language is intentionally absent: see hazard 3.

embedding:
  provider: local
  # model is intentionally absent: selected from the dataset language.
```

The parameter builder in `src/agent.py` supplies a default for every knob and
then forwards any remaining `agent_params` entry verbatim. The explicit list
exists so a config may omit a knob; the verbatim forwarding means a parameter
added to Q2Q later reaches the adapter without the allow-list having to be
updated first.

## Running

```bash
# From the MedMemoryBench root
bash scripts/run_eval.sh q2q_gpt-5.1 locomo
```

The first argument is the method config filename without its extension, the
second a dataset config name. Further arguments pass through to `main.py`.

Before the first run, confirm that `q2q_project_path` points at this
repository, that `OPENAI_API_KEY` and `OPENAI_BASE_URL` are exported, and that
the checkpoints named in the method config exist under `MODELS_DIR`.

## Checking a run

A run that reports a plausible accuracy having indexed nothing is the failure
mode worth guarding against, so check the build before the score.

Confirm that `memory_build.json` has a non-zero `chunk_count` and non-empty
`memory_entries`. In each query record, confirm that `retrieved_count` is
non-zero and that the injected counters in `extra` are non-zero; the
`num_*_sessions` fields report the retrieval pool, while the `injected_*`
fields report what the token budget actually admitted into the prompt. A run
where the injected counts are zero is measuring the answer model alone.
`context_used_tokens` against `context_budget_tokens` shows whether the budget
was the binding constraint.
