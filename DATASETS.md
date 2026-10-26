# Third-party datasets and licensing

The MIT license in `LICENSE` covers **the code in this repository only**. It
does not and cannot cover the evaluation corpora, which carry their own terms.

## Datasets are not redistributed here

No dataset file ships with this repository. Earlier revisions tracked a derived
index under `experiment/store/` that embedded verbatim LoCoMo transcripts; that
content has been removed from the working tree and from the git history, and
`experiment/store/` is now gitignored as a local build artifact.

Fetch each corpus yourself and place it under the dataset root, which defaults
to `<project_root>/data` and is redirected by the `DATA_DIR` environment
variable. The paths below are what `experiment/configs/*.yaml` expect.

| Dataset | Expected path under the dataset root | License | Source |
|---|---|---|---|
| LoCoMo | `locomo/data/locomo10.json` | CC BY-NC 4.0 | https://github.com/snap-research/locomo |
| LongMemEval | `LongMemEval/data/longmemeval-cleaned/longmemeval_s_cleaned.json` | see upstream | https://github.com/xiaowu0162/LongMemEval |
| AMA-Hub | `AMA-Hub/dataset/test/open_end_qa_set.jsonl` | see upstream | upstream release |

## What CC BY-NC 4.0 means for you

LoCoMo is released under Creative Commons Attribution-NonCommercial 4.0
International. Two consequences follow, and they survive processing:

1. **Attribution.** Any published use of LoCoMo, or of material derived from
   it, must credit the original authors.
2. **NonCommercial.** The NC restriction is not removed by transformation. Text
   that this project generates *from* LoCoMo sessions -- anchor queries,
   propositions, notes, reflections, embeddings -- is a derivative work and
   remains non-commercial. A local `experiment/store/` built by running the
   pipeline therefore may not be redistributed commercially, which is a second
   reason it is gitignored rather than committed.

The MIT grant on the code is unaffected: you may use the Q2Q implementation
commercially. You simply may not do so *on LoCoMo data*.

## Citing the datasets

```bibtex
@inproceedings{maharana2024locomo,
  title     = {Evaluating Very Long-Term Conversational Memory of LLM Agents},
  author    = {Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and
               Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for
               Computational Linguistics},
  year      = {2024}
}

@inproceedings{wu2025longmemeval,
  title     = {LongMemEval: Benchmarking Chat Assistants on Long-Term
               Interactive Memory},
  author    = {Wu, Di and Wang, Hongwei and Yu, Wenhao and Zhang, Yuwei and
               Chang, Kai-Wei and Yu, Dong},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}
```
