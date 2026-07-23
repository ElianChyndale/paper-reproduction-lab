# Paper Sources and Scope

## Retrieval

### Dense Passage Retrieval for Open-Domain Question Answering

- Authors: Vladimir Karpukhin et al.
- Year: 2020
- Source: https://arxiv.org/abs/2004.04906
- Source claim used for context: learned dual-encoder dense retrieval can be
  effective for open-domain question answering and was compared with a strong
  BM25 system on the paper's datasets.
- v0.1 status: **not reproduced**. The lab does not train DPR, download its
  datasets, or compare against its reported values.
- Local use: TF-IDF + SVD is a deterministic semantic-retrieval control.

### Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods

- Authors: Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher.
- Year: 2009.
- Source: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- Source claim used for context: reciprocal-rank fusion combined multiple
  rankings effectively in the paper's retrieval experiments.
- v0.1 status: **method-behavior reproduction**. The rank-fusion formula is
  implemented, but the TREC/LETOR experiments and paper-level comparisons are
  not reproduced.

## Calibration and abstention

### On Calibration of Modern Neural Networks

- Authors: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger.
- Year: 2017.
- Source: https://proceedings.mlr.press/v70/guo17a.html
- Source claim used for context: temperature scaling was a simple and effective
  post-hoc calibration method in many experiments reported by the paper.
- v0.1 status: **method-behavior reproduction** and **scoped claim check**.
  Temperature scaling is fitted on synthetic logits; neural architectures,
  original datasets, and paper table values are not reproduced.

### Selective Classification for Deep Neural Networks

- Authors: Yonatan Geifman, Ran El-Yaniv.
- Year: 2017.
- Source: https://arxiv.org/abs/1705.08500
- Source claim used for context: selective classification trades coverage for
  risk by rejecting uncertain predictions.
- v0.1 status: **method-behavior reproduction**. Confidence thresholding and a
  risk-coverage curve are computed, but the paper's guarantees, datasets, and
  neural networks are not reproduced.

## Workflow and agent evaluation

### ReAct: Synergizing Reasoning and Acting in Language Models

- Authors: Shunyu Yao et al.
- Year: 2022/2023.
- Source: https://arxiv.org/abs/2210.03629
- Source claim used for context: interleaving reasoning traces and actions is
  an agent design pattern evaluated by the paper.
- v0.1 status: **not reproduced**. No LLM, prompt, environment, or ReAct result
  is reproduced.

### AI Agents That Matter

- Authors: Sayash Kapoor et al.
- Year: 2024.
- Source: https://arxiv.org/abs/2407.01502
- Source claim used for context: agent evaluations should consider cost,
  holdouts, standardization, and reproducibility in addition to accuracy.
- v0.1 status: **methodological reference only**. The local fixture records
  accuracy, unsafe actions, steps, and zero API cost for deterministic proxies.
