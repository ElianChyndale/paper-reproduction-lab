# Workflow and Agent-Proxy Study

## Source context

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — The paper evaluates an LLM pattern that interleaves reasoning traces and environment actions.
- [AI Agents That Matter](https://arxiv.org/abs/2407.01502) — The paper argues that agent evaluation should account for cost, holdouts, standardization, and reproducibility alongside accuracy.

## Local hypothesis

An allowlisted proposer/verifier proxy should select fewer unsafe actions than an unconstrained heuristic proxy on fixed prompt-injection tasks.

## Reproduced mechanics

- fixed task holdout
- accuracy, unsafe-action, refusal, step, and cost recording
- proposal validation against an allowlist
- prompt-injection failure fixtures

## Not reproduced

- LLM inference or training
- ReAct prompts, traces, environments, or reported results
- AgentBench or other agent benchmark environments
- claims about real single-agent or multi-agent systems

## Synthetic test results

| Method | task-accuracy | unsafe-action-rate | refusal-rate | mean-steps | api-cost-usd |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deterministic-workflow` | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| `heuristic-single-agent` | 0.666667 | 0.333333 | 0.000000 | 2.000000 | 0.000000 |
| `constrained-multi-agent` | 1.000000 | 0.000000 | 0.333333 | 2.333333 | 0.000000 |

## Scoped finding

The heuristic proxy unsafe-action rate was `0.333333`; the constrained proxy rate was `0.000000`. No tool was executed and API cost was zero by construction.

## Rerun

```text
paper-repro run --study workflow-agent --output research/results/v0.1
```
