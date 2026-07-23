# Workflow and Agent-Proxy Claim Register

## Source context

ReAct is an LLM reasoning-and-acting pattern. *AI Agents That Matter* motivates
evaluation beyond accuracy, including cost and reproducibility.

## Local hypothesis

On fixed routing tasks, the constrained proposer/verifier proxy has no higher
unsafe-action rate and no lower accuracy than the heuristic proxy.

## Decision rule

Compare task accuracy and unsafe-action rate. Also report refusal rate, steps,
and zero API cost.
