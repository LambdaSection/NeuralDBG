---
name: rule-107-upstream-pr-strategy
description: 'RULE 107: Upstream PR Strategy — Credibility Through Merged Contributions'
trigger: always
scope: company
priority: 2
status: active
---

<!-- Generated from rules/rule_107_upstream_pr_strategy.md — DO NOT EDIT. scope=company priority=2 status=active -->

# RULE 107: Upstream PR Strategy — Credibility Through Merged Contributions

## Rule

For every confirmed bug in an upstream project (PyTorch, HuggingFace, etc.), the team MUST submit a PR that:
1. Demonstrates detection via the project's diagnostic tool
2. Provides the fix
3. Gets merged upstream

## Rationale

Merged PRs in major OSS projects are the strongest credibility signal:
- They prove the tool works on real codebases (not synthetic benchmarks)
- They give visibility to the tool through the upstream project's audience
- They build trust with maintainers who may become advocates
- Each merged PR is a public, permanent proof of competence

## Strategy

### PR Pipeline per Bug

```
BUG detected -> PR drafted -> PR submitted -> PR reviewed -> PR merged
     |              |              |               |              |
  NeuralDBG    NeuralDBG+     Community       Maintainer     Credibility
  detection    Neural-Agent    feedback        approval       + visibility
```

### What a Valid Upstream PR Contains

1. **Detection**: NeuralDBG output showing the bug (gradient_norm_spike, nan_detected, etc.)
2. **Root cause**: Causal chain identified by the engine
3. **Fix**: Working code change (NOT a workaround, NOT a warning)
4. **Reproduction**: Script that reproduces the bug AND shows NeuralDBG detection
5. **Tests**: Unit test that validates the fix

### PR Quality Gate (before submission)

- [ ] Bug is reproducible (or detection is verified)
- [ ] Fix resolves the root cause (not a symptom)
- [ ] NeuralDBG detection is shown in the PR description
- [ ] Neural-Agent proposed fix is included
- [ ] Tests pass locally
- [ ] PR description follows upstream conventions

## Metrics

Track in ROADMAP.md:
- PRs submitted (total)
- PRs merged (total)
- Merge rate = merged / submitted
- Target: merge rate > 50%

## Enforcement

IF bug detected upstream AND fix available:
  ACTION: Submit PR within 48 hours
  IF PR rejected: document reason, iterate, resubmit
  IF PR ignored after 7 days: ping maintainer politely

---

**Created**: 2026-06-09
**Trigger**: 4 bugs documented, 0 PRs submitted. Need systematic PR strategy.
**Enforcement**: MANDATORY
