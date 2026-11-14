# Reasoning-Chain Quality-Assurance Protocol (Cohen's κ) — Planned, Not Executed

> Status: **prospective specification**. No annotation was performed for the paper and
> no κ value is reported. All κ-related statements in the manuscript (§III-C) are
> design commitments, not measured results. This document, the blank annotation
> sheet (`data/annotation_sheet.csv`) and the aggregation script
> (`scripts/summarize_kappa.py`) are released so that independent raters can run
> the study on the released reasoning chains.

## Protocol (mirrors manuscript §III-C verbatim in substance)

- **Holdout sample**: 50 country-year reasoning chains from the released
  2015–2024 test-window chains (backend: `rule-template`), stratified
  **25 conflict-positive / 25 conflict-negative**.
- **Annotators**: two independent human annotators, blind to model internals;
  they see the reasoning chain plus the source panel row only.
- **Dimensions** (each scored by both annotators):
  1. *factual accuracy* — are the claims supported by the input data;
  2. *logical coherence* — does the inference from factors to conclusion hold;
  3. *actionability* — does the output give a decision-maker something usable.
- **Agreement statistic**: Cohen's κ per dimension (binary: pass/fail per chain
  per dimension, per annotator), plus pooled κ over all 150 judgments.
- **Deployment gate**: **κ ≥ 0.7** on every dimension is the prospective gate for
  authorizing analyst-facing deployment of the reasoning layer.

## How to run the study

1. Draw the 50-chain holdout with `scripts`-level stratification on the gold
   label; record chain IDs in `data/annotation_sheet.csv` (one row per
   chain × annotator).
2. Each annotator fills `pass`/`fail` in the three dimension columns.
3. Aggregate: `python scripts/summarize_kappa.py --sheet data/annotation_sheet.csv`
   → prints per-dimension κ, pooled κ, and pass/fail against the 0.7 gate.

The script contains **no scoring logic** — it only aggregates human labels.
