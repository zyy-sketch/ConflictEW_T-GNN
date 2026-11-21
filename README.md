# Conflict Early Warning: Kalman-Filter Signal Extraction with Multi-Factor XGBoost

Reproducible research pipeline for the manuscript *"Multi-Indicator Conflict Early Warning: A Kalman Filter Approach"* — data ingestion → Kalman local-linear-trend signal extraction → multi-factor features → XGBoost with blocked temporal cross-validation → baseline comparisons → uncertainty quantification.

> **Honest-results statement.** This is a negative-result release on real data. With real UCDP conflict-onset labels (v25.1) and real WDI military-expenditure covariates, the multi-factor Kalman model reaches **AUC 0.545 (DeLong 95% CI [0.493, 0.597])** on strictly out-of-sample years (2015–2024, 61 onsets) — **not better than a rare-events logistic baseline (AUC 0.572, paired diff n.s. p=0.61)**, and the headline Kalman military-anomaly signal is *harmful*: removing it raises AUC to 0.680. The model only significantly outperforms the pooled VAR(2) baseline (AUC 0.448, p=0.018). We report these findings as-is; see the manuscript and `PROVENANCE.md`.

## What is inside

| Path | Content |
|---|---|
| `src/conflict_ew/` | Pipeline library (data, Kalman, factors, XGBoost, baselines, metrics, agent) |
| `scripts/` | Stage runners `00`–`06`, sensitivity/calibration `09`–`11`, release tooling `12`, provenance `fetch_external.py`, `verify_wdi_sipri.py` |
| `tests/` | 33 tests incl. look-ahead-leakage suite (L1–L3, L5; L4 = disclosed exception) and golden-value regressions |
| `config.yaml` | Every hyperparameter, seed, and path (no magic numbers in code) |
| `PROVENANCE.md` | Data source chain, SHA-256 registry, and a required disclosure on simulated external covariates |
| `data/external/raw/` | Official raw downloads (Polity5, COW Intra-State Wars v5.1, COW Direct Contiguity v3.2) |
| `data/external_real/` | Converted official originals — consumed by the pipeline when `data.external_source: real` (default since A2) |
| `reports/` | Self-contained pipeline reports (`run_all.py` output) |
| `dashboard_v3.html`, `dashboard_rt.html` | Static/real-time demo dashboards — **hard-coded demo data, not pipeline output** |

## Headline reproduced numbers (golden regressions in `tests/`, real-data stack)

| Metric | Value |
|---|---|
| AUC (multi-factor, OOS 2015–2024) | 0.5447 [0.4928, 0.5967] |
| Brier (before → after isotonic calibration) | 0.0391 → 0.0370 |
| Rare-events logistic baseline | 0.5716 [0.4897, 0.6534] (paired diff −0.027, p=0.61 n.s.) |
| Pooled VAR(2) baseline | 0.4479 [0.3739, 0.5219] (paired diff +0.098, p=0.018) |
| Ablation: drop military-anomaly F1 | **0.6801** (F1 is harmful on real labels) |
| Ablation: military signal only (F1) | 0.4859 (below chance) |
| Kalman estimation-window sensitivity (M2) | r = 0.926, flag agreement 98.9% |
| Test window | 1,650 rows, 61 real onsets |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.lock                      # exact environment (Python 3.13)
python scripts/run_all.py                             # load → features → train → eval → figures
python -m pytest -q                                   # 33 tests incl. golden regressions
python scripts/06_repro_check.py                      # two-run determinism check
```

All pipeline stages (F01–F18) run inside `run_all.py` (see `src/conflict_ew/pipeline.py`); auxiliary scripts: `06_repro_check.py` (determinism), `09_window_sensitivity.py` / `10_calibration_reliability.py` / `11_secondary_metrics.py` (sensitivity & secondary metrics), `12_export_release.py` (release artifacts + MANIFEST).

> **Note on fresh clones**: the WDI snapshot `data/raw_real.json` that pins the golden numbers **is included**, so `run_all.py` reproduces the released metrics as-is (live-WDI API drift would otherwise shift results — observed +0.07 AUC on 2026-11-21). Only `artifacts/` is git-ignored and absent after cloning, so golden-regression tests that read pipeline outputs mostly *skip* until you run `python scripts/run_all.py` once to generate them.

## Data

- **Military expenditure & macro covariates (real, auto-fetched)**: World Bank WDI `MS.MIL.XPND.CD` (SIPRI-reported, current US$), GDP growth, inflation, unemployment. A 100 country-year spot-check against the official SIPRI 2025 xlsx found median difference 0 (86% within 1%); see `scripts/verify_wdi_sipri.py`.
- **External covariates & conflict labels**: since the A2 decision (2026-11-15) the pipeline consumes **real data** by default (`data.external_source: real`): conflict labels from **UCDP Intrastate Country-Year Onset v25.1** (1946–2024), Polity5 polity2 scores, and COW Direct Contiguity v3.2 adjacency — all auto-downloaded to `data/external/raw/` with SHA-256 registered in `data/external/provenance.json` and converted by `python scripts/fetch_external.py --convert`. ICOW/ATOP/V-Dem require manual download (official sites block scripting) — F5/F6 and the V-Dem leg of F4 degrade to zero and are marked in run logs. The legacy simulated stack remains switchable (`data.external_source: simulated`) purely as a regression fixture. Full chain: **`PROVENANCE.md`**.
- All labels/features derive from `(iso3, year)` identifiers; the country identity mapping is provided in the repository (`data/external/countries.csv`).

## Reproducibility design

- Fixed seed 20240101 across Python/NumPy/XGBoost; deterministic two-run check (`06_repro_check.py`).
- Strictly out-of-sample evaluation: blocked expanding-window CV with `test_start: 2015`; features use ≤ t−1 information only (look-ahead suite L1–L3/L5 in `tests/test_lookahead.py`; the Kalman full-sequence MLE is a disclosed exception, quantified by the sensitivity analysis).
- Rare-event handling: `scale_pos_weight` (XGBoost) and King–Zeng prior correction (logistic baseline).
- Uncertainty: DeLong CIs, paired bootstrap with BH-FDR correction, Wilson intervals, parametric bootstrap for Brier.
- Release artifacts (model JSON, OOS predictions, SHAP values, MANIFEST with size+SHA-256) are generated by `scripts/12_export_release.py`; `artifacts/` is git-ignored and archived to Zenodo at submission time.

## Demo dashboards (not part of the scientific claims)

`dashboard_v3.html` (static) and `dashboard_rt.html` (SSE-driven) are self-contained front-end demos. They load hard-coded demo data and a rules-based "AI reasoning" template — **no LLM, no pipeline output**. The real-time architecture sketch lives in the project's internal working notes (not part of this public repository).

## Environment

Python 3.13; key versions pinned in `requirements.lock` (xgboost 3.4.1, scikit-learn 1.9.0, numpy 2.5.2, pandas 3.0.5) and matching the manuscript's reproducibility statement.

## Citation

See `CITATION.cff`. A Zenodo DOI will be minted for the submission release tag.

## License

MIT (see `LICENSE`). Data files retain their original providers' terms: World Bank (CC BY 4.0), SIPRI (CC BY 4.0), Polity5 (free for research use, cite Marshall & Gurr 2020), COW project data (public domain per project citation norms).
