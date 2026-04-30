# Reproducibility Guide

This guide provides step-by-step instructions to reproduce every result in the paper — Tables III, IV, V, and VI — from a clean environment.

**Prerequisite:** Complete all steps in [README.md](README.md) (verify Java, set up Python, compile) before running any command below. All commands assume `L4Project/` as the working directory.

---

## Overview

| Paper result | Ablation | Key varied factor | Fixed parameters |
|---|---|---|---|
| Table III (MIP vs. baselines, Pc=0.25) | `abl2_pc` at level 0.25 | Surprise measure (MIS/CC/BF/none) | lambda=1.0, lookback=4, mecThreshold=20, rplThreshold=0.2 |
| Table IV (dual link failure) | `abl5_disaster`, scenario `fail100_12-7_12-3` | Surprise measure | lambda=1.0, p_c=0.5, lookback=4, mecThreshold=20, rplThreshold=0.2 |
| Table V (lambda ablation) | `abl1_lambda` | lambda ∈ {0.0,0.1,0.5,1.0,2.0,5.0,20.0} | p_c=0.5, lookback=4, mecThreshold=20, rplThreshold=0.2 |
| Table VI (NFR tightening, level 17/0.17) | `abl4_nfr` at level (17,0.17) | NFR threshold pair | lambda=1.0, p_c=0.5, lookback=4 |

All experiments use seeds 222, 223, 224 (3 runs each) and 500 timesteps per run. Results in the paper are the average of the 3 seeds.

---

## Running All Experiments at Once

The simplest way to reproduce everything is to run all ablations together:

```bash
python scripts/run_ablation.py run
```

This runs every ablation family (lambda, p_c, lookback, NFR, disaster), all four surprise variants (MIS, CC, BF, no_surprise), all three seeds. It skips any configuration that already has valid output, so it is safe to resume after interruption.

Optional flags:
```bash
# Force re-run of all configurations (overwrite existing output)
python scripts/run_ablation.py run --overwrite

# Rebuild summary CSVs only from existing run directories (no Java runs)
python scripts/run_ablation.py summary

# Regenerate figures only from existing summary CSVs
python scripts/run_ablation.py plots

# Quick sanity check: 1 seed, first 2 levels per ablation (~20 min)
python scripts/run_ablation.py run --quick --no-plots
```

After all runs complete, summary CSVs are written to `output_dir/results/summary_<abl_id>_avg.csv`. Each CSV has one row per `(level, surprise)` combination with columns `mean_MEC`, `median_MEC`, `Q1_MEC`, `Q3_MEC`, `lower_fence_MEC`, `upper_fence_MEC`, and RPL equivalents.

---

## Table III — MIP vs. Deviation Signals (Pc = 0.25)

**What the table shows:** Mean and distributional MEC/RPL statistics for ERPerseus paired with each surprise measure (MIS, CC, BF, none) at Pc=0.25, mecThreshold=20, rplThreshold=0.2.

**Ablation:** `abl2_pc`, level `0.25`

**Exact configuration:**
```
algorithmType           = erperseus
lambda                  = 1.0
p_c                     = 0.25
lookback                = 4
mecThreshold            = 20
rplThreshold            = 0.2
runSeed                 = 222, 223, 224 (one run each)
surpriseMeasureForGamma = MIS / CC / BF / (useSurpriseUpdating=false)
```

**Run:**
```bash
python scripts/run_ablation.py run --ablations abl2_pc
```

**Output location:** `output_dir/results/abl2_pc/<surprise>/pc0.25_seed<N>/`

**Summary CSV:** `output_dir/results/summary_abl2_pc_avg.csv`  
Filter to `level = 0.25`. The four rows (one per surprise value) map to the four method columns in Table III.

**Verify against Table III:**
- MIS row: `mean_MEC` ≈ 18.809, `median_MEC` ≈ 19.293
- CC row: `mean_MEC` ≈ 19.823, `median_MEC` ≈ 20.429 (exceeds 20 C threshold)
- BF row: `mean_MEC` ≈ 19.992, `median_MEC` ≈ 20.484
- None row: `mean_MEC` ≈ 18.850, `median_MEC` ≈ 19.423

---

## Table IV — Dual Link Failure Scenario

**What the table shows:** MEC/RPL statistics when links 12-7 and 12-3 are disabled for the first 100 timesteps then restored, across all surprise measures.

**Ablation:** `abl5_disaster`, scenario `fail100_12-7_12-3`

**Run:**
```bash
python scripts/run_ablation.py run --ablations abl5_disaster
```

**Output location:** `output_dir/results/abl5_disaster/<surprise>/fail100_12-7_12-3_seed<N>/`

**Summary CSV:** `output_dir/results/summary_abl5_disaster_avg.csv`  
Filter to rows where `level` contains `fail100_12-7_12-3`.

**Verify against Table IV:**
- All methods exceed mecThreshold=20 in this scenario (performance ceiling due to network routing constraints, not adaptive failure).
- BF shows substantially worse performance, consistent with the paper's analysis.

---

## Table V — Lambda Ablation

**What the table shows:** Mean and median MEC for MIP, CC, and no-surprise configurations across lambda ∈ {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0}.

**Ablation:** `abl1_lambda`

**Exact configuration (fixed parameters):**
```
p_c          = 0.5
lookback     = 4
mecThreshold = 20
rplThreshold = 0.2
```

**Run:**
```bash
python scripts/run_ablation.py run --ablations abl1_lambda
```

**Output location:** `output_dir/results/abl1_lambda/<surprise>/lam<value>_seed<N>/`

**Summary CSV:** `output_dir/results/summary_abl1_lambda_avg.csv`

**Verify against Table V:**
- At lambda=20: all methods collapse to mean_MEC > 41 C (policy degrades to near-uniform).
- At lambda ∈ {0.1 … 5.0}: MIP maintains mean_MEC between 19.17 and 19.52.
- CC at lambda=5.0 and 20.0: median_MEC exceeds 20 C threshold (marked † and * in paper).

---

## Table VI — NFR Threshold Tightening (Level 17 C / 0.17)

**What the table shows:** MEC/RPL statistics for all surprise measures when NFR thresholds are tightened to mecThreshold=17, rplThreshold=0.17.

**Ablation:** `abl4_nfr`, level `(17, 0.17)`

**Exact configuration (fixed parameters):**
```
algorithmType = erperseus
lambda        = 1.0
p_c           = 0.5
lookback      = 4
```

**Run:**
```bash
python scripts/run_ablation.py run --ablations abl4_nfr
```

**Output location:** `output_dir/results/abl4_nfr/<surprise>/mec17_rpl0.17_seed<N>/`

**Summary CSV:** `output_dir/results/summary_abl4_nfr_avg.csv`  
Filter to `level = (17, 0.17)`.

**Verify against Table VI:**
- MIS: `mean_MEC` ≈ 16.302, `mean_RPL` ≈ 14.587 — both within the tightened thresholds (conjunctive satisfaction achieved).
- CC: `median_MEC` ≈ 20.103 — exceeds threshold; conjunctive satisfaction not achieved.
- BF and None: similarly non-compliant.

---

## Running Individual Ablations

To run only a specific ablation:

```bash
python scripts/run_ablation.py run --ablations abl1_lambda
python scripts/run_ablation.py run --ablations abl2_pc
python scripts/run_ablation.py run --ablations abl3_lookback
python scripts/run_ablation.py run --ablations abl4_nfr
python scripts/run_ablation.py run --ablations abl5_disaster
```

To limit to specific surprise measures (e.g. MIS and CC only):
```bash
python scripts/run_ablation.py run --ablations abl2_pc --surprise MIS CC
```

---

## Generating Charts for Individual Runs

After any run completes, generate charts for that specific output directory:

```bash
python createCharts.py \
  --output-dir output_dir/results/abl2_pc/MIS/pc0.25_seed222 \
  --mec-threshold 20 \
  --rpl-threshold 0.2
```

Use the `mecThreshold` and `rplThreshold` values that match the config used for that run.

---

## Interpreting Output Files

| File | Format | How to use |
|---|---|---|
| `MECSattimestep.txt` | Two columns: `timestep value` | Column 2 mean = mean_MEC for that run |
| `RPLSattimestep.txt` | Two columns: `timestep value` | Column 2 mean = mean_RPL for that run |
| `MISBounds.txt` | Three columns: `timestep lower upper` | Theorem 1 confidence interval; used in MIP bounds figure |
| `gamma.txt` | One column per mote, one row per timestep | SMiLe mixing factor time series |
| `surpriseMIS.txt` | Same shape as gamma.txt | MIP surprise value time series |
| `summary_*_avg.csv` | One row per (level, surprise); averaged across 3 seeds | Primary input for all paper tables |

---

## Minimum Sanity Check (~10 minutes)

To verify the code runs end-to-end without running full ablations:

1. Edit `src/solver.config`: set `algorithmType=erperseus`, `lambda=1.0`, `surpriseMeasureForGamma=MIS`, `runSeed=222`, `mecThreshold=20`, `rplThreshold=0.2`, `useSurpriseUpdating=true`.
2. Run:
   ```bash
   java -cp ".:bin:libraries/*" main.SolvePOMDP
   ```
3. Verify that `output_dir/MECSattimestep.txt` and `output_dir/RPLSattimestep.txt` exist and contain 500 rows.
4. Run:
   ```bash
   python createCharts.py --output-dir output_dir --mec-threshold 20 --rpl-threshold 0.2
   ```
5. Verify that MEC and RPL satisfaction plots open in the browser.
