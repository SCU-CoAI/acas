# Reproducing the paper tables

Every ACAS table row is one setting of three toggles:

| Table column | Accuracy runner | C++ forks |
|---|---|---|
| **Ctrl** = PID / RL | `--controller pid\|rl` | `ACAS_CONTROLLER` |
| **`*`** superscript | `--cadence mimd` (no `*` = `fixed`) | `ACAS_CADENCE` |
| **Signal** | `--signal` (see below) | `ACAS_SIGNAL` |

The signal names differ between the two pipelines, and on the Python side they differ per
method:

| Table column | Accuracy `--signal` | C++ `ACAS_SIGNAL` | Applies to |
|---|---|---|---|
| **Prec.** | `precision` | `precision` | ACAS-SI, ACAS-GR |
| **L2** | `activation` | `l2` | ACAS-CATS |
| **L2** | `baseline` | `l2` | ACAS-CATS-GP |
| **Contrib.** | `contrib` | `contrib` | all four |

Operating points:

| Method | Prec./L2 rows | Contrib. rows |
|---|---|---|
| ACAS-SI | precision 0.995 | 0.986 |
| ACAS-GR | precision 0.997 | 0.96 (0.985 for the BBH) |
| ACAS-CATS | L2 0.80 | 0.89 |
| ACAS-CATS-GP | L2 0.80 | 0.89 |

ACAS-CATS-GP carries a second knob, `--gate-target` / `ACAS_GATE_TARGET`, whose meaning
follows the signal: **0.05** in the L2 rows (a wrong-skip target) and **0.89** in the
contrib rows (a contrib-drift target).

Apart from the ACAS-GR BBH cells, every published row runs at its default target. The
accuracy runner picks the default from `--method` and `--signal`, so no `--target` is
needed.

Implementation constants (as deployed):

**Error signal.** Eq. (7) exactly: logit difference, sign · sigmoid. Sensitivity λ = 5;
inputs clamped to [1e-10, 1−1e-10]. Identical in the Python and C++ implementations.

**PID gains** (Kp, Ki, Kd):

| Knob | Gains |
|---|---|
| ReLU methods | 1.0, 0.2, 0.15 |
| SiLU activation | 0.5, 0.1, 0.05 |
| CATS-GP gate prediction | 1.0, 0.2, 0.15 |

**RL.** η = 0.3, γ = 0.85, ε decays 0.30 → 0.02 over 200 updates. State is the error binned
at ±{1, 2, 3}σ, with σ taken over the last 40 checks; reward is −|e|. Action sets:

| Methods | Actions |
|---|---|
| ReLU | −8, −3, −1, 0, 1, 3, 8 |
| SiLU | −5, −0.7, −0.1, 0, 0.1, 0.7, 5 (finer steps for the multiplicative threshold knob) |

**MIMD.** N₀ = 200, γ_inc = 1.5, γ_dec = 0.5, tolerance 0.01, bounds Nmin = 50 /
Nmax = 500 (Alg. 1). Bounds are overridable with `ACAS_NMIN` / `ACAS_NMAX` in every C++
fork and every Python MIMD file.

---

## Tables 1–2: accuracy columns (GSM8K / BBH / MBPP)

Pipeline: `accuracy/` on any CUDA box (setup: [INSTALL.md](INSTALL.md)). One `run_bench.sh`
command per row × task.

```bash
cd accuracy
# ACAS rows: --signal x --controller x --cadence  (defaults: contrib / pid / mimd)
bash run_bench.sh --method si     --signal precision --cadence fixed --controller pid --task gsm8k   # SI Prec. PID
bash run_bench.sh --method si     --signal precision --cadence mimd  --controller pid --task gsm8k   # SI Prec. PID*
bash run_bench.sh --method si     --cadence fixed --controller pid --task gsm8k                      # SI Contrib. PID
bash run_bench.sh --method si     --cadence mimd  --controller pid --task gsm8k                      # SI Contrib. PID*
bash run_bench.sh --method si     --cadence fixed --controller rl  --task gsm8k                      # SI Contrib. RL
bash run_bench.sh --method si     --cadence mimd  --controller rl  --task gsm8k                      # SI Contrib. RL*
# same pattern for:
#   --method grasp   (BBH contrib cells: add --target 0.985)
#   --method cats    (--signal activation for the L2 rows)
#   --method catsgp  (--signal baseline for the L2 + wrong-skip rows)
# tasks: gsm8k | bbh | mbpp   (run all three per row for the table's Avg.)
```

Reading the numbers: lm-eval prints its results table at the end of every run (kept in
`accuracy/runs/<name>/run.log`, with the full results JSON alongside):
- **GSM8K** = `exact_match,flexible-extract` (8-shot) × 100
- **BBH** = the `bbh_cot_fewshot` aggregate `exact_match` × 100
- **MBPP** = `pass@1` (3-shot) × 100 — the runner sets `HF_ALLOW_CODE_EVAL=1`

### Baseline rows

**No Sparsity (dense)**: plain lm-eval on the stock model, one row per model family, each
in that family's method venv (the ReLU venvs carry the bundled patched lm-eval that
supplies the BOS token ProSparse's tokenizer omits):

```bash
# ReLU dense (ProSparse) — pairs with the ACAS-SI / ACAS-GR rows
~/acas_venv_si/bin/python -m lm_eval --model hf \
  --model_args pretrained=$HOME/prosparse-si,dtype=float16,trust_remote_code=True \
  --tasks gsm8k --num_fewshot 8 --batch_size 1

# SiLU dense (Llama-3.1-8B-Instruct) — pairs with the ACAS-CATS / ACAS-CATS-GP rows
~/acas_venv_cats/bin/python -m lm_eval --model hf \
  --model_args pretrained=$HOME/Llama-3.1-8B-Instruct,dtype=float16 \
  --tasks gsm8k --num_fewshot 8 --batch_size 1
```

**SparseInfer / Grasp / CATS-50% (static)**: the static configurations are the paper's
"SparseInfer alpha=1.03" (1.03 for the first 20 layers, 1.0 for the rest), uniform
alpha 1.0 (Grasp), and the 50th-percentile calibration thresholds. The ReLU modeling files
run frozen (`--cadence fixed`) with `ACAS_SI_FREEZE=1` / `ACAS_GRASP_FREEZE=1` and
`ACAS_*_INIT_ALPHAS=<json>` (a JSON list of per-layer integer alphas, 10000 = α 1.0).

CATS-50% loads its thresholds through `ACAS_FROZEN_THRESHOLDS`, which parses **JSON**
(`{layer: threshold}` or a list), so the calibration file at
`efficiency/configs/cats/static_cats50.txt` is converted first:

```bash
python3 -c 'import json; ls=[l.split()[0] for l in open("../efficiency/configs/cats/static_cats50.txt") if l.strip() and not l.startswith("#")]; json.dump({str(i):float(v) for i,v in enumerate(ls)}, open("/tmp/cats50.json","w"))'
ACAS_FROZEN_THRESHOLDS=/tmp/cats50.json \
  bash run_bench.sh --method cats --signal activation --cadence fixed --task gsm8k --name cats50_static_gsm8k
```

## Tables 1–2: efficiency columns (TPOT / Power / Energy)

Pipeline: `efficiency/` on the Jetson (setup: [INSTALL.md](INSTALL.md)).

```bash
mkdir -p efficiency/results
tmux new -d -s sweep 'bash efficiency/sweep_table.sh > efficiency/results/sweep.log 2>&1'   # 24 ACAS configs
tmux new -d -s base  'bash efficiency/baselines.sh  > efficiency/results/baselines.log 2>&1' # dense + static rows
```

### How each run is measured

Both scripts call `efficiency/measure_power_latency.py` once per run. It:

- samples the Jetson's INA3221 power rails every 25 ms while `llama-cli` generates;
- takes prefill, TPOT, and token counts from llama.cpp's own timing output;
- averages power over the generation window only.

The generation window is llama.cpp's reported eval time, anchored at the end of power
activity so prefill and the idle tail are excluded from the average.

Energy per token is then `avg power × TPOT` (W × ms = mJ).

### Reading the output

Results land in `efficiency/results/sweep_table.tsv`, one row per rep:

```
label · prefill_ms · tpot_ms · eval_tokens · avg_power_w · energy_per_tok_mj · …
```

**Median the reps for each label** (six reps by default; `N=` to change), then map columns
to the table:

| TSV column | Table column |
|---|---|
| `tpot_ms` | TPOT |
| `avg_power_w` | Power |
| `energy_per_tok_mj` | Energy/Tok |

Speedup is `dense TPOT / row TPOT`, using the dense row for the **same model family**:
`DENSE prosparse` for the ACAS-SI and ACAS-GR rows, `DENSE llama31` for the ACAS-CATS and
ACAS-CATS-GP rows.

### Label → row

Labels map 1:1 to table rows:

| Label pattern | Row |
|---|---|
| `<M> contrib PID` / `PID*` / `RL` / `RL*` | that method's four Contrib. rows |
| `<M> ANCHOR precPID` / `<M> ANCHOR L2PID` | the Prec. (ReLU) / L2 (SiLU) PID row |
| `<M> prec PID*` / `<M> L2 PID*` | the starred Prec. / L2 row |
| `DENSE …` (from `baselines.sh`) | No Sparsity |
| `BASE …` (from `baselines.sh`) | the static baseline rows |

where `<M>` is `SI`, `GR`, `CATS`, or `CATSGP`. For example, `SI contrib RL*` is
ACAS-SI · Contrib. · RL\*.

## Tables 3–4: per-token decode latency

Pipeline: `efficiency/per_token_latency/` on the Jetson.

```bash
bash efficiency/per_token_latency/run_token_timing.sh        # contrib rows (4 methods)
bash efficiency/per_token_latency/run_token_timing_prec.sh   # Prec./L2 rows
python3 efficiency/per_token_latency/analyze_acas_token.py   # -> sparse vs dense-update latency (mean/p50/p95/p99)
```

The analyzer joins per-token latency (`ptok_*.csv`, from llama.cpp's own per-token counter)
with the update markers (`tok_*.csv`) and splits normal (sparse) vs dense-update tokens.
Baselines are omitted from these tables (no update events).

## Table 5: flexible update period bounds (interval ablation)

The MIMD bounds are env-configurable: `ACAS_NMIN` / `ACAS_NMAX` (defaults 50/500 = Algorithm 1).

Accuracy side (`accuracy/`, ACAS-SI, contrib PID\*):

```bash
cd accuracy
ACAS_NMIN=25 ACAS_NMAX=500 bash run_bench.sh --method si --cadence mimd --task gsm8k --name si_mimd_nmin25_nmax500_gsm8k
# repeat per (Nmin,Nmax) grid point x {gsm8k,bbh,mbpp};
# accuracy/interval_ablation/{sweep_interval.sh, run_interval_point.sh} script the grid
```

The same env vars drive the C++ forks, so the latency/power side of the ablation runs
through `efficiency/sweep_table.sh` commands with `ACAS_NMIN`/`ACAS_NMAX` prefixed.
