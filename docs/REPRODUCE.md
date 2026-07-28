# Reproducing the paper tables

Every ACAS table row is a setting of the three toggles. The table columns map directly:

| Table column | Knob |
|---|---|
| **Signal** = Prec. (ReLU) / L2 (SiLU) | proxy signal: `precision` / `l2` (C++) — `precision` / `activation` / `baseline` (Python, per method) |
| **Signal** = Contrib. | `contrib` |
| **Ctrl** = PID / RL | `pid` / `rl` |
| **`*`** superscript | MIMD cadence (`mimd` in C++, `--cadence mimd` in the accuracy runner; no `*` = `fixed`) |

Operating points:

| Method | proxy-signal target | contrib target |
|---|---|---|
| ACAS-SI | precision 0.995 | 0.986 |
| ACAS-GR | precision 0.997 | 0.96 (BBH cells: 0.985) |
| ACAS-CATS | L2 0.80 | 0.89 |
| ACAS-CATS-GP | L2 0.80 (pred 0.05 wrong-skip) | 0.89 / pred 0.89 |

Implementation constants (as deployed):
- Error signal: Eq. (7) exactly (logit difference, sign·sigmoid), sensitivity λ=5, metric
  clamped to [1e-10, 1-1e-10]. Identical in the Python and C++ implementations.
- PID gains: ReLU methods (Kp,Ki,Kd) = (1.0, 0.2, 0.15); SiLU activation knob
  (0.5, 0.1, 0.05); CATS-GP gate-prediction knob (1.0, 0.2, 0.15).
- RL: η=0.3, γ=0.85, ε: 0.30→0.02 over 200 updates, state = error binned at ±{1,2,3}σ
  (σ over the last 40 checks), reward −|e|. Action sets: ReLU methods {−8,−3,−1,0,1,3,8}; SiLU methods the per-method-scaled set
  (−5,−0.7,−0.1,0,0.1,0.7,5) (finer steps for the multiplicative threshold knob).
- MIMD: N0=200, γinc=1.5, γdec=0.5, tolerance 0.01, bounds Nmin=50 / Nmax=500 (Alg. 1),
  overridable via `ACAS_NMIN`/`ACAS_NMAX` (all C++ forks and all Python MIMD files).

---

## Tables 1–2 — accuracy columns (GSM8K / BBH / MBPP)

Pipeline: `accuracy/` on any CUDA box (setup: [INSTALL.md](INSTALL.md)). One `run_bench.sh` command
per row × task (below); runs are independent, so they parallelize across GPUs with `--gpu N`.

```bash
cd accuracy
# ACAS rows: --signal x --controller x --cadence  (defaults: contrib / pid / mimd)
bash run_bench.sh --method si     --signal precision --cadence fixed --controller pid --task gsm8k   # SI Prec. PID
bash run_bench.sh --method si     --signal precision --cadence mimd  --controller pid --task gsm8k   # SI Prec. PID*
bash run_bench.sh --method si     --cadence fixed --controller pid --task gsm8k                      # SI Contrib. PID
bash run_bench.sh --method si     --cadence mimd  --controller pid --task gsm8k                      # SI Contrib. PID*
bash run_bench.sh --method si     --cadence fixed --controller rl  --task gsm8k                      # SI Contrib. RL
bash run_bench.sh --method si     --cadence mimd  --controller rl  --task gsm8k                      # SI Contrib. RL*
# same pattern for: --method grasp | cats (--signal activation for the L2 rows) | catsgp (--signal baseline for the CATS-GP L2+wrong-skip rows; contrib uses --target/--gate-target)
# tasks: gsm8k | bbh | mbpp   (run all three per row for the table's Avg.)
```

Reading the numbers: lm-eval prints its results table at the end of every run (kept in
`accuracy/runs/<name>/run.log`, with the full results JSON alongside):
- **GSM8K** = `exact_match,flexible-extract` (8-shot) × 100
- **BBH** = the `bbh_cot_fewshot` aggregate `exact_match` × 100
- **MBPP** = `pass@1` (3-shot) × 100 — the runner sets `HF_ALLOW_CODE_EVAL=1`

Baseline rows:
- **No Sparsity (dense)**: plain lm-eval on the stock model with the same venv, e.g.
  `~/acas_venv_si/bin/python -m lm_eval --model hf --model_args pretrained=$HOME/prosparse-si,dtype=float16,trust_remote_code=True --tasks gsm8k --num_fewshot 8 --batch_size 1`.
- **SparseInfer / Grasp / CATS-50% (static)**: the static configurations are the paper's
  "SparseInfer alpha=1.03" (1.03 for the first 20 layers, 1.0 for the rest), uniform
  alpha 1.0 (Grasp), and the 50th-percentile calibration thresholds
  (`efficiency/configs/cats/static_cats50.txt`). On the efficiency side `baselines.sh`
  generates the alpha files inline. For accuracy, the ReLU modeling
  files run frozen (`--cadence fixed`) with `ACAS_SI_FREEZE=1` / `ACAS_GRASP_FREEZE=1`
  (+ `ACAS_*_INIT_ALPHAS=<json>`).

## Tables 1–2 — efficiency columns (TPOT / Power / Energy)

Pipeline: `efficiency/` on the Jetson (setup: [INSTALL.md](INSTALL.md)). Two scripts, one session:

```bash
mkdir -p efficiency/results
tmux new -d -s sweep 'bash efficiency/sweep_table.sh > efficiency/results/sweep.log 2>&1'   # 24 ACAS configs
tmux new -d -s base  'bash efficiency/baselines.sh  > efficiency/results/baselines.log 2>&1' # dense + static rows
```

Both scripts measure each run with `efficiency/measure_power_latency.py`: it samples the
Jetson's INA3221 power rails every 25 ms while `llama-cli` generates, takes prefill /
TPOT / token counts from llama.cpp's own timing output, and averages power over the
generation window (window length = llama.cpp's reported eval time, anchored at the end
of power activity, so prefill and idle tails are excluded). Energy/Tok = avg power ×
TPOT (W × ms = mJ).

Output: `efficiency/results/sweep_table.tsv`, one row per rep with columns
`label · prefill_ms · tpot_ms · eval_tokens · avg_power_w · energy_per_tok_mj · …`.
Median the reps per label. Column map: `tpot_ms` → **TPOT** (speedup = dense TPOT / row TPOT),
`avg_power_w` → **Power**, `energy_per_tok_mj` → **Energy/Tok**.
Labels map 1:1 to rows (e.g. `SI contrib RL*` = ACAS-SI · Contrib. · RL\*; `SI ANCHOR precPID` / `CATS ANCHOR L2PID` =
the Prec./L2 PID rows; `DENSE …`/`BASE …` from baselines.sh = the No Sparsity / static rows).

## Tables 3–4 — per-token decode latency

Pipeline: `efficiency/per_token_latency/` on the Jetson.

```bash
bash efficiency/per_token_latency/run_token_timing.sh        # contrib rows (4 methods)
bash efficiency/per_token_latency/run_token_timing_prec.sh   # Prec./L2 rows
python3 efficiency/per_token_latency/analyze_acas_token.py   # -> sparse vs dense-update latency (mean/p50/p95/p99)
```

The analyzer joins per-token latency (`ptok_*.csv`, from llama.cpp's own per-token counter)
with the update markers (`tok_*.csv`) and splits normal (sparse) vs dense-update tokens.
Output is raw milliseconds. Baselines are omitted from these tables (no update events).

## Table 5 — flexible update period bounds (interval ablation)

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
