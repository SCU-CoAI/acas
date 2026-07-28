# efficiency/

Reproduces the efficiency side of the paper on a Jetson AGX Orin (INA3221 power rails).
Setup: [INSTALL.md](../docs/INSTALL.md). Reproduction: [REPRODUCE.md](../docs/REPRODUCE.md).

| Entry point | Produces |
|---|---|
| `sweep_table.sh` | Tables 1–2 TPOT/Power/Energy, all 24 ACAS configs, one session |
| `baselines.sh` | the dense (No Sparsity) + static baseline rows, same TSV/frame |
| `per_token_latency/run_token_timing.sh` | Tables 3–4 contrib rows (per-token CSVs) |
| `per_token_latency/run_token_timing_prec.sh` | Tables 3–4 Prec./L2 rows |
| `per_token_latency/analyze_acas_token.py` | the Tables 3–4 stats from those CSVs |
| `measure_power_latency.py` | shared single-run power+latency measurement (used by the sweeps) |

- `configs/<method>/` — the operating-point alpha/threshold files for every table row,
  passed to every fork via `ACAS_CONFIG=<file>`.
- `gsm8k_0shot-prompt.txt` — the fixed generation prompt used by every measurement.
- Outputs land under `results/` (gitignored); compare against the published paper's tables.
- Model paths default to `~/prosparse-llama-2-7b.gguf` / `~/llama-3.1-8b-instruct.gguf`;
  override with `PROSPARSE=` / `LLAMA=`. Binary paths override with `SI_BIN=` / `GRASP_BIN=`
  / `CATS_BIN=` / `CATSGP_BIN=` / `DENSE_CLI=`.
