# accuracy/

Runs lm-eval over the ACAS Python implementations (Hugging Face modeling files with the
controller inside the forward pass) on any CUDA Linux box.
Setup: [INSTALL.md](../docs/INSTALL.md). Reproduction: [REPRODUCE.md](../docs/REPRODUCE.md).

## Files

| File | Role |
|---|---|
| `setup_relu.sh` / `setup_silu.sh` | one-time per box: pinned venvs, model download, stock-file backups |
| `run_bench.sh` | **the per-row runner**: `--method × --signal × --controller × --cadence × --task` |
| `code/` | the 8 modeling files (4 methods × {fixed, mimd cadence}) + CATS-GP calibration CSV |
| `lm-evaluation-harness/` | bundled patched lm-eval: adds the BOS token ProSparse's tokenizer omits; fixes an import crash under the ReLU transformers pin |
| `interval_ablation/` | Table 5: `sweep_interval.sh`, `run_interval_point.sh` |

Runs land in `runs/<name>/` (gitignored) with the lm-eval results JSON, run.log, and
controller tracking logs.

## The per-row command (Tables 1–2 accuracy)

```bash
bash run_bench.sh --method {si|grasp|cats|catsgp} \
                  --signal {precision|activation|baseline|contrib} \
                  --controller {pid|rl} --cadence {fixed|mimd} \
                  --task {gsm8k|bbh|mbpp}
```

Row → flags: **Signal** column Prec./L2 = the per-method proxy signal
(`precision` for si/grasp, `activation` for cats, `baseline` for catsgp), Contrib. =
`contrib` (default) · **Ctrl** = `--controller` · `*` = `--cadence mimd` (no star = `fixed`).

Targets default to the paper's operating points:

| Method | proxy signal rows | contrib rows |
|---|---|---|
| si | precision 0.995 | 0.986 |
| grasp | precision 0.997 | 0.96 |
| cats | activation (L2) 0.80 | 0.89 |
| catsgp | baseline (L2 0.80 + wrong-skip 0.05) | `--target 0.89 --gate-target 0.89` |

Smoke test: `bash run_bench.sh --method si --task gsm8k --limit 3`.

## Table 5 (interval ablation)

```bash
bash interval_ablation/sweep_interval.sh              # the full (Nmin,Nmax) grid, all three tasks
bash interval_ablation/run_interval_point.sh 25 500   # one point (parallelizable across GPUs)
```
