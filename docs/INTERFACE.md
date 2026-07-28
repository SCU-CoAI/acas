# ACAS environment interface

All four methods read the **same 8 `ACAS_*` variables**. Three are the configuration toggles (signal × controller × cadence); the rest are setpoints / plumbing.

## The 8 knobs

| Variable | Values (default) | Meaning |
|---|---|---|
| `ACAS_SIGNAL` | `precision`/`l2` \| `contrib`  (default precision/l2) | Which quality signal the controller watches: `precision` (ReLU) / `l2` (SiLU) is each method's cheap proxy signal (the tables' Prec./L2 rows); `contrib` is the output-contribution drift metric (the Contrib. rows). |
| `ACAS_CONTROLLER` | `pid` \| `rl`  (default `pid`) | How signal-error becomes a threshold/alpha step. `pid` = BufferedPID; `rl` = tabular Q-learning. |
| `ACAS_CADENCE` | `fixed` \| `mimd`  (default `fixed`) | How often the online (dense-reference) check fires. `fixed` = the `ACAS_DENSE_UPDATE_PERCENT` sampling; `mimd` = per-layer adaptive interval |
| `ACAS_TARGET` | float in (0,1) (default 0.995 ReLU / 0.99 SiLU) | Setpoint for the activation knob. Applies to whichever `ACAS_SIGNAL` is active: the controller drives that quality metric (precision/L2 or contrib) toward this value. |
| `ACAS_GATE_TARGET` | float in [0,1) | **CATS-GP only.** Gate-prediction knob setpoint (the paper's th_gate; `ACAS_TARGET` drives th_act); routed to the active signal's gate target (contrib-drift target in contrib mode, wrong-skip target in L2 mode). |
| `ACAS_DENSE_UPDATE_PERCENT` | float % (default 0.5) | Under `ACAS_CADENCE=fixed`, the probability (in %) that a generated token triggers a dense-reference check — 0.5 means on average one check per 200 tokens. Ignored under `mimd`, which schedules checks by its own per-layer interval. |
| `ACAS_CONFIG` | path | Per-layer config file: initial alphas (ReLU forks, one value per line) / thresholds (SiLU forks). |
| `ACAS_LOG` | `0`/`1` (default 0) | Per-layer controller debug to stderr. |

## Per-method support

| Knob | sparseinfer | grasp | cats | cats-gp |
|---|:-:|:-:|:-:|:-:|
| `ACAS_SIGNAL` (precision/l2/contrib) | ✓ | ✓ | ✓ | ✓ |
| `ACAS_CONTROLLER` (pid/rl) | ✓ | ✓ | ✓ | ✓ |
| `ACAS_CADENCE` (fixed/mimd) | ✓ | ✓ | ✓ | ✓ |
| `ACAS_TARGET` | ✓ | ✓ | ✓ | ✓ |
| `ACAS_GATE_TARGET` | — | — | — | ✓ |
| `ACAS_DENSE_UPDATE_PERCENT` | ✓ | ✓ | ✓ | ✓ |
| `ACAS_CONFIG` | ✓ | ✓ | ✓ | ✓ |
| `ACAS_LOG` | ✓ | ✓ | ✓ | ✓ |

## MIMD cadence (`ACAS_CADENCE=mimd`)

Per layer, a check interval starts at 200 tokens and adapts on `|metric − target|`: shrinks ×0.5 when
drifting (`> 0.01`), grows ×1.5 when stable, bounded to `[50, 500]`. cats-gp adapts on `max(activation_err, predictor_err)`. Decode-only; constants match
the Python reference. Bounds are overridable with `ACAS_NMIN` / `ACAS_NMAX`.