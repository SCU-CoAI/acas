#!/usr/bin/env python3
"""
Measure ONE inference run on the Jetson AGX Orin: latency AND power/energy in a single
pass. Samples the INA3221 power rails while the llama.cpp fork runs, captures its stderr,
and parses the timing — so every table column comes from the SAME run under identical
conditions.

Emits one TSV row (to --tsv, appended) with:
  label  prefill_ms  tpot_ms  eval_tokens  avg_power_w  energy_per_tok_mj  gen_window_s  duration_s  n_samples

Usage (driven by sweep_table.sh / baselines.sh; can also run standalone):
  python3 measure_power_latency.py --label "ACAS-SI fixed:pid" --interval 0.025 \
      --tsv results/sweep_table.tsv \
      --cmd "$SI_BIN -m $PROSPARSE -ngl 33 -p \"$PROMPT\" -n 256 -no-cnv"

Self-contained (stdlib only), so it runs anywhere on the Jetson.
"""
import argparse
import os
import pathlib
import re
import subprocess
import threading
import time

# ---- INA3221 power rails (Jetson AGX Orin) : ch1=GPU, ch2=CPU, ch3=SYS ----
_BASE_CANDIDATES = [
    pathlib.Path("/sys/bus/i2c/devices/1-0040/hwmon/hwmon1"),
    pathlib.Path("/sys/bus/i2c/devices/1-0040/hwmon/hwmon0"),
]
CHANNELS = {"gpu": 1, "cpu": 2, "sys": 3}
_BASE = None  # resolved lazily on first read, so this module imports on non-Jetson hosts too


def _ina_base() -> pathlib.Path:
    global _BASE
    if _BASE is None:
        _BASE = next((b for b in _BASE_CANDIDATES if b.exists()), None)
        if _BASE is None:
            raise SystemExit("Could not find INA3221 hwmon path under /sys/bus/i2c/devices/1-0040/hwmon/")
    return _BASE


def read_power_mw() -> dict:
    base = _ina_base()
    p = {}
    for name, ch in CHANNELS.items():
        mv = int((base / f"in{ch}_input").read_text())
        ma = int((base / f"curr{ch}_input").read_text())
        p[name] = mv * ma / 1000.0  # mW
    p["total"] = sum(p.values())
    return p


# ---- llama.cpp timing parse (handles llama_print_timings & llama_perf_context_print) ----
def parse_timing(stderr: str):
    """Return (prefill_ms, tpot_ms, eval_tokens)."""
    prefill_ms = None
    m = re.search(r"prompt eval time\s*=\s*([0-9.]+)\s*ms", stderr)
    if m:
        prefill_ms = float(m.group(1))

    # generation eval line: the LAST 'eval time ... ( X ms per token)' (prompt-eval prints first)
    tpot_ms, eval_tokens = None, None
    eval_lines = [ln for ln in stderr.splitlines()
                  if "eval time" in ln and "ms per token" in ln and "prompt eval time" not in ln]
    if eval_lines:
        ln = eval_lines[-1]
        mt = re.search(r"\(\s*([0-9.]+)\s*ms per token", ln)
        if mt:
            tpot_ms = float(mt.group(1))
        mn = re.search(r"=\s*[0-9.]+\s*ms\s*/\s*([0-9]+)\s*(?:runs|tokens)", ln)
        if mn:
            eval_tokens = int(mn.group(1))
    return prefill_ms, tpot_ms, eval_tokens


def steady_window(times, watts, gen_s=None):
    """Indices [lo, hi) of the steady GENERATION plateau — drops the model-load lull at the
    front (the ~13s of loading the GGUF, low power) and any tail, so avg/std power is over the
    generation phase only. The threshold detection (on a 0.5s-smoothed copy) is used ONLY to
    locate the plateau boundaries; the returned window indexes the RAW samples, contiguous —
    no interior sample (e.g. a dense-update power spike) is ever dropped from the average. When gen_s (llama.cpp's own eval_tokens x TPOT) is given, the window LENGTH
    is taken exactly from it, anchored at the detected end of generation; the percentile
    heuristic + 5% edge trims are only the fallback when timing is unavailable. (Fallback
    validated against llama.cpp's clock: window/(tokens x TPOT) median 0.893, IQR
    0.884-0.900 across the 168 reference runs = plateau minus the trims, as designed.)"""
    n = len(watts)
    if n < 5:
        return 0, n
    dt = (times[-1] - times[0]) / max(1, n - 1)
    w = max(1, int(0.5 / dt)) if dt > 0 else 1
    sm = []
    for i in range(n):
        a = max(0, i - w // 2); b = min(n, i + w // 2 + 1)
        sm.append(sum(watts[a:b]) / (b - a))
    srt = sorted(sm)
    mid = (srt[int(0.25 * (n - 1))] + srt[int(0.75 * (n - 1))]) / 2
    active = [x > mid for x in sm]
    if not any(active):
        return 0, n
    lo = active.index(True)
    hi = n - active[::-1].index(True)
    if gen_s and gen_s > 0:
        # exact-length window from llama.cpp's own generation duration, ending at the last
        # active sample (end of generation = where power collapses); clamped to the plateau.
        lo_t = times[hi - 1] - gen_s
        lo_exact = next((i for i in range(lo, hi) if times[i] >= lo_t), lo)
        return lo_exact, hi
    trim = int((hi - lo) * 0.05)
    return lo + trim, hi - trim


def main():
    ap = argparse.ArgumentParser(description="Measure latency + power/energy for one Jetson inference run")
    ap.add_argument("--cmd", required=True, help="full shell command to run (the llama.cpp fork invocation)")
    ap.add_argument("--label", required=True, help="config label for the output row")
    ap.add_argument("--interval", type=float, default=0.025, help="power sampling interval s (default 0.025 = 25ms)")
    ap.add_argument("--tsv", default=None, help="append a result row to this TSV (header written if new)")
    ap.add_argument("--csv", default=None, help="optional: also dump the raw power trace to this CSV")
    ap.add_argument("--n-gen", type=int, default=0, help="fallback generated-token count if stderr lacks it")
    args = ap.parse_args()

    stop = threading.Event()
    samples = []  # (t, power_dict)

    def sampler():
        t0 = time.time()
        while not stop.is_set():
            try:
                samples.append((time.time() - t0, read_power_mw()))
            except Exception:
                pass
            time.sleep(args.interval)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t_start = time.time()
    proc = subprocess.run(args.cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - t_start
    stop.set(); th.join()

    if proc.returncode != 0:
        print(f"WARNING [{args.label}]: command exited {proc.returncode}")
    if len(samples) < 2:
        raise SystemExit(f"ERROR [{args.label}]: too few power samples ({len(samples)})")

    prefill_ms, tpot_ms, eval_tokens = parse_timing(proc.stderr)
    if eval_tokens is None:
        eval_tokens = args.n_gen or 0

    # avg/std power over the steady GENERATION window only (drops the model-load lull),
    # matching the paper's table. Energy/token = Avg Power x TPOT (W x ms = mJ), NOT a
    # whole-run integral — that's how every existing table row was computed.
    times = [t for t, _ in samples]
    watts_all = [p["total"] / 1000.0 for _, p in samples]
    gen_s = (tpot_ms * eval_tokens / 1000.0) if (tpot_ms and eval_tokens) else None
    lo, hi = steady_window(times, watts_all, gen_s)
    watts = watts_all[lo:hi] or watts_all
    avg_w = sum(watts) / len(watts)
    e_per_tok_mj = (avg_w * tpot_ms) if tpot_ms is not None else float("nan")  # W*ms = mJ
    win_s = times[hi - 1] - times[lo] if hi > lo else 0.0

    if args.csv:
        cols = ["gpu", "cpu", "sys", "total"]
        with open(args.csv, "w") as f:
            f.write("t_sec," + ",".join(cols) + "\n")
            for t, p in samples:
                f.write(f"{t:.4f}," + ",".join(f"{p[c]:.1f}" for c in cols) + "\n")
        # Prefill (from llama.cpp's prompt-eval time) written next to the raw trace, in seconds,
        # for downstream power-trace tooling.
        if prefill_ms is not None:
            with open(args.csv + ".prefill", "w") as f:
                f.write(f"{prefill_ms / 1000.0:.6f}\n")

    row = [args.label,
           f"{prefill_ms:.2f}" if prefill_ms is not None else "NA",
           f"{tpot_ms:.3f}" if tpot_ms is not None else "NA",
           str(eval_tokens),
           f"{avg_w:.2f}",
           f"{e_per_tok_mj:.1f}", f"{win_s:.1f}", f"{duration:.2f}", str(len(samples))]
    header = ["label", "prefill_ms", "tpot_ms", "eval_tokens",
              "avg_power_w", "energy_per_tok_mj", "gen_window_s", "duration_s", "n_samples"]

    if args.tsv:
        new = not os.path.exists(args.tsv)
        with open(args.tsv, "a") as f:
            if new:
                f.write("\t".join(header) + "\n")
            f.write("\t".join(row) + "\n")

    print("  " + " | ".join(f"{h}={v}" for h, v in zip(header, row)))


if __name__ == "__main__":
    main()
