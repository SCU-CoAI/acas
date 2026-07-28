#!/usr/bin/env python3
# Per-token stats for the ACAS configs (Tables 3-4): joins each rep's ptok CSV (real per-token
# latency from llama.cpp's own accumulator) with its tok CSV (is_update marker) by token index,
# pools reps, and splits normal (sparse) vs dense-update tokens.
#
# Output is RAW milliseconds from this measurement session — no rescaling. Run the per-token
# scripts in the same session as sweep_table.sh and the means line up with your own TPOT column
# directly. (The paper's printed Tables 3-4 express these stats in its Tables 1-2 TPOT frame;
# the frame-free quantities to compare are the Dense/Sparse ratio and the tail shape.)
import csv, glob, os, re, statistics as st
RB = os.environ.get("ACAS_RESULTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CFG = [  # (key, results dir, label)
  ("si",        f"{RB}/token_timing",      "ACAS-SI (contrib.)"),
  ("si_prec",   f"{RB}/token_timing_prec", "ACAS-SI (prec.)"),
  ("grasp",     f"{RB}/token_timing",      "ACAS-GR (contrib.)"),
  ("grasp_prec",f"{RB}/token_timing_prec", "ACAS-GR (prec.)"),
  ("cats",      f"{RB}/token_timing",      "ACAS-CATS (contrib.)"),
  ("cats_l2",   f"{RB}/token_timing_prec", "ACAS-CATS (L2)"),
  ("catsgp",    f"{RB}/token_timing",      "ACAS-CATS-GP (contrib.)"),
  ("catsgp_l2", f"{RB}/token_timing_prec", "ACAS-CATS-GP (L2)"),
]
def _read(p, col):
    out = {}
    for r in csv.DictReader(open(p)):
        try: out[int(r["idx"])] = float(r[col])
        except Exception: pass
    return out
def load(d, key):
    dt, upd = [], []
    for pf in sorted(glob.glob(f"{d}/ptok_{key}_r*.csv")):
        rep = re.search(r"_r(\d+)\.csv$", pf).group(1)
        tf = f"{d}/tok_{key}_r{rep}.csv"
        dd = _read(pf, "dt_us"); uu = _read(tf, "is_update") if os.path.exists(tf) else {}
        for i in sorted(dd):
            dt.append(dd[i] / 1000.0); upd.append(int(uu.get(i, 0)))
    return dt, upd
def pct(x, p):
    xs = sorted(x); return xs[min(len(xs) - 1, int(len(xs) * p / 100))]
print("%-26s %5s %6s | %6s %6s %6s | %7s %7s %6s" %
      ("config", "#upd", "mean", "p50", "p95", "p99", "Sparse", "Dense", "D/S"))
for key, d, lab in CFG:
    dt, upd = load(d, key)
    if not dt: print(f"{lab:26} (no data {key})"); continue
    norm = [v for v, f in zip(dt, upd) if f == 0]; dns = [v for v, f in zip(dt, upd) if f == 1]
    sp = st.mean(norm); de = st.mean(dns) if dns else float("nan")
    print("%-26s %5d %6.2f | %6.2f %6.2f %6.2f | %7.2f %7.2f %5.2fx" %
          (lab, len(dns), st.mean(dt), pct(dt,50), pct(dt,95), pct(dt,99), sp, de, de/sp))
