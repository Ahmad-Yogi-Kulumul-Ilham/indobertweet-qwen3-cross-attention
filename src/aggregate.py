"""Rekap hasil semua run (mean ± std antar-fold) dan uji signifikansi.

    python src/aggregate.py                                # tabel ringkasan
    python src/aggregate.py --per_class fusion_cross       # F1 per kelas
    python src/aggregate.py --compare fusion_cross indobertweet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, ttest_rel

METRICS = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]


def load_runs(results_dir):
    runs = {}
    for d in sorted(Path(results_dir).iterdir()):
        files = sorted(d.glob("fold*.json")) if d.is_dir() else []
        if files:
            runs[d.name] = {}
            for f in files:
                with open(f, encoding="utf-8") as fh:
                    r = json.load(fh)
                runs[d.name][r["fold"]] = r
    return runs


def fmt(values):
    v = np.array(values) * 100
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def summarize(runs, results_dir):
    rows = []
    for name, folds in runs.items():
        row = {"run": name, "folds": len(folds)}
        for m in METRICS:
            vals = [f["test_metrics"][m] for f in folds.values()]
            row[m] = fmt(vals)
            row[f"_{m}_mean"] = float(np.mean(vals))
        gates = [f["gate_mean"] for f in folds.values() if f.get("gate_mean")]
        row["gate(enc,llm)"] = ", ".join(f"{g:.2f}" for g in np.mean(gates, axis=0)) if gates else "-"
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("_macro_f1_mean", ascending=False)
    cols = ["run", "folds"] + METRICS + ["gate(enc,llm)"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    table = "\n".join(lines)
    print(table)
    Path(results_dir, "summary.md").write_text(table + "\n", encoding="utf-8")
    df[cols].to_csv(Path(results_dir, "summary.csv"), index=False)


def per_class(runs, name):
    folds = runs[name]
    any_fold = next(iter(folds.values()))
    labels = [k for k in any_fold["test_metrics"]["per_class"]
              if k not in ("accuracy", "macro avg", "weighted avg")]
    print(f"| kelas | precision | recall | F1 |  ({name})\n|---|---|---|---|")
    for lab in labels:
        stats = {m: fmt([f["test_metrics"]["per_class"][lab][m] for f in folds.values()])
                 for m in ("precision", "recall", "f1-score")}
        print(f"| {lab} | {stats['precision']} | {stats['recall']} | {stats['f1-score']} |")


def pooled_correct(folds):
    out = {}
    for f in folds.values():
        pr = f["predictions"]
        for i, t, p in zip(pr["id"], pr["y_true"], pr["y_pred"]):
            out[i] = t == p
    return out


def compare(runs, a, b):
    ca, cb = pooled_correct(runs[a]), pooled_correct(runs[b])
    common = sorted(set(ca) & set(cb))
    only_a = sum(ca[i] and not cb[i] for i in common)
    only_b = sum(cb[i] and not ca[i] for i in common)
    n = only_a + only_b
    if n == 0:
        p_mc = 1.0
    elif n < 25:
        p_mc = binomtest(min(only_a, only_b), n, 0.5).pvalue
    else:
        p_mc = chi2.sf((abs(only_a - only_b) - 1) ** 2 / n, df=1)
    print(f"McNemar {a} vs {b} ({len(common)} sampel): "
          f"hanya {a} benar={only_a}, hanya {b} benar={only_b}, p={p_mc:.4g}")

    folds = sorted(set(runs[a]) & set(runs[b]))
    if len(folds) >= 2:
        fa = [runs[a][k]["test_metrics"]["macro_f1"] for k in folds]
        fb = [runs[b][k]["test_metrics"]["macro_f1"] for k in folds]
        t = ttest_rel(fa, fb)
        print(f"Paired t-test macro-F1 ({len(folds)} fold): t={t.statistic:.3f}, p={t.pvalue:.4g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=None, help="Baca results/<task>/")
    p.add_argument("--results_dir", default=None)
    p.add_argument("--per_class", default=None)
    p.add_argument("--compare", nargs=2, default=None, metavar=("RUN_A", "RUN_B"))
    args = p.parse_args()
    args.results_dir = args.results_dir or (f"results/{args.task}" if args.task else "results")

    runs = load_runs(args.results_dir)
    if args.compare:
        compare(runs, *args.compare)
    elif args.per_class:
        per_class(runs, args.per_class)
    else:
        summarize(runs, args.results_dir)


if __name__ == "__main__":
    main()
