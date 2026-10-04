"""Analisis untuk bagian pembahasan -> paper/analisis.md

  1. Komplementaritas IndoBERTweet vs Qwen3 (kesepakatan, batas atas oracle)
  2. Siapa yang diperbaiki fusion: sampel yang benar oleh fusion tetapi salah oleh kedua model tunggal
  3. Kata/frasa paling berpengaruh per kelas (koefisien TF-IDF + LogReg, seluruh data)
  4. F1 per kelas Fusion v2

    python src/analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TASKS = ["depresi", "distortion"]
OUT = Path("paper/analisis.md")


def preds(task, run):
    out = {}
    for f in Path("results", task, run).glob("fold*.json"):
        p = json.load(open(f, encoding="utf-8"))["predictions"]
        out.update({i: (t, y) for i, t, y in zip(p["id"], p["y_true"], p["y_pred"])})
    return out


def fmt(x):
    return f"{x * 100:.2f}".replace(".", ",")


def main():
    lines = ["# Analisis tambahan", ""]

    lines += ["## 1. Komplementaritas model tunggal", "",
              "| Task | Akurasi IndoBERTweet | Akurasi Qwen3 | Prediksi sama | Oracle (salah satu benar) | Akurasi Fusion v2 |",
              "|---|---|---|---|---|---|"]
    for t in TASKS:
        a, b, f = preds(t, "indobertweet"), preds(t, "qwen3_lora"), preds(t, "fusion_cross_v2")
        ids = sorted(set(a) & set(b) & set(f))
        acc = lambda d: np.mean([d[i][0] == d[i][1] for i in ids])
        agree = np.mean([a[i][1] == b[i][1] for i in ids])
        oracle = np.mean([a[i][0] == a[i][1] or b[i][0] == b[i][1] for i in ids])
        lines.append(f"| {t} | {fmt(acc(a))} | {fmt(acc(b))} | {fmt(agree)} | {fmt(oracle)} | {fmt(acc(f))} |")
    lines.append("")

    lines += ["## 2. Kasus yang diperbaiki / dirusak fusion", "",
              "Hitungan pada seluruh sampel test (gabungan 5 fold).", "",
              "| Task | Kedua model tunggal salah | ...diperbaiki fusion | Model tunggal berbeda pendapat | ...fusion benar | Kedua model tunggal benar | ...dirusak fusion |",
              "|---|---|---|---|---|---|---|"]
    examples = {}
    for t in TASKS:
        a, b, f = preds(t, "indobertweet"), preds(t, "qwen3_lora"), preds(t, "fusion_cross_v2")
        ids = sorted(set(a) & set(b) & set(f))
        ok = lambda d, i: d[i][0] == d[i][1]
        both_wrong = [i for i in ids if not ok(a, i) and not ok(b, i)]
        split = [i for i in ids if ok(a, i) != ok(b, i)]
        both_right = [i for i in ids if ok(a, i) and ok(b, i)]
        fixed = [i for i in both_wrong if ok(f, i)]
        split_ok = [i for i in split if ok(f, i)]
        broken = [i for i in both_right if not ok(f, i)]
        lines.append(f"| {t} | {len(both_wrong)} | {len(fixed)} ({len(fixed) / max(len(both_wrong), 1):.0%}) | "
                     f"{len(split)} | {len(split_ok)} ({len(split_ok) / max(len(split), 1):.0%}) | "
                     f"{len(both_right)} | {len(broken)} ({len(broken) / max(len(both_right), 1):.1%}) |")
        examples[t] = (fixed, split_ok)
    lines += ["", "Interpretasi: pada sampel yang diperdebatkan kedua model, persentase fusion benar menunjukkan "
              "apakah fusion lebih dari sekadar memilih acak salah satu model (50%).", ""]

    lines += ["## 3. Kata/frasa paling berpengaruh (TF-IDF + LogReg, seluruh data)", ""]
    for t in TASKS:
        df = pd.read_csv(Path("data", t, "processed.csv"))
        labels = json.load(open(Path("data", t, "labels.json"), encoding="utf-8"))["labels"]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        X = vec.fit_transform(df["text"])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, df["label_id"])
        vocab = np.array(vec.get_feature_names_out())
        coef = clf.coef_[0]  # biner: koefisien positif -> kelas labels[1]
        lines += [f"**{t}**", "", f"| Paling khas `{labels[1]}` | Paling khas `{labels[0]}` |", "|---|---|"]
        top_pos, top_neg = vocab[np.argsort(-coef)[:15]], vocab[np.argsort(coef)[:15]]
        lines += [f"| {p} | {n} |" for p, n in zip(top_pos, top_neg)]
        lines.append("")

    lines += ["## 4. F1 per kelas Fusion v2 (mean ± std, 5 fold)", ""]
    for t in TASKS:
        folds = [json.load(open(f, encoding="utf-8")) for f in Path("results", t, "fusion_cross_v2").glob("fold*.json")]
        labels = json.load(open(Path("data", t, "labels.json"), encoding="utf-8"))["labels"]
        lines += [f"**{t}**", "", "| Kelas | Precision | Recall | F1 |", "|---|---|---|---|"]
        for lab in labels:
            row = []
            for m in ("precision", "recall", "f1-score"):
                v = np.array([f["test_metrics"]["per_class"][lab][m] for f in folds]) * 100
                row.append(f"{v.mean():.2f} ± {v.std(ddof=1):.2f}".replace(".", ","))
            lines.append(f"| {lab} | " + " | ".join(row) + " |")
        lines.append("")

    lines += ["## 5. Contoh kasus (untuk analisis kualitatif; JANGAN dikutip utuh di naskah, parafrasekan)", ""]
    for t in TASKS:
        df = pd.read_csv(Path("data", t, "processed.csv")).set_index("id")
        fixed, split_ok = examples[t]
        rng = np.random.default_rng(0)
        lines += [f"**{t}**: diperbaiki fusion padahal kedua model tunggal salah", ""]
        for i in rng.choice(fixed, size=min(6, len(fixed)), replace=False):
            lines.append(f"- [{df.loc[i, 'label']}] {df.loc[i, 'text'][:220]}")
        lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
