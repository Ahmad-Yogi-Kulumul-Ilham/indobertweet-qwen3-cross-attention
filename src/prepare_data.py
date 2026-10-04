"""Bersihkan data, buang duplikat/label konflik, dan buat stratified k-fold yang tetap.

Semua eksperimen (baseline, model tunggal, fusion) memakai data/processed.csv yang sama
sehingga hasil antar-model bisa dibandingkan dan diuji signifikansinya.

    python src/prepare_data.py --input data/translated_id.csv
    python src/prepare_data.py --input data/translated_id.csv data/tweet_native.csv \
        --label_map '{"Personality disorder": null}'
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import clean_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, nargs="+", help="Satu atau lebih CSV")
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="label")
    p.add_argument("--label_map", default=None,
                   help='JSON untuk mengganti/membuang label, mis. {"Personality disorder": null}')
    p.add_argument("--min_words", type=int, default=3)
    p.add_argument("--min_per_class", type=int, default=50)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="data")
    args = p.parse_args()

    df = pd.concat([pd.read_csv(f) for f in args.input], ignore_index=True)
    df = df[[args.text_col, args.label_col]].rename(columns={args.text_col: "text", args.label_col: "label"})
    df = df.dropna()
    df["label"] = df["label"].astype(str).str.strip()
    if args.label_map:
        mapping = json.loads(args.label_map)
        df["label"] = df["label"].map(lambda l: mapping.get(l, l))
        df = df.dropna(subset=["label"])
    n0 = len(df)

    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.split().str.len() >= args.min_words]
    n_labels_per_text = df.groupby("text")["label"].nunique()
    conflict = n_labels_per_text[n_labels_per_text > 1].index
    df = df[~df["text"].isin(conflict)].drop_duplicates("text")
    counts = df["label"].value_counts()
    df = df[df["label"].isin(counts[counts >= args.min_per_class].index)].reset_index(drop=True)

    labels = sorted(df["label"].unique())
    df["label_id"] = df["label"].map({l: i for i, l in enumerate(labels)})
    df.insert(0, "id", range(len(df)))
    df["fold"] = -1
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for k, (_, test_idx) in enumerate(skf.split(df, df["label_id"])):
        df.loc[test_idx, "fold"] = k

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df[["id", "text", "label", "label_id", "fold"]].to_csv(out / "processed.csv", index=False)
    with open(out / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "n_splits": args.n_splits}, f, indent=2, ensure_ascii=False)

    print(f"{n0} -> {len(df)} baris setelah cleaning, dedup ({len(conflict)} teks label-konflik dibuang)")
    print(df["label"].value_counts().to_string())
    words = df["text"].str.split().str.len()
    print(f"Panjang kata: median {words.median():.0f}, p90 {words.quantile(0.9):.0f}")


if __name__ == "__main__":
    main()
