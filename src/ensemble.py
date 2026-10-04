"""Baseline late fusion: rata-rata probabilitas (soft voting) dari dua run per fold.

Butuh run yang menyimpan "y_prob" (train.py versi terbaru).

    python src/ensemble.py --task depresi --runs indobertweet qwen3_lora
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import compute_metrics, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--name", default=None, help="Nama run hasil (default: ensemble_<run1>_<run2>)")
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    base = Path(args.results_dir, args.task)
    name = args.name or "ensemble_" + "_".join(args.runs)
    with open(Path("data", args.task, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    folds = sorted(set.intersection(*[{p.stem for p in (base / r).glob("fold*.json")} for r in args.runs]))
    for fold in folds:
        preds = []
        for r in args.runs:
            with open(base / r / f"{fold}.json", encoding="utf-8") as f:
                pr = json.load(f)["predictions"]
            if "y_prob" not in pr:
                raise SystemExit(f"{r}/{fold} tidak punya y_prob; jalankan ulang dengan train.py terbaru")
            preds.append(pr)
        ids = preds[0]["id"]
        for pr in preds[1:]:
            assert pr["id"] == ids, "urutan sampel test berbeda antar-run"
        prob = np.mean([np.array(pr["y_prob"]) for pr in preds], axis=0)
        y_pred = prob.argmax(-1).tolist()
        metrics = compute_metrics(preds[0]["y_true"], y_pred, labels)
        k = int(fold.replace("fold", ""))
        save_json({
            "run": name, "fold": k, "members": args.runs, "test_metrics": metrics,
            "predictions": {"id": ids, "y_true": preds[0]["y_true"], "y_pred": y_pred,
                            "y_prob": prob.round(4).tolist()},
        }, base / name / f"{fold}.json")
        print(f"{name} {fold}: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
