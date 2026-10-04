import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(\w)\1{2,}")
SPACE_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Normalisasi ala IndoBERTweet: lowercase, URL -> HTTPURL, mention -> @USER."""
    text = str(text).lower()
    text = URL_RE.sub("HTTPURL", text)
    text = MENTION_RE.sub("@USER", text)
    text = REPEAT_RE.sub(r"\1\1", text)  # "bangeeettt" -> "bangeett"
    return SPACE_RE.sub(" ", text).strip()


def load_data(data_path, labels_path):
    df = pd.read_csv(data_path)
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    return df, labels


def compute_metrics(y_true, y_pred, labels):
    ids = list(range(len(labels)))
    kw = dict(labels=ids, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", **kw)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", **kw)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", **kw)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", **kw)),
        "per_class": classification_report(
            y_true, y_pred, target_names=labels, output_dict=True, **kw
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


def resolve_task_paths(args):
    """--task X  ->  data/X/processed.csv, data/X/labels.json, results/X/."""
    data_dir = f"data/{args.task}" if args.task else "data"
    args.data = args.data or f"{data_dir}/processed.csv"
    args.labels = args.labels or f"{data_dir}/labels.json"
    args.out_dir = args.out_dir or (f"results/{args.task}" if args.task else "results")
    return args


def parse_folds(spec, n_folds):
    if spec == "all":
        return list(range(n_folds))
    return [int(x) for x in spec.split(",")]
