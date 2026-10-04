"""Baseline klasik TF-IDF (word + char n-gram) dengan split fold yang sama.

    python src/baselines.py --models svm logreg nb
"""
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from common import compute_metrics, load_data, parse_folds, resolve_task_paths, save_json


def build(name, seed):
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=100_000, sublinear_tf=True)),
    ])
    clf = {
        "svm": LinearSVC(class_weight="balanced", random_state=seed),
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "nb": ComplementNB(),
    }[name]
    return make_pipeline(features, clf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=None, help="depresi / suicide / multiclass / distortion")
    p.add_argument("--data", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--models", nargs="+", default=["svm", "logreg", "nb"])
    p.add_argument("--folds", default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    args = resolve_task_paths(p.parse_args())

    df, labels = load_data(args.data, args.labels)
    for name in args.models:
        run = f"tfidf_{name}"
        for fold in parse_folds(args.folds, df["fold"].nunique()):
            train, test = df[df["fold"] != fold], df[df["fold"] == fold]
            model = build(name, args.seed).fit(train["text"], train["label_id"])
            pred = model.predict(test["text"])
            metrics = compute_metrics(test["label_id"], pred, labels)
            save_json({
                "run": run, "fold": fold, "test_metrics": metrics,
                "predictions": {"id": test["id"].tolist(), "y_true": test["label_id"].tolist(),
                                "y_pred": pred.tolist()},
            }, f"{args.out_dir}/{run}/fold{fold}.json")
            print(f"{run} fold {fold}: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
