"""Konversi dataset mentah (data/raw/) ke format standar text,label di data/raw_std/<task>.csv.

Task:
  depresi          Budiman (2021), tweet berlabel psikolog, biner
  suicide          apricitea, tweet ide bunuh diri, 3 anotator (majority vote dihitung ulang)
  depresi_anxiety  Kaggle stevenhans, tweet depresi/kecemasan, 3 anotator, biner (datd_rand tidak dipakai)
  distortion       Cognitive Distortion ID (Suputra dkk., 2025), 2 psikolog, biner terdistorsi/tidak
  distortion_type  sama, hanya teks terdistorsi, 11 jenis distorsi
  multiclass       CortiSoul, terjemahan Reddit/Kaggle ke Indonesia, 7 kelas

    python src/build_datasets.py
    python src/build_datasets.py --tasks depresi suicide
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def build_depresi(raw):
    df = pd.read_csv(raw / "budiman_depresi.csv")
    label = df["sentiment"].str.strip().map({"Terindikasi Depresi": "depresi", "Tidak Terindikasi": "tidak_depresi"})
    return pd.DataFrame({"text": df["tweet"], "label": label})


def build_suicide(raw):
    # kolom "majority" di file berupa rumus Excel tanpa nilai tersimpan, jadi dihitung ulang
    df = pd.read_excel(raw / "apricitea_suicide.xlsx", sheet_name="current_data",
                       usecols=["tweet", "ACN", "BME", "AIP"]).dropna()
    votes = df[["ACN", "BME", "AIP"]].astype(int).sum(axis=1)
    label = (votes >= 2).map({True: "ide_bunuh_diri", False: "tidak"})
    return pd.DataFrame({"text": df["tweet"], "label": label})


def build_depresi_anxiety(raw):
    # datd_rand tidak dipakai: positifnya identik dengan datd_train (bocor) dan negatifnya tweet acak
    df = pd.concat([pd.read_csv(raw / "datd_train.csv"), pd.read_csv(raw / "datd_test.csv")])
    return pd.DataFrame({"text": df["text"], "label": df["label"].map({0: "normal", 1: "depresi_anxiety"})})


def load_distortion(raw):
    df = pd.read_csv(raw / "cognitive_distortion.csv")
    # hanya data asli: baris DIS-* adalah augmentasi back-translation (parafrase lintas fold = bocor)
    df = df[df["DATA STATUS"] == "RAW-ORI"]
    # tanda $...$ menandai span distorsi dan hampir hanya ada di teks terdistorsi -> bocoran label
    text = df["TEXT"].map(lambda t: re.sub(r"\s+", " ", str(t).replace("$", " ")).strip())
    return text, df["FIRST ANNOTATOR"].str.strip()


def build_distortion(raw):
    text, label = load_distortion(raw)
    return pd.DataFrame({"text": text, "label": label.map(lambda l: "tidak" if l == "No Distortion" else "distorsi")})


def build_distortion_type(raw):
    text, label = load_distortion(raw)
    keep = label != "No Distortion"
    return pd.DataFrame({"text": text[keep], "label": label[keep]})


def build_multiclass(raw):
    df = pd.read_csv(raw / "cortisoul_multiclass.csv", usecols=["text", "status"])
    return pd.DataFrame({"text": df["text"], "label": df["status"].str.strip().str.lower()})


BUILDERS = {
    "depresi": build_depresi,
    "suicide": build_suicide,
    "depresi_anxiety": build_depresi_anxiety,
    "distortion": build_distortion,
    "distortion_type": build_distortion_type,
    "multiclass": build_multiclass,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--out_dir", default="data/raw_std")
    p.add_argument("--tasks", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    args = p.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        df = BUILDERS[task](raw).dropna()
        df.to_csv(out / f"{task}.csv", index=False)
        print(f"== {task}: {len(df)} baris -> {out / f'{task}.csv'}")
        print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
