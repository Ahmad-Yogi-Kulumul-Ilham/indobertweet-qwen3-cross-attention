"""Terjemahkan dataset kesehatan mental berbahasa Inggris ke bahasa Indonesia dengan NLLB-200.

Teks dipecah per kalimat sebelum diterjemahkan (NLLB menurun kualitasnya untuk input panjang),
lalu digabung kembali. Hasil ditulis bertahap sehingga bisa di-resume jika sesi Kaggle terputus.

Contoh (dataset Kaggle "Sentiment Analysis for Mental Health"):
    python src/translate_dataset.py --input "/kaggle/input/.../Combined Data.csv" \
        --text_col statement --label_col status --max_per_class 3000
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_segments(text, max_chars):
    """Pecah teks menjadi segmen <= max_chars, sebisa mungkin di batas kalimat."""
    pieces = []
    for sent in SENT_RE.split(text):
        words = sent.split()
        while words:  # kalimat sangat panjang tanpa tanda baca dipotong per kata
            chunk = []
            while words and len(" ".join(chunk + words[:1])) <= max_chars:
                chunk.append(words.pop(0))
            if not chunk:
                chunk.append(words.pop(0))
            pieces.append(" ".join(chunk))
    segments, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            segments.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        segments.append(cur)
    return segments


@torch.no_grad()
def translate(segments, tok, model, args, tgt_id):
    order = sorted(range(len(segments)), key=lambda i: len(segments[i]))
    out = [None] * len(segments)
    for i in range(0, len(order), args.batch_size):
        idx = order[i:i + args.batch_size]
        enc = tok([segments[j] for j in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len).to(model.device)
        gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                             max_new_tokens=args.max_len, num_beams=args.num_beams)
        for j, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[j] = t
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/translated_id.csv")
    p.add_argument("--text_col", default="statement")
    p.add_argument("--label_col", default="status")
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Subsampling per kelas agar waktu terjemahan masuk akal")
    p.add_argument("--max_chars", type=int, default=2000,
                   help="Potong teks sumber; model hanya membaca ~128-256 token pertama")
    p.add_argument("--segment_chars", type=int, default=400)
    p.add_argument("--rows_per_block", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.input).dropna(subset=[args.text_col, args.label_col])
    df = df.drop_duplicates(args.text_col)
    if args.max_per_class:
        df = df.sample(frac=1, random_state=args.seed).groupby(args.label_col).head(args.max_per_class)
    df = df.reset_index(drop=True)
    df["src_id"] = df.index

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out_path)["src_id"]) if out_path.exists() else set()
    todo = df[~df["src_id"].isin(done)]
    print(f"Total {len(df)} baris, sudah {len(done)}, sisa {len(todo)}")
    print(df[args.label_col].value_counts().to_string())

    tok = AutoTokenizer.from_pretrained(args.model, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    tgt_id = tok.convert_tokens_to_ids("ind_Latn")

    for start in tqdm(range(0, len(todo), args.rows_per_block), desc="blok"):
        block = todo.iloc[start:start + args.rows_per_block]
        segments, owner = [], []
        for row_i, text in enumerate(block[args.text_col]):
            for seg in split_segments(str(text)[:args.max_chars], args.segment_chars):
                segments.append(seg)
                owner.append(row_i)
        translated = translate(segments, tok, model, args, tgt_id)
        joined = [[] for _ in range(len(block))]
        for row_i, t in zip(owner, translated):
            joined[row_i].append(t)
        pd.DataFrame({
            "src_id": block["src_id"].values,
            "text": [" ".join(parts) for parts in joined],
            "label": block[args.label_col].values,
            "text_en": block[args.text_col].values,
        }).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

    print(f"Selesai -> {out_path}")


if __name__ == "__main__":
    main()
