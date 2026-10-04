"""Training + evaluasi k-fold untuk model encoder, LLM (QLoRA), dan fusion.

Setiap fold: test = fold k, sisanya dipecah train/val (stratified) untuk early stopping.
Di Kaggle T4 x2, branch LLM otomatis ditaruh di GPU kedua.

    # uji cepat
    python src/train.py --model fusion --run_name debug --folds 0 --epochs 1 --max_train_samples 500
    # model usulan
    python src/train.py --model fusion --fusion cross_attention --run_name fusion_cross --folds 0,1
"""
import argparse
import gc
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from common import compute_metrics, load_data, parse_folds, resolve_task_paths, save_json
from models import build_model

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Collator:
    def __init__(self, tokenizers, max_len):
        self.tokenizers = tokenizers
        self.max_len = max_len

    def __call__(self, rows):
        texts = [r["text"] for r in rows]
        batch = {"labels": torch.tensor([r["label_id"] for r in rows]),
                 "ids": torch.tensor([r["id"] for r in rows])}
        for name, tok in self.tokenizers.items():
            enc = tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            batch[name] = (enc["input_ids"], enc["attention_mask"])
        return batch


def load_tokenizers(args):
    toks = {}
    if args.model in ("encoder", "fusion"):
        toks["encoder"] = AutoTokenizer.from_pretrained(args.encoder_name)
    if args.model in ("llm", "fusion"):
        tok = AutoTokenizer.from_pretrained(args.llm_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        toks["llm"] = tok
    return toks


@torch.no_grad()
def predict(model, loader, amp):
    model.eval()
    ids, y_true, probs, gates = [], [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits, gate, _ = model(batch)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu())
        y_true.append(batch["labels"])
        ids.append(batch["ids"])
        if gate is not None:
            gates.append(gate.float().cpu())
    probs = torch.cat(probs)
    gate_mean = torch.cat(gates).mean(0).tolist() if gates else None
    return (torch.cat(ids).tolist(), torch.cat(y_true).tolist(), probs.argmax(-1).tolist(),
            gate_mean, probs.numpy().round(4).tolist())


def run_fold(args, df, labels, fold, toks, head_device, llm_device):
    set_seed(args.seed + fold)
    test_df = df[df["fold"] == fold]
    trainval = df[df["fold"] != fold]
    train_df, val_df = train_test_split(trainval, test_size=args.val_size,
                                        stratify=trainval["label_id"], random_state=args.seed)
    if args.max_train_samples:
        train_df = train_df.sample(min(args.max_train_samples, len(train_df)), random_state=args.seed)
        val_df = val_df.sample(min(args.max_train_samples // 4 + 1, len(val_df)), random_state=args.seed)
        test_df = test_df.sample(min(args.max_train_samples // 4 + 1, len(test_df)), random_state=args.seed)

    collate = Collator(toks, args.max_len)

    def loader(d, shuffle, bs):
        return DataLoader(d.to_dict("records"), batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=0 if os.name == "nt" else 2)

    train_loader = loader(train_df, True, args.batch_size)
    val_loader = loader(val_df, False, args.batch_size * 2)
    test_loader = loader(test_df, False, args.batch_size * 2)

    model = build_model(args, len(labels), head_device, llm_device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[fold {fold}] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"trainable params={n_trainable / 1e6:.1f}M")

    weight = None
    if args.class_weight:
        counts = np.bincount(train_df["label_id"], minlength=len(labels))
        weight = torch.tensor(len(train_df) / (len(labels) * np.maximum(counts, 1)),
                              dtype=torch.float32, device=head_device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    lrs = {"encoder": args.lr_encoder, "llm": args.lr_llm, "head": args.lr_head}
    optimizer = torch.optim.AdamW(model.param_groups(lrs), weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_f1, best_state, bad_epochs, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running, skipped = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"fold {fold} epoch {epoch + 1}")
        for step, batch in enumerate(bar):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits, _, aux_logits = model(batch)
            y = batch["labels"].to(head_device)  # jangan dinamai `labels`: itu daftar nama kelas
            loss = loss_fn(logits.float(), y)
            if aux_logits and args.aux_weight > 0:
                loss = loss + args.aux_weight * sum(loss_fn(a.float(), y) for a in aux_logits) / len(aux_logits)
            loss = loss / args.grad_accum
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * args.grad_accum
            bar.set_postfix(loss=f"{running / (step + 1 - skipped or 1):.4f}")

        _, vt, vp, _, _ = predict(model, val_loader, amp)
        val = compute_metrics(vt, vp, labels)
        history.append({"epoch": epoch + 1, "train_loss": running / max(len(train_loader) - skipped, 1),
                        "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"],
                        "skipped_nonfinite": skipped})
        print(f"[fold {fold}] epoch {epoch + 1}: val macro_f1={val['macro_f1']:.4f} acc={val['accuracy']:.4f}"
              + (f" (lewati {skipped} batch NaN)" if skipped else ""))
        if val["macro_f1"] > best_f1:
            best_f1, best_state, bad_epochs = val["macro_f1"], model.trainable_state(), 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[fold {fold}] early stopping")
                break

    model.load_trainable_state(best_state)
    ids, yt, yp, gate_mean, probs = predict(model, test_loader, amp)
    metrics = compute_metrics(yt, yp, labels)
    print(f"[fold {fold}] TEST acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
          + (f" gate(encoder, llm)={[round(g, 3) for g in gate_mean]}" if gate_mean else ""))

    out_dir = f"{args.out_dir}/{args.run_name}"
    save_json({
        "run": args.run_name, "fold": fold, "args": vars(args),
        "trainable_params": n_trainable, "train_minutes": (time.time() - t0) / 60,
        "history": history, "best_val_macro_f1": best_f1,
        "test_metrics": metrics, "gate_mean": gate_mean,
        "predictions": {"id": ids, "y_true": yt, "y_pred": yp, "y_prob": probs},
    }, f"{out_dir}/fold{fold}.json")
    if args.save_model:
        torch.save(best_state, f"{out_dir}/fold{fold}_trainable.pt")

    del model, optimizer, best_state
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=None, help="depresi / suicide / multiclass / distortion")
    p.add_argument("--data", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--model", choices=["encoder", "llm", "fusion"], default="fusion")
    p.add_argument("--fusion", choices=["concat", "gated", "cross_attention"], default="cross_attention")
    p.add_argument("--encoder_name", default="indolem/indobertweet-base-uncased")
    p.add_argument("--llm_name", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--llm_compute_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--run_name", required=True)
    p.add_argument("--folds", default="all", help='"all" atau mis. "0,1"')
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr_encoder", type=float, default=2e-5)
    p.add_argument("--lr_llm", type=float, default=2e-4)
    p.add_argument("--lr_head", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--aux_weight", type=float, default=0.3,
                   help="Bobot loss auxiliary classifier per branch (mode fusion)")
    p.add_argument("--branch_dropout", type=float, default=0.15,
                   help="Peluang satu branch di-nol-kan per sampel saat training (mode fusion)")
    p.add_argument("--class_weight", action="store_true", help="Weighted CE untuk kelas tidak seimbang")
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--max_train_samples", type=int, default=None, help="Untuk debugging")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--single_gpu", action="store_true")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    args = resolve_task_paths(p.parse_args())

    assert torch.cuda.is_available(), "Butuh GPU (aktifkan Accelerator GPU T4 x2 di Kaggle)"
    head_device = torch.device("cuda:0")
    llm_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 and not args.single_gpu else head_device
    print(f"GPU: {torch.cuda.device_count()} | encoder/head -> {head_device} | llm -> {llm_device}")

    df, labels = load_data(args.data, args.labels)
    toks = load_tokenizers(args)
    for fold in parse_folds(args.folds, df["fold"].nunique()):
        run_fold(args, df, labels, fold, toks, head_device, llm_device)


if __name__ == "__main__":
    main()
