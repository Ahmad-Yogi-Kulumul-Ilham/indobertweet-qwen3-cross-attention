"""Model hybrid: encoder IndoBERTweet + LLM (Qwen3, 4-bit QLoRA) dengan fusion head.

Mode fusion (untuk ablation):
  single           satu branch saja (baseline encoder / LLM)
  concat           gabung vektor pooled kedua branch
  gated            bobot softmax per branch (bobotnya bisa dianalisis)
  cross_attention  cross-attention dua arah antar-token, pooling, lalu concat (model usulan)

Mode >= 2 branch memakai auxiliary classifier per branch + branch dropout (lihat FusionHead).
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, BitsAndBytesConfig

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def masked_mean(seq, mask):
    mask = mask.unsqueeze(-1).to(seq.dtype)
    return (seq * mask).sum(1) / mask.sum(1).clamp(min=1)


class EncoderBranch(nn.Module):
    def __init__(self, name, device):
        super().__init__()
        self.model = AutoModel.from_pretrained(name).to(device)
        self.hidden_size = self.model.config.hidden_size
        self.device = device

    def forward(self, input_ids, attention_mask):
        seq = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return seq, seq[:, 0]


class LLMBranch(nn.Module):
    def __init__(self, name, device, lora_r, lora_alpha, lora_dropout,
                 compute_dtype=torch.float16, grad_ckpt=True):
        super().__init__()
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
        base = AutoModel.from_pretrained(name, quantization_config=bnb, torch_dtype=compute_dtype,
                                         device_map={"": device.index or 0})
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None)
        lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules=LORA_TARGETS, bias="none", task_type="FEATURE_EXTRACTION")
        self.model = get_peft_model(base, lora)
        # GradScaler membutuhkan parameter trainable dalam fp32
        for p in self.model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
        self.hidden_size = base.config.hidden_size
        self.device = device

    def forward(self, input_ids, attention_mask):
        seq = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        return seq, masked_mean(seq, attention_mask)


class GatedFusion(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.score = nn.Linear(dim * n, n)

    def forward(self, xs):
        w = torch.softmax(self.score(torch.cat(xs, dim=-1)), dim=-1)
        return sum(w[:, i:i + 1] * x for i, x in enumerate(xs)), w


class CrossAttentionFusion(nn.Module):
    """Cross-attention dua arah; hasil kedua arah di-pool lalu digabung (concat).

    Versi awal memakai gate softmax di sini, tetapi gate-nya kolaps ke salah satu branch
    (bobot 0/1 berganti-ganti antar-fold), sehingga fusion praktis hanya memilih satu model.
    """

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.a2b = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.b2a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, seq_a, mask_a, seq_b, mask_b):
        att_a, _ = self.a2b(seq_a, seq_b, seq_b, key_padding_mask=~mask_b.bool())
        att_b, _ = self.b2a(seq_b, seq_a, seq_a, key_padding_mask=~mask_a.bool())
        a = self.norm_a(seq_a + att_a)
        b = self.norm_b(seq_b + att_b)
        return torch.cat([masked_mean(a, mask_a), masked_mean(b, mask_b)], dim=-1)


class FusionHead(nn.Module):
    """Proyeksi per branch + fusion + klasifier.

    Untuk >= 2 branch ditambah dua mekanisme agar kedua branch benar-benar dipakai:
      - auxiliary classifier per branch (loss tambahan di train.py), dan
      - branch dropout: saat training, representasi satu branch di-nol-kan secara acak per sampel.
    """

    def __init__(self, dims, num_labels, fusion, proj_dim, heads, dropout, branch_dropout=0.0):
        super().__init__()
        self.fusion = fusion
        self.branch_dropout = branch_dropout
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.Dropout(dropout)) for d in dims
        ])
        if fusion == "gated":
            self.gate = GatedFusion(proj_dim, len(dims))
        elif fusion == "cross_attention":
            assert len(dims) == 2, "cross_attention butuh tepat 2 branch"
            self.cross = CrossAttentionFusion(proj_dim, heads, dropout)
        out_dim = proj_dim * len(dims) if fusion in ("concat", "cross_attention") else proj_dim
        self.classifier = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(out_dim, num_labels))
        self.aux = nn.ModuleList([nn.Linear(proj_dim, num_labels) for _ in dims]) if len(dims) > 1 else None

    def _branch_keep(self, n_branches, batch, device):
        """Mask (B, n): saat training, maksimal satu branch per sampel di-drop."""
        keep = torch.ones(batch, n_branches, device=device)
        if self.training and self.branch_dropout > 0 and n_branches > 1:
            drop = torch.rand(batch, device=device) < self.branch_dropout
            which = torch.randint(n_branches, (batch,), device=device)
            keep[drop, which[drop]] = 0.0
        return keep

    def forward(self, feats):
        gate = None
        pooled = [proj(f[1]) for proj, f in zip(self.projs, feats)]
        aux_logits = [head(p) for head, p in zip(self.aux, pooled)] if self.aux is not None else None
        keep = self._branch_keep(len(feats), pooled[0].size(0), pooled[0].device)
        if self.fusion == "cross_attention":
            (seq_a, _, mask_a), (seq_b, _, mask_b) = feats
            seq_a = self.projs[0](seq_a) * keep[:, 0, None, None]
            seq_b = self.projs[1](seq_b) * keep[:, 1, None, None]
            x = self.cross(seq_a, mask_a, seq_b, mask_b)
        else:
            pooled_kept = [p * keep[:, i:i + 1] for i, p in enumerate(pooled)]
            if self.fusion == "single":
                x = pooled_kept[0]
            elif self.fusion == "concat":
                x = torch.cat(pooled_kept, dim=-1)
            else:
                x, gate = self.gate(pooled_kept)
        return self.classifier(x), gate, aux_logits


class FusionClassifier(nn.Module):
    def __init__(self, branches, num_labels, fusion, proj_dim, heads, dropout, head_device, branch_dropout=0.0):
        super().__init__()
        self.branches = nn.ModuleDict(branches)
        self.head_device = head_device
        dims = [b.hidden_size for b in branches.values()]
        self.head = FusionHead(dims, num_labels, fusion, proj_dim, heads, dropout,
                               branch_dropout).to(head_device)

    def forward(self, batch):
        feats = []
        for name, branch in self.branches.items():
            ids, mask = batch[name]
            seq, pooled = branch(ids.to(branch.device), mask.to(branch.device))
            feats.append((seq.to(self.head_device), pooled.to(self.head_device), mask.to(self.head_device)))
        return self.head(feats)

    def param_groups(self, lrs):
        groups = [{"params": [p for p in branch.parameters() if p.requires_grad], "lr": lrs[name]}
                  for name, branch in self.branches.items()]
        groups.append({"params": list(self.head.parameters()), "lr": lrs["head"]})
        return groups

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        params = dict(self.named_parameters())
        for n, v in state.items():
            params[n].data.copy_(v.to(params[n].device))


def build_model(args, num_labels, head_device, llm_device):
    branches = {}
    if args.model in ("encoder", "fusion"):
        branches["encoder"] = EncoderBranch(args.encoder_name, head_device)
    if args.model in ("llm", "fusion"):
        branches["llm"] = LLMBranch(
            args.llm_name, llm_device, args.lora_r, args.lora_alpha, args.lora_dropout,
            compute_dtype=getattr(torch, args.llm_compute_dtype), grad_ckpt=not args.no_grad_ckpt)
    fusion = args.fusion if args.model == "fusion" else "single"
    return FusionClassifier(branches, num_labels, fusion, args.proj_dim, args.heads, args.dropout, head_device,
                            branch_dropout=args.branch_dropout if fusion != "single" else 0.0)
