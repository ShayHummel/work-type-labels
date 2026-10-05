"""Resumable fine-tuning job: one (model, fold) at a time. Same code on Colab (CUDA) and on this Mac (MPS).

For fold k it trains on the other four folds (data/folds.csv) and writes
    <out>/ft/<key>/fold{k}/val_proba.npy      (n_fold_k, 10) softmax, LABELS order
    <out>/ft/<key>/fold{k}/val_idx.npy        row indices into train.jsonl
    <out>/ft/<key>/fold{k}/holdout_proba.npy  (6199, 10)
    <out>/ft/<key>/fold{k}/done.json          config, timings, device, fold-k metrics
    <out>/exports/ft_<key>.zip                every finished fold of <key> (re-zipped after each fold)
Checkpoints go to <out>/ft/<key>/fold{k}/ckpt/ (or --ckpt-root/<key>/fold{k}) every --save-steps steps; a rerun
resumes from the last complete one and skips finished folds. Checkpoints are deleted once the fold is done.
Results per fold are small (≈ 0.5 MB); checkpoints are not (DeBERTa-v3-base ≈ 2.2 GB with optimizer state).

The loss is weighted cross-entropy computed in float32 on float32 logits (TASK.md: a Half/Float mismatch in
the weighted loss made a larger encoder collapse to one class).

Usage:
    python src/finetune.py --model deberta-v3-base --folds 0 --out /content/drive/MyDrive/zuzai-hw
    python src/finetune.py --model deberta-v3-xsmall --folds 0 --limit 400 --max-steps 20 --out /tmp/x  # smoke test
"""

import argparse
import os
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Bug fix", "Feature dev", "Refactoring", "Architecting", "Researching",
          "Testing", "Review", "Optimize", "Setup", "Other"]

# key -> hf id, lr, per-device batch, grad accumulation, max_len, lora
MODELS = {
    "deberta-v3-xsmall": dict(hf="microsoft/deberta-v3-xsmall", lr=5e-5, bs=16, accum=1, max_len=512),  # smoke test
    "deberta-v3-base": dict(hf="microsoft/deberta-v3-base", lr=2e-5, bs=16, accum=1, max_len=512),
    "deberta-v3-large": dict(hf="microsoft/deberta-v3-large", lr=1e-5, bs=8, accum=2, max_len=512),
    "modernbert-large": dict(hf="answerdotai/ModernBERT-large", lr=2e-5, bs=16, accum=1, max_len=1024),
    "mmbert-base": dict(hf="jhu-clsp/mmBERT-base", lr=3e-5, bs=16, accum=1, max_len=1024),
    # LoRA: frozen base in bf16 + gradient checkpointing; 512 tokens covers ~90% of prompts whole (head+tail for the
    # rest). At 1024 tokens × batch 16 with an fp32 base, Qwen3-0.6B ran out of memory on a 40 GB A100.
    "qwen3-0.6b-lora": dict(hf="Qwen/Qwen3-0.6B", lr=1e-4, bs=16, accum=1, max_len=512, lora=True),
    "qwen3-1.7b-lora": dict(hf="Qwen/Qwen3-1.7B", lr=1e-4, bs=8, accum=2, max_len=512, lora=True),
    "qwen3-4b-lora": dict(hf="Qwen/Qwen3-4B", lr=1e-4, bs=8, accum=2, max_len=512, lora=True),
    "qwen3-8b-lora": dict(hf="Qwen/Qwen3-8B", lr=5e-5, bs=8, accum=2, max_len=512, lora=True),
    "qwen3-emb-8b-lora": dict(hf="Qwen/Qwen3-Embedding-8B", lr=5e-5, bs=8, accum=2, max_len=512, lora=True),
    # Full fine-tuning (all weights, fp32 master + bf16 autocast, gradient checkpointing). 8B in full does not fit
    # one 40 GB A100 (bf16 weights + grads + fp32 Adam ≈ 130 GB), so the LoRA-vs-full comparison uses 0.6B/1.7B.
    "qwen3-0.6b-full": dict(hf="Qwen/Qwen3-0.6B", lr=2e-5, bs=16, accum=1, max_len=512, gc=True),
    "qwen3-1.7b-full": dict(hf="Qwen/Qwen3-1.7B", lr=1e-5, bs=8, accum=2, max_len=512, gc=True),
}
TAIL_TOKENS = 128  # head+tail truncation keeps the last 128 tokens (the request often comes after pasted code)


def log(out: Path, key: str, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{key}] {msg}"
    print(line, flush=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    with open(out / "logs" / f"ft_{key}.log", "a") as f:
        f.write(line + "\n")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, enc: list[list[int]], labels: np.ndarray | None):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, i):
        item = {"input_ids": self.enc[i], "attention_mask": [1] * len(self.enc[i])}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def special_wrap(tok) -> tuple[list[int], list[int]]:
    """Prefix/suffix special tokens the tokenizer adds (e.g. [CLS] … [SEP]); works across tokenizer classes."""
    bare = tok("hello", add_special_tokens=False)["input_ids"]
    full = tok("hello")["input_ids"]
    for i in range(len(full) - len(bare) + 1):
        if full[i:i + len(bare)] == bare:
            return full[:i], full[i + len(bare):]
    return [], []


def encode(tok, texts: list[str], max_len: int) -> list[list[int]]:
    pre, suf = special_wrap(tok)
    budget = max_len - len(pre) - len(suf)
    out = []
    for ids in tok(texts, add_special_tokens=False, truncation=False)["input_ids"]:
        if len(ids) > budget:
            tail = TAIL_TOKENS if budget >= 2 * TAIL_TOKENS else budget // 4  # short max_len (smoke tests) only
            ids = ids[:budget - tail] + ids[-tail:]
        out.append(pre + ids + suf)
    return out


def last_complete_checkpoint(ckpt_dir: Path) -> str | None:
    """Newest checkpoint whose save finished. A GPU lost mid-save leaves a partial folder: delete it."""
    if not ckpt_dir.exists():
        return None
    needed = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    for c in sorted(ckpt_dir.glob("checkpoint-*"), key=lambda c: int(c.name.split("-")[1]), reverse=True):
        if all((c / f).exists() for f in needed):
            return str(c)
        shutil.rmtree(c, ignore_errors=True)
    return None


def bench(trainer, key: str, cfg: dict, n_train: int, args, gpu: str, t0: float) -> None:
    from transformers import TrainerCallback

    stamps = []

    class Clock(TrainerCallback):
        def on_step_end(self, *a, **k):
            stamps.append(time.time())

    trainer.add_callback(Clock())
    trainer.args.save_strategy = "no"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    skip = min(5, len(stamps) // 4)  # first steps include warm-up / compilation
    sec_per_step = (stamps[-1] - stamps[skip]) / max(1, len(stamps) - 1 - skip)
    steps_per_epoch = -(-n_train // (cfg["bs"] * cfg["accum"]))
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    res = {"key": key, "hf": cfg["hf"], "device": gpu, "gradient_checkpointing": not args.no_gc, "bench_steps": len(stamps), "sec_per_step": round(sec_per_step, 3),
           "steps_per_epoch": steps_per_epoch, "peak_gpu_gb": round(peak, 1),
           **{f"est_fold_min_{e}ep": round(sec_per_step * steps_per_epoch * e / 60 + 3, 0) for e in (1, 2, 3)},
           "note": "random batches; the length-grouped sampler starts with the longest batch, so the estimate is "
                   "slightly pessimistic. +3 min for loading and prediction."}
    (args.out / "bench").mkdir(parents=True, exist_ok=True)
    (args.out / "bench" / f"{key}.json").write_text(json.dumps(res, indent=2))
    log(args.out, key, f"BENCH [{gpu}, gc={'off' if args.no_gc else 'on'}, bs={cfg['bs']}x{cfg['accum']}] "
                       f"{sec_per_step:.2f} s/step, peak {peak:.1f} GB → est. per fold: "
                       f"{res['est_fold_min_2ep']:.0f} min (2 ep), {res['est_fold_min_3ep']:.0f} min (3 ep); "
                       f"5 folds × 3 ep ≈ {5 * res['est_fold_min_3ep'] / 60:.1f} h")


def export_zip(out: Path, key: str) -> None:
    src = out / "ft" / key
    stage = Path("/tmp") / f"ft_{key}"
    shutil.rmtree(stage, ignore_errors=True)
    for fd in sorted(src.glob("fold*")):
        if (fd / "done.json").exists():
            dst = stage / "ft" / key / fd.name
            dst.mkdir(parents=True)
            for f in ("val_proba.npy", "val_idx.npy", "holdout_proba.npy", "done.json"):
                shutil.copy(fd / f, dst / f)
    if stage.exists():
        (out / "exports").mkdir(exist_ok=True)
        shutil.make_archive(str(out / "exports" / f"ft_{key}"), "zip", stage)
        log(out, key, f"exported {out / 'exports' / f'ft_{key}.zip'}")


def run_fold(key: str, fold: int, args) -> None:
    from sklearn.metrics import f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainingArguments)

    cfg = dict(MODELS[key])
    if args.batch_size:
        cfg["bs"] = args.batch_size
    if args.accum:
        cfg["accum"] = args.accum
    fdir = args.out / "ft" / key / f"fold{fold}"
    if (fdir / "done.json").exists():
        log(args.out, key, f"fold {fold} already done — skipping")
        return
    fdir.mkdir(parents=True, exist_ok=True)
    # Checkpoints (weights + optimizer, GBs) can live off Drive (--ckpt-root /content/ckpt); results stay in fdir.
    ckpt = (args.ckpt_root / key / f"fold{fold}") if args.ckpt_root else (fdir / "ckpt")
    t0 = time.time()

    train = pd.read_json(args.data_dir / "train.jsonl", lines=True)
    hold = pd.read_json(args.data_dir / "holdout.jsonl", lines=True)
    folds = pd.read_csv(args.data_dir / "data" / "folds.csv")
    assert (folds.id.values == train.id.values).all()
    y = train.label.map({lab: i for i, lab in enumerate(LABELS)}).values
    tr_idx = np.flatnonzero(folds.fold.values != fold)
    va_idx = np.flatnonzero(folds.fold.values == fold)
    if args.limit:
        tr_idx, va_idx, hold = tr_idx[:args.limit], va_idx[:args.limit // 4], hold.iloc[:args.limit // 4]

    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else dev
    bf16 = dev == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8  # native bf16 (A100/L4), not T4 emulation
    fp16 = dev == "cuda" and not bf16
    max_len = args.max_len or cfg["max_len"]

    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lora = bool(cfg.get("lora"))
    use_gc = bool(cfg.get("gc", lora)) and not args.no_gc  # off = faster, needs memory (e.g. H100 80 GB)
    base_dtype = torch.bfloat16 if (lora and bf16) else torch.float32  # frozen base may be bf16; trained weights fp32
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf"], num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)}, dtype=base_dtype)
    model.config.pad_token_id = tok.pad_token_id
    if lora:
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()  # needed for gradient checkpointing with a frozen base
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                                                 target_modules="all-linear"))
        for name, prm in model.named_parameters():  # adapters and the new head train in fp32
            if prm.requires_grad:
                prm.data = prm.data.float()

    texts = train.text.values
    ds_tr = TextDataset(encode(tok, list(texts[tr_idx]), max_len), y[tr_idx])
    ds_va = TextDataset(encode(tok, list(texts[va_idx]), max_len), y[va_idx])
    ds_ho = TextDataset(encode(tok, hold.text.tolist(), max_len), None)

    counts = np.bincount(y[tr_idx], minlength=len(LABELS)).astype(np.float64)
    if args.class_weight == "inv":
        w = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
    elif args.class_weight == "sqrt_inv":
        w = np.sqrt(counts.sum() / (len(LABELS) * np.maximum(counts, 1)))
    else:
        w = np.ones(len(LABELS))
    class_w = torch.tensor(w, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits.float()  # loss strictly in float32
            loss = F.cross_entropy(logits, labels, weight=class_w.to(logits.device))
            return (loss, outputs) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(ckpt), num_train_epochs=args.epochs, max_steps=args.max_steps or -1,
        learning_rate=args.lr or cfg["lr"], per_device_train_batch_size=cfg["bs"],
        per_device_eval_batch_size=cfg["bs"] * 2, gradient_accumulation_steps=cfg["accum"], warmup_steps=0.1,
        weight_decay=0.01, lr_scheduler_type="linear", bf16=bf16, fp16=fp16, train_sampling_strategy="group_by_length",
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=1, eval_strategy="no",
        gradient_checkpointing=use_gc, gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        logging_steps=50, report_to="none", seed=42, dataloader_num_workers=2 if dev == "cuda" else 0)
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_tr, processing_class=tok,
                              data_collator=DataCollatorWithPadding(tok))
    last = last_complete_checkpoint(ckpt)
    log(args.out, key, f"fold {fold}: train {len(ds_tr)}, val {len(ds_va)}, device {gpu}, bf16={bf16} fp16={fp16}, "
                       f"class_weight={args.class_weight}, " + (f"RESUMING from {last}" if last else "starting"))
    if args.bench:  # measure speed and memory only: no checkpoints, no predictions, no results
        bench(trainer, key, cfg, len(ds_tr), args, gpu, t0)
        return
    trainer.train(resume_from_checkpoint=last)
    t_train = time.time() - t0

    def proba(ds):
        """Own prediction loop. Trainer.predict() must not be used: with train_sampling_strategy="group_by_length"
        transformers 5.x also length-groups (= reorders) the *prediction* batches, so its outputs do not line up
        with the dataset rows. Here batches are length-sorted for speed and the original order is restored."""
        model = trainer.model.eval()
        collate = DataCollatorWithPadding(tok)
        order = np.argsort([len(e) for e in ds.enc], kind="stable")
        out = np.zeros((len(ds), len(LABELS)), dtype=np.float32)
        bs = cfg["bs"] * 2
        with torch.no_grad():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                batch = collate([{"input_ids": ds.enc[i], "attention_mask": [1] * len(ds.enc[i])} for i in idx])
                batch = {k: v.to(model.device) for k, v in batch.items()}
                with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=bf16):
                    logits = model(**batch).logits.float()
                out[idx] = torch.softmax(logits, -1).cpu().numpy()
        return out

    pv, ph = proba(ds_va), proba(ds_ho)
    f1 = f1_score(y[va_idx], pv.argmax(1), labels=range(len(LABELS)), average=None, zero_division=0)
    np.save(fdir / "val_proba.npy", pv)
    np.save(fdir / "val_idx.npy", va_idx)
    np.save(fdir / "holdout_proba.npy", ph)
    import transformers
    done = {"key": key, "hf": cfg["hf"], "fold": fold, "n_train": len(ds_tr), "n_val": len(ds_va),
            "max_len": max_len, "truncation": f"head+tail{TAIL_TOKENS}", "epochs": args.epochs,
            "lr": args.lr or cfg["lr"], "batch": cfg["bs"] * cfg["accum"], "class_weight": args.class_weight,
            "lora": lora, "base_dtype": str(base_dtype), "gradient_checkpointing": use_gc, "device": gpu, "bf16": bf16, "fp16": fp16,
            "train_seconds": round(t_train, 1), "total_seconds": round(time.time() - t0, 1),
            "transformers": transformers.__version__, "predict_order": "own-loop-v2", "limit": args.limit, "max_steps": args.max_steps,
            "val_accuracy": float((pv.argmax(1) == y[va_idx]).mean()), "val_macro_F1": float(f1.mean()),
            "val_min_F1": float(f1.min()), "val_F1": dict(zip(LABELS, map(float, f1)))}
    (fdir / "done.json").write_text(json.dumps(done, indent=2))
    shutil.rmtree(ckpt, ignore_errors=True)
    log(args.out, key, f"fold {fold} DONE in {done['total_seconds']:.0f}s — val macro-F1 {f1.mean():.3f}, "
                       f"min-F1 {f1.min():.3f}, acc {done['val_accuracy']:.3f}")
    del trainer, model
    if dev == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--ckpt-root", type=Path, default=None,
                    help="where checkpoints go (default: inside --out). On Colab use local VM disk to spare Drive")
    ap.add_argument("--data-dir", type=Path, default=ROOT)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="per-device batch (default: model config)")
    ap.add_argument("--class-weight", choices=["inv", "sqrt_inv", "none"], default="inv")
    ap.add_argument("--save-steps", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many training rows")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after this many steps")
    ap.add_argument("--accum", type=int, default=None, help="gradient accumulation (default: model config)")
    ap.add_argument("--no-gc", action="store_true", help="disable gradient checkpointing (faster, more GPU memory)")
    ap.add_argument("--bench", action="store_true", help="speed/memory benchmark: use with --max-steps (e.g. 40)")
    args = ap.parse_args()
    for k in args.folds:
        run_fold(args.model, k, args)
        if not args.bench:
            export_zip(args.out, args.model)


if __name__ == "__main__":
    main()
