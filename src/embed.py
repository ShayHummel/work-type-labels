"""Resumable embedding job — same code on Colab (CUDA) and on this Mac (MPS).

Embeds every prompt of train.jsonl followed by holdout.jsonl with one or more models and writes, per model:

    <out>/emb/<key>/chunks/00000.npy …   one file per CHUNK rows, written atomically (resume points)
    <out>/emb/<key>/train.npy            (24792, d) float16, L2-normalised, train.jsonl order
    <out>/emb/<key>/holdout.npy          (6199, d)  float16, holdout.jsonl order
    <out>/emb/<key>/manifest.json        model id, prompt, max_len, dim, row counts, ids hash, runtime, device

If the process dies (Colab GPU reclaimed), re-running the same command skips finished chunks and models.

Usage:
    python src/embed.py --models bge-m3 qwen3-0.6b --out data          # local
    python src/embed.py --models qwen3-4b qwen3-8b nomic-code --out /content/drive/MyDrive/zuzai-hw
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHUNK = int(os.environ.get("EMBED_CHUNK", 2000))

QWEN_INSTRUCTION = ("Instruct: Classify the developer's request to a coding assistant by the type of software "
                    "work it asks for\nQuery: ")
# key -> (hf model id, prompt prefix or None, approx. fp16 GB needed, extra kwargs)
MODELS = {
    "bge-m3": ("BAAI/bge-m3", None, 2, {}),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", None, 2, {}),
    "qwen3-0.6b-instr": ("Qwen/Qwen3-Embedding-0.6B", QWEN_INSTRUCTION, 2, {}),
    "qwen3-4b": ("Qwen/Qwen3-Embedding-4B", None, 9, {}),
    "qwen3-4b-instr": ("Qwen/Qwen3-Embedding-4B", QWEN_INSTRUCTION, 9, {}),
    "qwen3-8b": ("Qwen/Qwen3-Embedding-8B", None, 17, {}),
    "qwen3-8b-instr": ("Qwen/Qwen3-Embedding-8B", QWEN_INSTRUCTION, 17, {}),
    "nomic-code": ("nomic-ai/nomic-embed-code", "Represent this query for searching relevant code: ", 15, {}),
}


def load_texts(data_dir: Path) -> tuple[list[str], list[str], int]:
    rows = []
    for name in ("train.jsonl", "holdout.jsonl"):
        with open(data_dir / name) as f:
            rows.append([json.loads(line) for line in f])
    ids = [r["id"] for r in rows[0]] + [r["id"] for r in rows[1]]
    texts = [r["text"] for r in rows[0]] + [r["text"] for r in rows[1]]
    return ids, texts, len(rows[0])


def device_info() -> tuple[str, float, str]:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return "cuda", p.total_memory / 1e9, p.name
    if torch.backends.mps.is_available():
        return "mps", 24.0, "Apple MPS"
    return "cpu", 0.0, "cpu"


def log(out: Path, key: str, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{key}] {msg}"
    print(line, flush=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    with open(out / "logs" / f"embed_{key}.log", "a") as f:
        f.write(line + "\n")


def save_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def embed_model(key: str, texts: list[str], ids: list[str], n_train: int, out: Path, max_len: int,
                batch_size: int) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    model_id, prompt, need_gb, kwargs = MODELS[key]
    mdir = out / "emb" / key
    if (mdir / "manifest.json").exists():
        log(out, key, "already done — skipping")
        return
    dev, mem_gb, gpu = device_info()
    if dev == "cuda" and mem_gb < need_gb * 1.15:
        log(out, key, f"SKIP: needs ~{need_gb} GB in fp16, GPU {gpu} has {mem_gb:.0f} GB")
        return
    (mdir / "chunks").mkdir(parents=True, exist_ok=True)
    n_chunks = (len(texts) + CHUNK - 1) // CHUNK
    todo = [c for c in range(n_chunks) if not (mdir / "chunks" / f"{c:05d}.npy").exists()]
    log(out, key, f"device={dev} ({gpu}, {mem_gb:.0f} GB); {n_chunks - len(todo)}/{n_chunks} chunks already done")
    t_start = time.time()
    if todo:
        dtype = torch.bfloat16 if dev == "cuda" else torch.float16
        model = SentenceTransformer(model_id, device=dev, trust_remote_code=True,
                                    model_kwargs={"torch_dtype": dtype, **kwargs})
        model.max_seq_length = max_len
        bs = batch_size
        for c in todo:
            chunk = texts[c * CHUNK:(c + 1) * CHUNK]
            t0 = time.time()
            while True:
                try:
                    emb = model.encode(chunk, batch_size=bs, prompt=prompt, normalize_embeddings=True,
                                       convert_to_numpy=True, show_progress_bar=False)
                    break
                except torch.OutOfMemoryError:
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    torch.cuda.empty_cache() if dev == "cuda" else torch.mps.empty_cache()
                    log(out, key, f"OOM — retrying chunk {c} with batch_size={bs}")
            save_atomic(mdir / "chunks" / f"{c:05d}.npy", emb.astype(np.float16))
            log(out, key, f"chunk {c + 1}/{n_chunks} done in {time.time() - t0:.0f}s (bs={bs})")
        del model
    allv = np.concatenate([np.load(mdir / "chunks" / f"{c:05d}.npy") for c in range(n_chunks)])
    assert allv.shape[0] == len(texts) and np.isfinite(allv.astype(np.float32)).all()
    save_atomic(mdir / "train.npy", allv[:n_train])
    save_atomic(mdir / "holdout.npy", allv[n_train:])
    manifest = {"key": key, "model_id": model_id, "prompt": prompt, "max_len": max_len, "dim": int(allv.shape[1]),
                "n_train": n_train, "n_holdout": len(texts) - n_train, "dtype": "float16", "normalized": True,
                "ids_sha1": hashlib.sha1("\n".join(ids).encode()).hexdigest(), "device": f"{dev} {gpu}",
                "seconds_last_session": round(time.time() - t_start, 1)}
    (mdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(out, key, f"DONE dim={allv.shape[1]}")


def export_zip(out: Path, key: str) -> None:
    mdir = out / "emb" / key
    if not (mdir / "manifest.json").exists():
        return
    stage = Path("/tmp") / f"emb_{key}"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "emb" / key).mkdir(parents=True)
    for f in ("train.npy", "holdout.npy", "manifest.json"):
        shutil.copy(mdir / f, stage / "emb" / key / f)
    (out / "exports").mkdir(exist_ok=True)
    shutil.make_archive(str(out / "exports" / f"emb_{key}"), "zip", stage)
    log(out, key, f"exported {out / 'exports' / f'emb_{key}.zip'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--data-dir", type=Path, default=ROOT)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--export", action="store_true", help="zip finished models into <out>/exports/")
    args = ap.parse_args()
    ids, texts, n_train = load_texts(args.data_dir)
    for key in args.models:
        embed_model(key, texts, ids, n_train, args.out, args.max_len, args.batch_size)
        if args.export:
            export_zip(args.out, key)


if __name__ == "__main__":
    main()
