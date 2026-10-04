"""Step 3a — Measure embedding runtime on this machine before committing to a model.

For each model: real token-length distribution of train+holdout (model tokenizer), and throughput on a
random sample of 600 prompts (lengths as in the data) at max length 512, sorted-by-length batching, fp16 on
MPS. Extrapolates to all 30,991 prompts. Larger models are extrapolated by parameter count.

Run: uv run python src/03_embed_benchmark.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"BAAI/bge-m3": 0.568e9, "Qwen/Qwen3-Embedding-0.6B": 0.6e9}
MAX_LEN = 512
N_SAMPLE = 600

texts = pd.concat([pd.read_json(ROOT / "train.jsonl", lines=True).text,
                   pd.read_json(ROOT / "holdout.jsonl", lines=True).text]).tolist()
sample = list(np.random.default_rng(0).choice(texts, N_SAMPLE, replace=False))
device = "mps" if torch.backends.mps.is_available() else "cpu"
rows = []
for name, params in MODELS.items():
    model = SentenceTransformer(name, device=device, model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = MAX_LEN
    lens = np.array([len(x) for x in model.tokenizer(texts, add_special_tokens=True)["input_ids"]])
    model.encode(sample[:32], batch_size=16)  # warm-up
    t0 = time.perf_counter()
    model.encode(sample, batch_size=16)
    if device == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    full_min = dt / N_SAMPLE * len(texts) / 60
    rows.append({"model": name, "params": f"{params / 1e9:.2f}B", "median tokens": int(np.median(lens)),
                 "p90 tokens": int(np.percentile(lens, 90)), ">512 tokens": f"{(lens > 512).mean():.1%}",
                 ">1024 tokens": f"{(lens > 1024).mean():.1%}", "prompts/s": round(N_SAMPLE / dt, 1),
                 "est. all 30,991 (min)": round(full_min, 1)})
    print(rows[-1], flush=True)
    del model
    torch.mps.empty_cache()

# Extrapolate from the Qwen3 0.6B rate: Qwen3-Embedding 4B/8B and nomic-embed-code (Qwen2.5-Coder) share its
# decoder architecture. Lower bound: 8B in fp16 needs ~16 GB of the 24 GB unified memory and may swap.
per_param_min = rows[1]["est. all 30,991 (min)"] / MODELS["Qwen/Qwen3-Embedding-0.6B"]
for name, params in {"Qwen/Qwen3-Embedding-4B": 4.0e9, "nomic-ai/nomic-embed-code (7B)": 7.1e9,
                     "Qwen/Qwen3-Embedding-8B": 7.6e9}.items():
    rows.append({"model": name + " (extrapolated, lower bound)", "params": f"{params / 1e9:.1f}B",
                 "est. all 30,991 (min)": round(per_param_min * params, 0)})
df = pd.DataFrame(rows)
(ROOT / "reports" / "03_embed_benchmark.md").write_text(
    "# Step 3a — Embedding runtime benchmark (Apple M5 Pro, MPS, fp16, max_len 512)\n\n"
    + df.to_markdown(index=False) + "\n")
print(df.to_markdown(index=False))
