"""Import result zips downloaded from Colab (My Drive/zuzai-hw/exports/) into data/ and validate them.

Usage: uv run python src/import_colab.py ~/Downloads/emb_qwen3-8b.zip [more.zip …]
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def local_ids_sha1() -> str:
    ids = []
    for name in ("train.jsonl", "holdout.jsonl"):
        with open(ROOT / name) as f:
            ids += [json.loads(line)["id"] for line in f]
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()


def main(paths: list[str]) -> None:
    sha = local_ids_sha1()
    for p in paths:
        with zipfile.ZipFile(p) as z:
            z.extractall(ROOT / "data")
            manifests = [n for n in z.namelist() if n.endswith("manifest.json")]
        for m in manifests:
            d = (ROOT / "data" / m).parent
            man = json.loads((d / "manifest.json").read_text())
            problems = []
            if man["ids_sha1"] != sha:
                problems.append("ids hash differs from local jsonl")
            if (d / "train.npy").exists():  # embeddings
                tr, ho = np.load(d / "train.npy"), np.load(d / "holdout.npy")
                if tr.shape != (man["n_train"], man["dim"]) or ho.shape != (man["n_holdout"], man["dim"]):
                    problems.append(f"shape mismatch {tr.shape} {ho.shape}")
                if not (np.isfinite(tr.astype(np.float32)).all() and np.isfinite(ho.astype(np.float32)).all()):
                    problems.append("non-finite values")
            status = "OK" if not problems else "PROBLEM: " + "; ".join(problems)
            print(f"{p} → {d.relative_to(ROOT)}  [{man.get('model_id', '?')}, dim {man.get('dim', '?')}]  {status}")


if __name__ == "__main__":
    main(sys.argv[1:])
