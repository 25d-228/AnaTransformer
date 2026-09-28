"""Prepare only this task's official Multi30k English/French text and joint vocabulary."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

TASK_DATA = Path("/mango/homes/YUE_Ziran/workspace/ana-power-enfr-cogs-v1/data")
RAW_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"
SPLITS = {"train": ("train", 29_000), "dev": ("val", 1_014), "test": ("test_2016_flickr", 1_000)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=TASK_DATA)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    if data_dir != TASK_DATA.resolve():
        raise ValueError(f"This preparation script is scoped to {TASK_DATA}")
    os.environ["ANA_DATA"] = str(data_dir)
    folder = data_dir / "multi30k_enfr"
    manifest_path = folder / "preparation.json"
    tokenizer_path = data_dir / "multi30k_enfr.model"
    if tokenizer_path.exists() and not manifest_path.exists():
        raise RuntimeError(
            "An existing EN/FR tokenizer has no preparation record; refusing reuse."
        )

    # Validate all six downloads and all existing targets before writing any raw file.
    downloads: dict[Path, bytes] = {}
    sources: dict[str, str] = {}
    for split, (upstream, expected) in SPLITS.items():
        for language in ("en", "fr"):
            url = f"{RAW_BASE}/{upstream}.{language}.gz"
            request = Request(url, headers={"User-Agent": "AnaTransformer-Multi30k-ENFR"})
            with urlopen(request, timeout=90) as response:
                content = gzip.decompress(response.read())
            lines = content.decode("utf-8").splitlines()
            if len(lines) != expected or any(
                not line.strip() or line.strip() == "__NULL__" for line in lines
            ):
                raise ValueError(
                    f"Unexpected or empty rows in {url}: {len(lines)} (expected {expected})"
                )
            target = folder / f"{split}.{language}"
            if target.exists() and target.read_bytes() != content:
                raise RuntimeError(f"Refusing to overwrite different existing data: {target}")
            downloads[target] = content
            sources[target.name] = url

    provenance = {
        "corpus": "multi30k_enfr",
        "languages": ["en", "fr"],
        "files": sources,
        "split_counts": {name: count for name, (_, count) in SPLITS.items()},
        "tokenizer": "joint SentencePiece BPE, train English + train French only",
        "requested_vocab_size": 10_000,
        "german_tokenizer_reused": False,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in provenance.items()):
            raise RuntimeError(
                "Existing preparation record describes a different corpus or tokenizer."
            )
    folder.mkdir(parents=True, exist_ok=True)
    for target, content in downloads.items():
        if not target.exists():
            target.write_bytes(content)

    from ana.data.corpora import build_corpus
    from ana.experiment import prepare

    corpus = build_corpus("multi30k_enfr")
    tokenizer, splits = prepare(corpus, smoke=False)
    counts = {name: len(examples) for name, examples in splits.items()}
    if counts != provenance["split_counts"]:
        raise RuntimeError(f"Loaded split counts differ from official files: {counts}")
    if len(tokenizer) != 10_000:
        raise RuntimeError(f"Expected a joint 10k vocabulary, got {len(tokenizer)}")
    record = {
        **provenance,
        "actual_vocab_size": len(tokenizer),
        "tokenizer_path": str(tokenizer_path),
        "architecture": asdict(corpus.architecture),
        "recipe": asdict(corpus.recipe),
    }
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print("MULTI30K_ENFR_DATA_READY", flush=True)


if __name__ == "__main__":
    main()
