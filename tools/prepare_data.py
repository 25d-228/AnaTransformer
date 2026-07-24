"""Rebuild the data directory, and check it.

All three corpora are committed, so nothing here needs to run for the repository to work. It
exists to record where the text came from and to prove it has not changed: a result can then be
traced to the exact bytes that produced it, rather than to whatever a hub happens to be serving
on the day someone tries to reproduce it.

`--verify` checks every file against a manifest of sizes and hashes. That matters when the data
is copied to a training machine, because a truncated file does not announce itself — it trains a
model on part of a corpus and reports a number that looks perfectly reasonable.

    python tools/prepare_data.py --verify   # check the committed data against its manifest
    python tools/prepare_data.py            # refetch the two downloadable corpora, then check
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import urllib.request
import zipfile

# Both corpora are fetched from the Hugging Face hub. `$HF_ENDPOINT` redirects that to a mirror,
# which is the same variable the huggingface libraries themselves honour. It is not a nicety: the
# hub is unreachable from mainland China, and a training box rented there can pull the same bytes
# from https://hf-mirror.com at full speed while an scp from outside the country crawls.
#
#     HF_ENDPOINT=https://hf-mirror.com python tools/prepare_data.py
#
# The manifest is what makes this safe. Whatever the mirror serves is hashed and checked against
# the committed manifest, so a mirror serving different bytes is caught rather than trusted.
HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

IWSLT_ZIP = f"{HUB}/datasets/bbaaaa/iwslt14-de-en/resolve/main/data/de-en.zip"
MULTI30K = HUB + "/datasets/bentrevett/multi30k/resolve/main/{split}.jsonl"

EXPECTED = {
    "cogs": {"train": 24_155, "dev": 3_000, "test": 3_000, "gen": 21_000},
    "iwslt14": {"train": 160_239, "dev": 7_283, "test": 6_750},
    "multi30k": {"train": 29_000, "dev": 1_014, "test": 1_000},
}


def fetch(url: str) -> bytes:
    """Read a URL, announcing an ordinary user agent.

    urllib identifies itself as `Python-urllib/3.x`, and the CDN behind the Hugging Face mirror
    answers that with 403. The same request from curl succeeds. Nothing about the bytes changes;
    only whether they are served at all.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def write_pairs(folder: str, split: str, english: list[str], german: list[str]) -> int:
    kept = [(e, g) for e, g in zip(english, german, strict=True) if e and g]
    for language, index in (("en", 0), ("de", 1)):
        with open(os.path.join(folder, f"{split}.{language}"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(pair[index] for pair in kept) + "\n")
    return len(kept)


def count_cogs(out: str) -> dict[str, int]:
    """COGS is committed, not fetched. Count what is there and let the caller check it."""
    folder = os.path.join(out, "cogs")
    counts = {}
    for split in ("train", "dev", "test", "gen"):
        path = os.path.join(folder, f"{split}.tsv")
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing. COGS ships with the repository.")
        with open(path, encoding="utf-8") as handle:
            counts[split] = sum(1 for _ in handle)
    return counts


def prepare_iwslt(out: str) -> dict[str, int]:
    folder = os.path.join(out, "iwslt14")
    os.makedirs(folder, exist_ok=True)

    archive = zipfile.ZipFile(io.BytesIO(fetch(IWSLT_ZIP)))

    counts = {}
    for theirs, ours in (("train", "train"), ("valid", "dev"), ("test", "test")):
        text = {
            language: archive.read(f"de-en/{theirs}.{language}").decode("utf-8").splitlines()
            for language in ("en", "de")
        }
        counts[ours] = write_pairs(
            folder,
            ours,
            [line.strip() for line in text["en"]],
            [line.strip() for line in text["de"]],
        )
    return counts


def prepare_multi30k(out: str) -> dict[str, int]:
    folder = os.path.join(out, "multi30k")
    os.makedirs(folder, exist_ok=True)

    counts = {}
    for theirs, ours in (("train", "train"), ("val", "dev"), ("test", "test")):
        body = fetch(MULTI30K.format(split=theirs)).decode("utf-8")
        rows = [json.loads(line) for line in body.splitlines() if line.strip()]
        counts[ours] = write_pairs(
            folder,
            ours,
            [row["en"].replace("\n", " ").strip() for row in rows],
            [row["de"].replace("\n", " ").strip() for row in rows],
        )
    return counts


def manifest_for(out: str) -> dict:
    files = {}
    for folder, _, names in os.walk(out):
        for name in sorted(names):
            if name == "MANIFEST.json":
                continue
            path = os.path.join(folder, name)
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            files[os.path.relpath(path, out)] = {
                "bytes": os.path.getsize(path),
                "sha256": digest,
            }
    return {"files": files}


def differences(expected: dict, actual: dict) -> list[str]:
    problems = []
    for name, want in expected.items():
        got = actual.get(name)
        if got is None:
            problems.append(f"missing: {name}")
        elif got["sha256"] != want["sha256"]:
            problems.append(f"changed: {name} ({want['bytes']:,} -> {got['bytes']:,} bytes)")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument(
        "--verify", action="store_true", help="check against the manifest, build nothing"
    )
    parser.add_argument(
        "--rewrite-manifest",
        action="store_true",
        help="accept what was fetched as the new truth. Only for adding a corpus.",
    )
    args = parser.parse_args()

    manifest_path = os.path.join(args.out, "MANIFEST.json")

    if args.verify:
        with open(manifest_path, encoding="utf-8") as handle:
            expected = json.load(handle)["files"]

        problems = differences(expected, manifest_for(args.out)["files"])
        if problems:
            print("\n".join(problems))
            raise SystemExit("the data directory does not match its manifest")
        print(f"all {len(expected)} files match the manifest")
        return

    counts = {
        "cogs": count_cogs(args.out),
        "iwslt14": prepare_iwslt(args.out),
        "multi30k": prepare_multi30k(args.out),
    }

    for corpus, splits in counts.items():
        got = dict(splits)
        want = EXPECTED[corpus]
        line = "  ".join(f"{k} {v:,}" for k, v in got.items())
        status = "ok" if got == want else f"UNEXPECTED, wanted {want}"
        print(f"  {corpus:10} {line:48} {status}")
        if got != want:
            raise SystemExit(f"{corpus} does not have the row counts it should")

    manifest = manifest_for(args.out)
    manifest["counts"] = counts

    # A manifest already exists, so this is a REBUILD and the bytes just fetched have something
    # to be checked against. Checking rather than overwriting is the whole point: a mirror, or a
    # hub that has quietly re-uploaded a file, must not be able to change the data underneath a
    # study that has already been run against it. Overwriting would make that invisible.
    if os.path.exists(manifest_path) and not args.rewrite_manifest:
        with open(manifest_path, encoding="utf-8") as handle:
            expected = json.load(handle)["files"]

        problems = differences(expected, manifest["files"])
        if problems:
            print("\n".join(problems))
            raise SystemExit(
                "what was fetched does not match the committed manifest. The source has changed "
                "under us, or the mirror is serving something else. Do not train on it."
            )
        print(f"\n  all {len(expected)} files match the committed manifest, byte for byte")
        return

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    total = sum(entry["bytes"] for entry in manifest["files"].values())
    print(f"\n  {len(manifest['files'])} files, {total / 1e6:.1f} MB, hashed into {manifest_path}")


if __name__ == "__main__":
    main()
