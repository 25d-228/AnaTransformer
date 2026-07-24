"""Read `runs/` and write the result tables into `results/`.

The tables are generated, not transcribed. A number in `results/` can be traced to the
`results.json` that produced it, and regenerating after a new cell lands cannot introduce a
typo into a table nobody re-reads.

Two comparisons are drawn, because two different questions are being asked.

    against `baseline`      the ordinary transformer in the corpus's published configuration.
                            This is the accuracy every other model is trading away, and every
                            model is measured against it.
    against `shared_qkv`    the shared projection the ana_* operators are placed on top of, at
                            the same parameter count. This is the accuracy they are there to
                            recover, and the ana_* models are measured against it.

    a corpus run over SEEDS      +/- is the spread across training runs, and the comparison is
                                 a paired t-test across seeds
    a corpus run ONCE            +/- is a bootstrap over the test set, and the comparison is
                                 Koehn's paired bootstrap: both models are scored on the same
                                 resample, so the variation they share cancels

A cell with no run prints an em dash. Every model keeps its row.
"""

from __future__ import annotations

import json
import os
import statistics as st

from ana.cli.__main__ import DEFAULT_SEEDS
from ana.data.corpora import CORPORA, build_corpus
from ana.registry import REGISTRY
from ana.stats import across_seed_test, bootstrap_score, paired_bootstrap

RUNS = "runs"
OUT = "results"
BASELINE = "baseline"
SHARED = "shared_qkv"

# Two references, two marks, in the order the typographic footnote sequence runs (* then dagger).
# A bare `*` beside every number would leave the reader to remember which table they were in.
BASELINE_MARK = "*"
SHARED_MARK = "†"

PUBLISHED = {
    "cogs": "96% test (Kim & Linzen 2020); ~80% gen (Csordas et al. 2021, this embedding scaling)",
    "multi30k": "41.02 BLEU (Wu et al. 2021, transformer_tiny, beam 5, averaged checkpoints)",
    "iwslt14": "34.4 BLEU (Wu et al. 2019, transformer_iwslt_de_en, beam 4, averaged checkpoints)",
}

# The shape a model is built at is a property of the architecture, not of whether it has been
# trained, so the params and saved columns are filled from arithmetic and a row is never blank.
VOCAB = {"cogs": 835, "multi30k": 10_000, "iwslt14": 10_000}


def cell(corpus: str, model: str, seed: int) -> dict | None:
    path = os.path.join(RUNS, f"{corpus}_{model}_seed{seed}", "results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def lines(corpus: str, model: str, seed: int, kind: str, split: str) -> list[str] | None:
    path = os.path.join(RUNS, f"{corpus}_{model}_seed{seed}", f"{kind}.{split}.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read().splitlines()


def shape(corpus: str, model: str) -> tuple[float, float]:
    """Parameters and the share of the baseline they save, from arithmetic rather than a run."""
    from ana.registry import model_parameters

    config = build_corpus(corpus).model_config(VOCAB[corpus])
    count = model_parameters(model, config)
    base = model_parameters("baseline", config)
    return count / 1e6, (base - count) / base


def comparisons(
    corpus: str,
    reference: str,
    models: list[str],
    complete: dict,
    seeds: list[int],
    splits: list[str],
    metric,
    preface: str,
    mark: str,
) -> list[str]:
    """One `against X` section: every model in `models` measured against `reference`.

    Each reference carries its own `mark`, so a significance symbol names the test it came from
    and two tests are never read as one.
    """
    seeded = len(seeds) > 1
    if reference not in complete:
        return []

    out = ["", f"## against `{reference}`", "", preface, ""]
    if seeded:
        out.append(f"Paired across seeds. `{mark}` marks p < 0.05, uncorrected for multiplicity.")
    else:
        out.append(
            "Koehn's paired bootstrap: both models are scored on the same resample of the test "
            f"set, so the variation they share cancels. `{mark}` marks p < 0.05, uncorrected for "
            "multiplicity."
        )
    out.append("")
    out.append("| split | model | difference | 95% interval | p | |")
    out.append("|---|---|---|---|---|---|")

    for split in splits:
        for m in models:
            if m not in complete or m == reference:
                continue
            if seeded:
                shared = sorted(set(complete[m]) & set(complete[reference]))
                shared = [s for s in shared if split in complete[m][s]["scores"]]
                if len(shared) < 2:
                    continue
                result = across_seed_test(
                    [complete[m][s]["scores"][split] for s in shared],
                    [complete[reference][s]["scores"][split] for s in shared],
                )
            else:
                hyp = lines(corpus, m, seeds[0], "hypotheses", split)
                base = lines(corpus, reference, seeds[0], "hypotheses", split)
                ref = lines(corpus, BASELINE, seeds[0], "references", split)
                if not (hyp and base and ref) or len(hyp) != len(ref):
                    continue
                result = paired_bootstrap(hyp, base, ref, metric)

            out.append(
                f"| {split} | `{m}` | {result.difference:+.2f} | "
                f"[{result.low:+.2f}, {result.high:+.2f}] | {result.p_value:.3f} | "
                f"{mark if result.significant else ''} |"
            )

    return out


def write(corpus: str) -> str | None:
    seeds = [int(s) for s in DEFAULT_SEEDS[corpus].split(",")]
    seeded = len(seeds) > 1

    runs = {m: {s: cell(corpus, m, s) for s in seeds} for m in REGISTRY}
    runs = {m: {s: r for s, r in d.items() if r} for m, d in runs.items()}

    # Only a model with every one of its seeds gets a score. A mean drawn from fewer runs than the
    # rows beside it is a different quantity, and it is not one this table reports.
    complete = {m: d for m, d in runs.items() if len(d) == len(seeds)}
    if BASELINE not in complete:
        return None

    splits = sorted({k for d in complete.values() for r in d.values() for k in r["scores"]})
    metric = build_corpus(corpus).metric

    out = [f"# {corpus}", ""]
    out.append(f"metric: **{metric.name}** &nbsp;|&nbsp; published: {PUBLISHED[corpus]}")
    out.append("")
    if seeded:
        out.append(f"{len(seeds)} training runs (seeds {seeds}). **±** is the spread across them.")
    else:
        out.append(
            "1 training run, as the paper for this corpus reports. "
            "**±** is a bootstrap over the test set."
        )
    out.append("")

    head = ["model", "params", "saved"] + splits
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "---|" * len(head))

    for m in REGISTRY:
        params, saved = shape(corpus, m)
        row = [f"`{m}`", f"{params:.2f}M", f"{saved:.1%}"]
        for split in splits:
            values = [r["scores"][split] for r in complete.get(m, {}).values()]
            if not values:
                row.append("—")
            elif seeded:
                row.append(f"{st.mean(values):.2f} ± {st.stdev(values):.2f}")
            else:
                hyp = lines(corpus, m, seeds[0], "hypotheses", split)
                ref = lines(corpus, BASELINE, seeds[0], "references", split)
                if hyp and ref and len(hyp) == len(ref):
                    band = bootstrap_score(hyp, ref, metric)
                    row.append(f"{band.score:.2f} ± {band.half_width:.2f}")
                else:
                    row.append(f"{values[0]:.2f}")
        out.append("| " + " | ".join(row) + " |")

    ana = [m for m in REGISTRY if m.startswith("ana_")]
    shared_params = shape(corpus, SHARED)[0]
    spread = max(abs(shape(corpus, m)[0] - shared_params) for m in ana) / shared_params

    out += comparisons(
        corpus,
        BASELINE,
        list(REGISTRY),
        complete,
        seeds,
        splits,
        metric,
        "The accuracy each model trades away for its parameter saving.",
        BASELINE_MARK,
    )
    out += comparisons(
        corpus,
        SHARED,
        ana,
        complete,
        seeds,
        splits,
        metric,
        "The accuracy each ana_* operator recovers over the shared projection it is built on. "
        f"Every model here is within {spread:.1%} of `{SHARED}`'s parameter count, so this is a "
        "comparison at equal size.",
        SHARED_MARK,
    )

    out.append("")
    return "\n".join(out)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    written = []
    for corpus in CORPORA:
        if corpus == "synthetic":
            continue
        table = write(corpus)
        if table is None:
            continue
        path = os.path.join(OUT, f"{corpus}.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(table)
        written.append(path)
        print(f"  wrote {path}")

    if written:
        with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as handle:
            handle.write(
                "# Results\n\n"
                "Generated by `python tools/make_results.py`, which reads `runs/` and writes "
                "these tables.\n\n"
                "Each corpus is measured twice. Against `baseline`, the ordinary transformer in "
                "the corpus's published configuration: what a model gives up for its parameter "
                "saving. Against `shared_qkv`, the shared projection the ana_* operators are "
                "built on and match in size: what the operator recovers.\n\n"
                f"The two carry different marks, so a symbol names the test it came from: "
                f"`{BASELINE_MARK}` is p < 0.05 against `{BASELINE}`, `{SHARED_MARK}` is p < 0.05 "
                f"against `{SHARED}`.\n\n"
                + "\n".join(
                    f"- [{os.path.basename(p)[:-3]}]({os.path.basename(p)})" for p in written
                )
                + "\n"
            )
        print(f"  wrote {OUT}/README.md")


if __name__ == "__main__":
    main()
