# Repository Execution Contract

## Instruction precedence

Follow the latest human instruction first, then the current orchestrator handoff, then the
approved issue specification and review feedback, and then applicable repository rules. Within
the repository-rule tier, a more specific nested `AGENTS.md` takes precedence over this root
file. Stop and report a conflict that cannot be resolved by this order.

## Approved scientific scope

- Implement and execute only the approved scientific specification. Do not extend a matrix,
  hypothesis, metric, dataset, seed set, tuning space, or interpretation without approval.
- Keep code readable and maintained. Use clear names, avoid speculative abstractions and
  unrelated refactors, and do not add an unapproved dependency.
- Preserve existing models, commands, artifacts, checkpoints, datasets, and historical results
  unless the approved work explicitly changes them.

## Scientific integrity

- Never fabricate, alter, omit, or relabel results. Preserve negative and inconclusive evidence.
- Record failed and interrupted runs. Never hide failures, select favorable seeds, tune on a test
  split, or retry a valid completed run because its result is unfavorable.
- Treat the approved data files, preprocessing, tokenizer, splits, and evaluation protocol as
  fixed. Load and evaluate only approved splits, and record every split that was accessed.
- Every scientific run must record the exact repository commit, study and model identifiers,
  full model and training configurations, dataset and tokenizer identities and hashes, seed,
  exact command, start and end records, exit state, selected checkpoint, metrics, parameter and
  cache accounting, prediction and checkpoint references, retry history, and any deviation.

## Artifacts and pull requests

- Keep compact, deterministic result artifacts suitable for review. Do not commit large
  checkpoints, logs, caches, environments, or prediction dumps.
- Use one issue per draft pull request. Keep implementation and scientific execution on the same
  approved branch and pull request, whose description contains exactly one
  `Fixes #<issue number>` statement.
- Return implementation and run work in separate handoffs. Implementation approval does not
  imply approval to run an experiment.

## Runtime execution

- Execute model, data-pipeline, training, evaluation, inference, benchmark, and scientific
  preflight code only on an approved server and only after a run handoff freezes an exact commit.
- The executor selects current operational resources, paths, process counts, and scheduling
  details without changing the approved scientific behavior.
- Executor result reports are factual provenance and measurements. Leave scientific
  interpretation and approval decisions to the orchestrator.

## Naming

Newly proposed models, methods, modules, objectives, procedures, datasets, and experiment
families must have plain descriptive full names. Do not introduce acronyms, initialisms,
abbreviations, backronyms, or branded short names for them. Derive code identifiers from the full
names with descriptive words.

## Implementation gate

During an implementation-only handoff, do not write, modify, or intentionally run tests. Do not
run unit or integration tests, smoke or dry runs, synthetic execution, model initialization,
dataset loading, training, evaluation, inference, benchmarking, reduced experiments, or any
model or data-pipeline execution.

Unless the approved issue narrows the gate further, implementation validation is limited to:

```text
ruff check .
ruff format --check .
python -m compileall -q src tools
```

Automatic continuous integration may run the existing suite. Do not disable, weaken, or modify
it; report its state only as repository evidence. Runtime correctness remains unproven until the
approved server preflight runs from the frozen commit.
