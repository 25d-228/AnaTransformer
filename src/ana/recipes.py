"""What a recipe has to achieve, where the starting point comes from, and how one is stored.

The target comes first, because it is what a recipe is for. A recipe is good when it puts the
BASELINE on the number the literature reports for that corpus — not when it squeezes the most
out of the model, and certainly not when it makes our own models look best. Until the baseline
is on its number, nothing else in the study means anything: an undertrained baseline flatters
every model beneath it, and no amount of staring at the final table would reveal it.

Each corpus carries its own published recipe and its own target; see `ana/data/corpora.py`.
`ana tune` starts from the published learning rate, tries it against half and double, and
keeps whichever reaches the lowest development loss — a confirmation, not a fresh search.

Two disciplines hold this together.

The target is checked on a split that is not the outcome. On COGS the outcome is the
generalization split, so the target is the in-distribution split: it says the model learned
the task, and says nothing about the answer. Tuning until the split you are measuring looks
good is not tuning, it is fitting.

And the target is a check on the code, not something to chase. If the baseline will not reach
its number, the fault is in the model, the data or the step budget, and searching harder will
not find it — a wrong number looks exactly like a right one, which is precisely why the target
is written down before anything runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace

from ana.config import Schedule, Selection

RECIPE_DIR = "recipes"


@dataclass(frozen=True)
class Recipe:
    """How to train. Fixed within a corpus, free across corpora.

    Within a corpus, two models are only comparable if they were trained the same way; a
    different learning rate would throw away everything the parameter matching bought. Across
    corpora no number is ever compared with another, so each corpus gets whatever brings its
    own baseline to full strength.

    `verified` is False for the published starting point and True once `ana tune` has
    confirmed a rate against the development loss and written the file.
    """

    max_steps: int = 30_000
    batch_size: int = 128
    warmup_steps: int = 2_000
    learning_rate: float | None = None
    schedule: Schedule = Schedule.INVERSE_SQRT
    selection: Selection = Selection.BEST_DEV_LOSS
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.98)
    verified: bool = False
    source: str = "a published starting point, not a result. Run `ana tune` to confirm it."

    def __post_init__(self) -> None:
        # Enums survive a round trip through JSON as bare strings. Rebuild them here so that a
        # recipe loaded from disk behaves exactly like one built in memory, rather than failing
        # at the one `is` comparison somewhere deep in the trainer that nobody tested.
        object.__setattr__(self, "schedule", Schedule(self.schedule))
        object.__setattr__(self, "selection", Selection(self.selection))
        object.__setattr__(self, "adam_betas", tuple(self.adam_betas))


@dataclass(frozen=True)
class Calibration:
    """What the literature says a competent baseline reaches, and where that comes from.

    `split` is the split the gate is checked on, and it must not be the split the study is
    measuring. `also_report` names splits printed beside it for context but never gated on.
    """

    split: str
    low: float
    high: float
    reference: str

    def verdict(self, score: float) -> str:
        if score < self.low:
            return f"BELOW the published range [{self.low}, {self.high}] — suspect the code"
        if score > self.high:
            return f"above the published range [{self.low}, {self.high}] — check for leakage"
        return f"within the published range [{self.low}, {self.high}]"

    def passed(self, score: float) -> bool:
        return score >= self.low


def load(corpus: str, published: Recipe) -> Recipe:
    """The recipe `ana tune` confirmed, or the corpus's published starting point if not."""
    path = os.path.join(RECIPE_DIR, f"{corpus}.json")
    if not os.path.exists(path):
        return published

    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    known = {k: v for k, v in stored.items() if k in Recipe.__dataclass_fields__}
    return replace(published, **known)


def save(corpus: str, recipe: Recipe, search: list[dict]) -> str:
    """Write the confirmed recipe, and the search that confirmed it, so it has a history."""
    os.makedirs(RECIPE_DIR, exist_ok=True)
    path = os.path.join(RECIPE_DIR, f"{corpus}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({**asdict(recipe), "search": search}, handle, indent=2)
    return path
