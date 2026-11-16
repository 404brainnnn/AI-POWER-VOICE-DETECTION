from __future__ import annotations

from collections import defaultdict
import random


def group_split(records: list[dict], group_field: str, seed: int = 42, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> dict[str, list[dict]]:
    if abs(sum(fractions) - 1) > 1e-6 or not group_field: raise ValueError("fractions must sum to one")
    groups = defaultdict(list)
    for row in records: groups[row.get(group_field, row["sample_id"])].append(row)
    keys = list(groups); random.Random(seed).shuffle(keys)
    result = {"train": [], "validation": [], "test": []}
    targets = [fractions[0] * len(keys), (fractions[0] + fractions[1]) * len(keys)]
    for index, key in enumerate(keys): result["train" if index < targets[0] else "validation" if index < targets[1] else "test"].extend(groups[key])
    return result
