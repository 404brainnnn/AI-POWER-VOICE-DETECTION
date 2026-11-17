from __future__ import annotations

from .splits import group_split


def leave_one_group_out(records: list[dict], group_field: str) -> list[dict[str, list[dict]]]:
    groups = sorted({row.get(group_field) for row in records if row.get(group_field) is not None})
    folds=[]
    for held_out in groups:
        test=[row for row in records if row.get(group_field)==held_out]; train=[row for row in records if row.get(group_field)!=held_out]
        folds.append({"held_out_group": held_out, "train": train, "test": test})
    return folds


def speaker_disjoint(records: list[dict], seed: int = 42) -> dict[str, list[dict]]: return group_split(records, "speaker_id", seed)
def generator_disjoint(records: list[dict], seed: int = 42) -> dict[str, list[dict]]: return group_split(records, "generator_id", seed)
