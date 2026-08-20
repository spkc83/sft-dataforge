from __future__ import annotations

import pytest

from dataforge.curricula import Registry, compose


def _row(
    text: str, group_id: str, trajectory_id: str | None = None, pair_id: str | None = None
) -> dict:
    return {
        "text": text,
        "group_id": group_id,
        "trajectory_id": trajectory_id or group_id,
        "pair_id": pair_id,
        "example_kind": "unit",
    }


def test_dedup_prefers_eval_splits_by_default() -> None:
    registry = Registry()

    @registry.register("dupes", splits=("train", "validation", "test"))
    def dupes(split: str) -> list[dict]:
        # identical text in every split, distinct groups
        return [_row("duplicate text here", f"g-{split}")]

    splits, report = compose({"train": [], "validation": [], "test": []}, registry)
    # default priority is reversed(split_order) => test, validation, train: test wins
    assert len(splits["test"]) == 1
    assert len(splits["validation"]) == 0
    assert len(splits["train"]) == 0
    assert report["cross_split_duplicates_removed"] == 2


def test_dedup_priority_is_configurable() -> None:
    registry = Registry()

    @registry.register("dupes", splits=("train", "validation", "test"))
    def dupes(split: str) -> list[dict]:
        return [_row("duplicate text here", f"g-{split}")]

    splits, _ = compose(
        {"train": [], "validation": [], "test": []},
        registry,
        dedup_priority=("train", "validation", "test"),
    )
    assert len(splits["train"]) == 1
    assert len(splits["test"]) == 0


def test_group_leak_across_splits_raises() -> None:
    registry = Registry()

    @registry.register("bad", splits=("train", "test"))
    def bad(split: str) -> list[dict]:
        return [_row(f"text for {split}", "shared-group")]

    with pytest.raises(ValueError, match="shared-group"):
        compose({"train": [], "validation": [], "test": []}, registry)


def test_pair_leak_across_splits_raises() -> None:
    registry = Registry()

    @registry.register("bad-pair", splits=("train", "test"))
    def bad(split: str) -> list[dict]:
        return [_row(f"text for {split}", f"g-{split}", pair_id="shared-pair")]

    with pytest.raises(ValueError, match="shared-pair"):
        compose({"train": [], "validation": [], "test": []}, registry)


def test_report_counts_by_field() -> None:
    registry = Registry()

    @registry.register("counted", splits=("train",))
    def counted(split: str) -> list[dict]:
        return [_row("a", "g1"), _row("b", "g2")]

    _, report = compose(
        {"train": [], "validation": [], "test": []}, registry, count_fields=("example_kind",)
    )
    assert report["counts"]["example_kind"]["train"] == {"unit": 2}
    assert report["split_counts"]["train"] == 2


def test_held_out_texts_flagged_when_leaked_into_train() -> None:
    registry = Registry()

    @registry.register("leaky", splits=("train",))
    def leaky(split: str) -> list[dict]:
        return [_row("this is a held out phrase", "g1")]

    _, report = compose(
        {"train": [], "validation": [], "test": []},
        registry,
        held_out_texts=["this is a held out phrase"],
    )
    assert report["leakage"]["heldout_exact_leaks"]
