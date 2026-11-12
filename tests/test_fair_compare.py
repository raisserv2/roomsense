"""The promotion gate compares models only on sessions neither of them learned from."""
import pandas as pd

from roomsense import registry
from scripts import backfill_splits


def frame(rows):
    return pd.DataFrame(rows, columns=["session_id", "label", "split"])


def test_seen_sessions_are_train_and_val_only():
    champ = frame([("a", "at_desk", "train"), ("b", "empty", "val"), ("c", "empty", "test")])
    assert registry.seen_sessions(champ) == {"a", "b"}


def test_sessions_the_champion_studied_are_excluded():
    test = frame([("a", "at_desk", "test"), ("a", "at_desk", "test"), ("c", "empty", "test"),
                  ("d", "lights_off", "test")])
    rows, info = registry.fair_comparison_set(test, champion_seen={"a", "b"})
    assert set(rows.session_id) == {"c", "d"}
    assert info == {"test_frames": 4, "test_sessions": 3, "compare_frames": 2, "compare_sessions": 2,
                    "excluded_frames": 2, "compare_per_class": {"empty": 1, "lights_off": 1}}


def test_nothing_excluded_when_the_champion_saw_none_of_it():
    test = frame([("x", "at_desk", "test"), ("y", "empty", "test")])
    rows, info = registry.fair_comparison_set(test, champion_seen={"a"})
    assert len(rows) == 2 and info["excluded_frames"] == 0


def test_backfill_finds_the_split_in_a_dvc_lock_and_the_cache(tmp_path):
    lock = {"stages": {"split": {"outs": [{"path": "data/processed/other.csv", "md5": "0" * 32},
                                          {"path": "data/processed/splits.csv", "md5": "ab" + "c" * 30}]}}}
    md5 = backfill_splits.splits_md5(lock)
    assert md5 == "ab" + "c" * 30
    assert backfill_splits.cache_path(md5, tmp_path) is None
    f = tmp_path / "files" / "md5" / "ab" / ("c" * 30)
    f.parent.mkdir(parents=True)
    f.write_text("session_id,split\n")
    assert backfill_splits.cache_path(md5, tmp_path) == f


def test_backfill_reports_a_lock_without_a_split():
    import pytest
    with pytest.raises(KeyError):
        backfill_splits.splits_md5({"stages": {"train": {}}})
