"""Promotion and rollback only move the `champion` alias and leave a trail in the tags."""
import pytest

from roomsense import registry
from tests.conftest import FakeMlflow, FakeVersion


def make(champion=2):
    return FakeMlflow([FakeVersion(1, {"decision": "promoted: no champion yet", "test_macro_f1": "0.55"}),
                       FakeVersion(2, {"decision": "promoted: better", "test_macro_f1": "0.61"}),
                       FakeVersion(3, {"decision": "kept champion: worse", "test_macro_f1": "0.40"})], champion)


def test_list_versions_newest_first_with_champion_flag():
    v = registry.list_versions(make(), "roomsense")
    assert [x["version"] for x in v] == ["3", "2", "1"]
    assert [x["is_champion"] for x in v] == [False, True, False]
    assert v[1]["test_macro_f1"] == 0.61


def test_promote_remembers_the_replaced_version():
    c = make(champion=2)
    out = registry.promote(c, "roomsense", "3", reason="better")
    assert out == {"champion": "3", "previous": "2", "reason": "better"}
    assert c.aliases["champion"] == "3" and c.versions["3"].tags["previous_champion"] == "2"


def test_rollback_goes_to_the_version_the_champion_replaced():
    c = make(champion=1)
    registry.promote(c, "roomsense", "3")          # v3 replaced v1 (v2 was skipped)
    out = registry.rollback(c, "roomsense", reason="v3 is worse live")
    assert out["champion"] == "1" and out["rolled_back_from"] == "3"
    assert c.aliases["champion"] == "1" and "rolled_back_at" in c.versions["3"].tags


def test_rollback_without_history_picks_the_last_promoted_older_version():
    c = make(champion=3)                            # no previous_champion tag anywhere
    assert registry.rollback_target(c, "roomsense") == "2"


def test_rollback_to_an_explicit_version():
    c = make(champion=2)
    assert registry.rollback(c, "roomsense", to_version="1")["champion"] == "1"


@pytest.mark.parametrize("to_version, message", [("2", "already"), ("9", "does not exist")])
def test_rollback_refuses_bad_targets(to_version, message):
    with pytest.raises(ValueError, match=message):
        registry.rollback(make(champion=2), "roomsense", to_version=to_version)


def test_rollback_with_nothing_older():
    c = FakeMlflow([FakeVersion(1)], champion=1)
    with pytest.raises(ValueError, match="No earlier version"):
        registry.rollback(c, "roomsense")


def test_no_champion_yet():
    c = FakeMlflow([FakeVersion(1)])
    assert registry.champion_version(c, "roomsense") is None
    with pytest.raises(ValueError, match="no 'champion'"):
        registry.rollback(c, "roomsense")
