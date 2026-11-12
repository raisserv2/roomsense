"""Model registry helpers: list versions, promote a champion, roll back.

The serving API always loads ``models:/<name>@<alias>`` (alias ``champion``).
Deploying a model is therefore nothing more than moving the alias, and a
rollback is moving it back. Every move is written to the version's tags, so
the registry itself is the deployment history.

All functions take an ``mlflow.MlflowClient`` (or anything with the same
methods), which keeps them testable without a tracking server.
"""
import time


def champion_version(client, name: str, alias: str = "champion"):
    """Version number (str) the alias points to, or None."""
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except Exception:  # noqa: BLE001  (no such model or alias yet)
        return None


def list_versions(client, name: str, alias: str = "champion") -> list:
    """All registry versions, newest first, as plain dicts."""
    champ = champion_version(client, name, alias)
    try:
        found = list(client.search_model_versions(f"name='{name}'"))
    except Exception:  # noqa: BLE001  (model not registered yet)
        return []
    out = []
    for mv in found:
        tags = dict(mv.tags or {})
        out.append({
            "version": str(mv.version),
            "run_id": mv.run_id or "",
            "created_at": (mv.creation_timestamp or 0) / 1000,
            "is_champion": str(mv.version) == champ,
            "test_macro_f1": _to_float(tags.get("test_macro_f1")),
            "decision": tags.get("decision", ""),
            "previous_champion": tags.get("previous_champion", ""),
            "rolled_back_at": _to_float(tags.get("rolled_back_at")),
            "git_commit": tags.get("git_commit", ""),
            "data_version": tags.get("data_version", ""),
            "data_include": tags.get("data_include", ""),
        })
    return sorted(out, key=lambda v: int(v["version"]), reverse=True)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def promote(client, name: str, version: str, alias: str = "champion", reason: str = "") -> dict:
    """Point the alias at ``version`` and remember which version it replaced."""
    previous = champion_version(client, name, alias)
    version = str(version)
    if previous and previous != version:
        client.set_model_version_tag(name, version, "previous_champion", previous)
    client.set_model_version_tag(name, version, "promoted_at", f"{time.time():.0f}")
    client.set_registered_model_alias(name, alias, version)
    return {"champion": version, "previous": previous, "reason": reason}


def rollback_target(client, name: str, alias: str = "champion"):
    """The version a rollback would restore: the one the champion replaced,
    else the newest older version that was once promoted."""
    current = champion_version(client, name, alias)
    if current is None:
        return None
    versions = {v["version"]: v for v in list_versions(client, name, alias)}
    previous = versions.get(current, {}).get("previous_champion")
    if previous and previous in versions:
        return previous
    older = [v for v in versions.values() if int(v["version"]) < int(current)]
    promoted = [v for v in older if v["decision"].startswith("promoted")]
    pool = promoted or older
    return max(pool, key=lambda v: int(v["version"]))["version"] if pool else None


def rollback(client, name: str, alias: str = "champion", to_version=None, reason: str = "manual") -> dict:
    """Move the alias back. Raises ValueError when there is nothing to roll back to."""
    current = champion_version(client, name, alias)
    if current is None:
        raise ValueError(f"Model '{name}' has no '{alias}' yet")
    target = str(to_version) if to_version else rollback_target(client, name, alias)
    if target is None:
        raise ValueError("No earlier version to roll back to")
    if target == current:
        raise ValueError(f"Version {target} is already the {alias}")
    known = {v["version"] for v in list_versions(client, name, alias)}
    if target not in known:
        raise ValueError(f"Version {target} does not exist")
    now = f"{time.time():.0f}"
    client.set_model_version_tag(name, current, "rolled_back_at", now)
    client.set_model_version_tag(name, current, "rollback_reason", reason[:200])
    client.set_model_version_tag(name, target, "restored_at", now)
    client.set_registered_model_alias(name, alias, target)
    return {"champion": target, "rolled_back_from": current, "reason": reason}


def seen_sessions(splits) -> set:
    """Sessions a model learned from: its train and validation splits.

    `splits` is the model's own splits.csv as a DataFrame (columns session_id, split).
    """
    return set(splits.loc[splits["split"].isin(["train", "val"]), "session_id"])


def fair_comparison_set(test, champion_seen: set) -> tuple:
    """Test rows the champion never learned from, plus a summary of what is left.

    The splitter reassigns sessions whenever the dataset grows, so sessions the
    champion trained on can land in the candidate's test split. Scoring the
    champion on those would test it on pictures it has already studied. The
    candidate never saw its own test sessions, so rows whose session the champion
    did not see are unseen by both models: the only fair ground for a comparison.
    """
    keep = ~test["session_id"].isin(champion_seen)
    rows = test[keep]
    return rows, {
        "test_frames": int(len(test)),
        "test_sessions": int(test["session_id"].nunique()),
        "compare_frames": int(len(rows)),
        "compare_sessions": int(rows["session_id"].nunique()),
        "excluded_frames": int((~keep).sum()),
        "compare_per_class": {k: int(v) for k, v in rows["label"].value_counts().sort_index().items()},
    }
