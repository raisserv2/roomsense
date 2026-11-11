"""Roll the serving model back to an earlier registry version.

Serving loads `models:/roomsense@champion`, so a rollback only moves the alias;
the API on nova notices within MODEL_POLL_SECONDS (60 s) and hot-reloads.

    python scripts/rollback.py --list        # versions, champion marked with *
    python scripts/rollback.py               # back to the version the champion replaced
    python scripts/rollback.py --to 2        # to a specific version
Needs MLFLOW_TRACKING_URI (https://nova.tailb5cb87.ts.net:5443 on the Legion).
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomsense import registry  # noqa: E402
from roomsense.runlog import setup_logging  # noqa: E402

log = setup_logging("rollback")


def main() -> int:
    ap = argparse.ArgumentParser(description="Move the champion alias back to an earlier model version.")
    ap.add_argument("--to", help="registry version to restore (default: the one the champion replaced)")
    ap.add_argument("--list", action="store_true", help="only list the versions")
    ap.add_argument("--reason", default="manual rollback")
    a = ap.parse_args()

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.error("MLFLOW_TRACKING_URI is not set")
        return 2
    from mlflow import MlflowClient
    params = (yaml.safe_load(Path("params.yaml").read_text()) or {}).get("register") or {}
    name, alias = params.get("model_name", "roomsense"), params.get("alias", "champion")
    client = MlflowClient()

    if a.list:
        for v in registry.list_versions(client, name, alias):
            f1 = "-" if v["test_macro_f1"] is None else f"{v['test_macro_f1']:.3f}"
            print(f"{'*' if v['is_champion'] else ' '} v{v['version']:<4} macro-F1 {f1}  {v['decision']}")
        return 0
    try:
        result = registry.rollback(client, name, alias, a.to, a.reason)
    except ValueError as e:
        log.error("%s", e)
        return 1
    log.info("champion is now v%s (rolled back from v%s)", result["champion"], result["rolled_back_from"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
