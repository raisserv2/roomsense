#!/bin/sh
# Nightly RoomSense dataset snapshot: version the frames with DVC,
# copy them to the Seagate remote, and push the receipt to GitHub.
#
# nova is the only machine that creates data snapshots. Other machines push
# pipeline results (dvc.lock, reports), so catch up with GitHub first;
# otherwise every nightly push is rejected as non-fast-forward.
# --autostash: a stray local edit on nova must not block the pull.
set -e
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd "$HOME/roomsense"
git pull --rebase --autostash -q || echo "WARNING: git pull failed, continuing with the local copy"
dvc add data/raw -q
git add data/raw.dvc
if git diff --cached --quiet; then
  echo "No new frames since the last snapshot"
  exit 0
fi
git commit -q -m "dataset: nightly snapshot $(date +%F)"
dvc push -q
# The Legion may have pushed while dvc push ran; catch up once more before pushing.
git pull --rebase --autostash -q || true
git push -q
echo "Snapshot $(date +%F) pushed"
