#!/bin/sh
# Nightly RoomSense dataset snapshot: version the frames with DVC,
# copy them to the Seagate remote, and push the receipt to GitHub.
set -e
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
cd "$HOME/roomsense"
dvc add data/raw -q
git add data/raw.dvc
if git diff --cached --quiet; then
  echo "No new frames since the last snapshot"
  exit 0
fi
git commit -q -m "dataset: nightly snapshot $(date +%F)"
dvc push -q
git push -q
echo "Snapshot $(date +%F) pushed"
