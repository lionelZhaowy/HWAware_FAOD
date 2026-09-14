#!/usr/bin/env bash
# Run once with: sudo bash scripts/resume_dsec_migration.sh
# This script only migrates DSEC_DET; PEOD is untouched.
set -euo pipefail

if [[ "$EUID" -ne 0 ]]; then
    echo 'Run with sudo bash scripts/resume_dsec_migration.sh' >&2
    exit 1
fi

src=/srv/datasets/DSEC_DET
dst=/data/lab_dataset/RGB_DVS_DET/DSEC_DET
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
control="$project_root/logs/dsec_migration"
mkdir -p -- "$control"
exec 9>"$control/migration.lock"
flock -n 9 || { echo 'Another migration using this script is active.' >&2; exit 1; }
exec > >(tee -a "$control/resume.log") 2>&1
trap 'rc=$?; echo "Migration stopped: exit=$rc, line=$LINENO. Inspect source/target before restarting." >&2; exit "$rc"' ERR

date -Is
if [[ -L "$src" ]]; then
    [[ "$(readlink -f -- "$src")" == "$dst" && -d "$dst" ]]
    echo "Already migrated: $src -> $dst"
    exit 0
fi
[[ -d "$src" && ! -L "$dst" ]]
mkdir -p -- "$dst"
[[ "$(readlink -f -- "$src")" != "$(readlink -f -- "$dst")" ]]

echo 'Stage 1/3: resume copy; already matching files are skipped.'
rsync -aHAX --info=progress2 -- "$src/" "$dst/"

echo 'Stage 2/3: full checksum verification. This can take a long time without progress output.'
# --delete is ONLY used with --dry-run below; target files are never deleted.
rsync -aHAXnci --delete -- "$src/" "$dst/" > "$control/verification.diff"
if [[ -s "$control/verification.diff" ]]; then
    cat "$control/verification.diff"
    echo "Verification differences found. Source retained. Report: $control/verification.diff" >&2
    exit 1
fi

echo 'Stage 3/3: checksum matched; remove source tree and retain old path as a symlink.'
rm -rf --one-file-system -- "$src"
[[ ! -e "$src" && ! -L "$src" ]]
ln -s -- "$dst" "$src"
echo "Migration complete: $src -> $dst"
date -Is
df -h -- /srv/datasets "$dst"
