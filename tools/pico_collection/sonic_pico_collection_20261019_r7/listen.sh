#!/usr/bin/env bash
set -e
PICO_COLLECTION_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$PICO_COLLECTION_ROOT"
sha256sum --quiet -c SHA256SUMS
source "$PICO_COLLECTION_ROOT/environment.sh"
# One dependency preflight per listener startup, not another Torch import per button.
/usr/bin/python3 -B -c 'from official_runtime import check_ros; check_ros()'
exec /usr/bin/python3 -B -u "$PICO_COLLECTION_ROOT/resident_launcher.py" "$@"
