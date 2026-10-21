#!/usr/bin/env bash
set -e
WRIST_PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$WRIST_PACKAGE"
sha256sum --quiet -c SHA256SUMS
source "$WRIST_PACKAGE/environment.sh"
exec /usr/bin/python3 -B -u "$WRIST_PACKAGE/launch_collection.py" "$@"
