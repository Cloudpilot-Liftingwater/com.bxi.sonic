#!/usr/bin/env bash
set -e
WRIST_PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$WRIST_PACKAGE/environment.sh"
exec bash "$WRIST_PACKAGE/collector/control.sh" "$@"
