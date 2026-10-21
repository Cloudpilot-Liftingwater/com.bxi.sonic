#!/usr/bin/env bash
set -e
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$repo/tools/customer/environment.sh"
exec /usr/bin/python3 -B -u "$repo/tools/tracking_eval/recording/record.py" "$@"
