#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/scripts/common.sh"
PROFILE="${COLLECTION_PROFILE:-hybrid_debug}"

case "${PROFILE}" in
  sim|hybrid_debug|real) DEFAULT_CONFIG="$(collection_default_config "${PROFILE}")" ;;
  *)
    echo "COLLECTION_PROFILE must be sim, hybrid_debug, or real: ${PROFILE}" >&2
    exit 2
    ;;
esac
CONFIG="${REAL_COLLECTION_CONFIG:-${DEFAULT_CONFIG}}"
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="$(collection_absolute_path "${CONFIG}")"
fi

collection_source_environment "${PROFILE}"
collection_validate_config "${PROFILE}" "${CONFIG}"

if [[ "${PROFILE}" == "real" ]]; then
  echo "WARNING: real profile subscribes to live hardware commands and feedback." >&2
fi

cd "${SCRIPT_DIR}"
exec python3 -m real_robot_collection.node \
  --config "${CONFIG}" \
  --data-id "${DATA_ID:-}"
