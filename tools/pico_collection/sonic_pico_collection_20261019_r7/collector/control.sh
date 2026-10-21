#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/scripts/common.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  control.sh status|watch
  control.sh start [task_id]
  control.sh finish|abort|rearm
EOF
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
COMMAND="$1"
collection_source_environment "${COLLECTION_PROFILE:-}"

read_status_output() {
  ROS2CLI_NO_DAEMON=1 timeout "${STATUS_WAIT_SECONDS:-8}" \
    ros2 topic echo --once --full-length /elf3/recording_status
}

decode_status_output() {
  python3 -c '
import json, sys, yaml
try:
    document = yaml.safe_load(sys.stdin.read().split("\n---", 1)[0])
    print(json.dumps(json.loads(document["data"]), ensure_ascii=False, indent=2))
except Exception as exc:
    raise SystemExit(f"could not decode recording status: {exc}")
'
}

get_status_json() {
  local output
  if ! output="$(read_status_output)"; then
    echo "Could not read /elf3/recording_status; is start.sh still running with the same profile?" >&2
    return 1
  fi
  decode_status_output <<<"${output}"
}

case "${COMMAND}" in
  status)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    get_status_json
    exit 0
    ;;
  watch)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    exec ros2 topic echo --full-length /elf3/recording_status
    ;;
  start)
    ;;
  finish|abort|rearm)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    ;;
  *)
    echo "Unsupported command: ${COMMAND}" >&2
    usage
    exit 2
    ;;
esac

if [[ "${COMMAND}" == "finish" ]]; then
  STATUS_JSON="$(get_status_json)"
  STATUS_SUMMARY="$(python3 -c '
import json, sys
status = json.load(sys.stdin)
print(
    "true" if status.get("recording") is True else "false",
    status.get("phase", "unknown"),
    int(status.get("accepted_rows", 0)),
    int(status.get("minimum_episode_rows", 0)),
)
' <<<"${STATUS_JSON}")"
  read -r IS_RECORDING CURRENT_PHASE ACCEPTED_ROWS MINIMUM_ROWS <<<"${STATUS_SUMMARY}"
  if [[ "${IS_RECORDING}" != "true" ]]; then
    echo "REFUSED: no active recording (phase=${CURRENT_PHASE})." >&2
    echo "Rows are created only after missing is empty and a collection start request succeeds." >&2
    exit 2
  fi
  if (( MINIMUM_ROWS > 0 && ACCEPTED_ROWS < MINIMUM_ROWS )); then
    echo "REFUSED: episode has ${ACCEPTED_ROWS}/${MINIMUM_ROWS} synchronized rows." >&2
    echo "Recording is still active; wait until remaining_rows_to_finish reaches 0." >&2
    exit 2
  fi
fi

if [[ $# -eq 2 ]]; then
  ROS_MESSAGE="$(python3 -c '
import json, sys, yaml
payload = json.dumps({"command": sys.argv[1], "task_id": sys.argv[2]}, separators=(",", ":"))
print(yaml.safe_dump({"data": payload}, default_flow_style=True).strip())
' "${COMMAND}" "$2")"
else
  ROS_MESSAGE="$(python3 -c '
import json, sys, yaml
payload = json.dumps({"command": sys.argv[1]}, separators=(",", ":"))
print(yaml.safe_dump({"data": payload}, default_flow_style=True).strip())
' "${COMMAND}")"
fi
ros2 topic pub --once /elf3/recording_control std_msgs/msg/String "${ROS_MESSAGE}"
echo "Command '${COMMAND}' published. Run 'COLLECTION_PROFILE=${COLLECTION_PROFILE:-} bash control.sh status' to verify the result."
