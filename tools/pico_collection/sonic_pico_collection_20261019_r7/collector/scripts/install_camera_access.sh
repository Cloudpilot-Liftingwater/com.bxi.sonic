#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTROLLER_ROOT="${BXI_CONTROLLER_ROOT:-$(cd "${COLLECTION_DIR}/.." && pwd)/bxi_rl_controller_ros2_example}"
DEPTH_CAMERA_INSTALLER="${CONTROLLER_ROOT}/src/bxi_depth_camera/tools/install_udev_rules.sh"
WRIST_RULE="${COLLECTION_DIR}/config/60-bxi-icspring-camera.rules"

[[ -x "${DEPTH_CAMERA_INSTALLER}" ]] || {
  echo "Missing camera rule installer: ${DEPTH_CAMERA_INSTALLER}" >&2
  exit 2
}
[[ -f "${WRIST_RULE}" ]] || {
  echo "Missing wrist camera rule: ${WRIST_RULE}" >&2
  exit 2
}

if ((EUID == 0)); then
  AS_ROOT=()
else
  command -v sudo >/dev/null 2>&1 || {
    echo "sudo is required to install camera udev rules." >&2
    exit 2
  }
  AS_ROOT=(sudo)
fi

"${AS_ROOT[@]}" bash "${DEPTH_CAMERA_INSTALLER}"
"${AS_ROOT[@]}" install -m 0644 "${WRIST_RULE}" \
  /etc/udev/rules.d/60-bxi-icspring-camera.rules
"${AS_ROOT[@]}" udevadm control --reload-rules
"${AS_ROOT[@]}" udevadm trigger --subsystem-match=video4linux --action=add
"${AS_ROOT[@]}" udevadm settle

echo
echo "Camera access rules installed. Unplug and reconnect all three cameras once."
echo "Then verify with: bash ${COLLECTION_DIR}/scripts/check_camera_access.sh"
