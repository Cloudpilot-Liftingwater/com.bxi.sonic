#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
collection_source_environment "${COLLECTION_PROFILE:-hybrid_debug}"

if ! ros2 pkg executables usb_cam 2>/dev/null |
    grep -Eq '^usb_cam[[:space:]]+usb_cam_node_exe$'; then
  echo "FAIL: usb_cam_node_exe is not available in the active ROS environment." >&2
  echo "The project-local ROS dependency bundle may be missing." >&2
  exit 1
fi
echo "PASS: usb_cam=$(ros2 pkg prefix usb_cam)"

read_device() {
  python3 -c '
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
print(config["/**"]["ros__parameters"]["video_device"])
' "$1"
}

failed=0
for side in left right; do
  config="${COLLECTION_DIR}/config/${side}_wrist_usb.yaml"
  device="$(read_device "${config}")"
  if [[ ! -e "${device}" ]]; then
    echo "FAIL: ${side} wrist device does not exist: ${device}" >&2
    failed=1
  elif [[ ! -r "${device}" || ! -w "${device}" ]]; then
    echo "FAIL: ${side} wrist device is not readable/writable: ${device}" >&2
    failed=1
  else
    echo "PASS: ${side}_wrist=${device} -> $(readlink -f "${device}")"
  fi
done

realsense_usb=""
for sys_device in /sys/bus/usb/devices/*; do
  [[ -r "${sys_device}/idVendor" && -r "${sys_device}/idProduct" \
    && -r "${sys_device}/busnum" && -r "${sys_device}/devnum" ]] || continue
  [[ "$(<"${sys_device}/idVendor")" == "8086" ]] || continue
  [[ "$(<"${sys_device}/idProduct")" == "0b3a" ]] || continue
  printf -v realsense_usb '/dev/bus/usb/%03d/%03d' \
    "$((10#$(<"${sys_device}/busnum")))" "$((10#$(<"${sys_device}/devnum")))"
  break
done
if [[ -z "${realsense_usb}" ]]; then
  echo "FAIL: Intel RealSense USB device was not found." >&2
  failed=1
elif [[ ! -r "${realsense_usb}" || ! -w "${realsense_usb}" ]]; then
  echo "FAIL: RealSense USB device is not readable/writable: ${realsense_usb}" >&2
  failed=1
else
  echo "PASS: realsense_usb=${realsense_usb}"
fi

if ((failed)); then
  echo >&2
  echo "Install/reload access rules, reconnect the cameras, and retry:" >&2
  echo "  bash ${COLLECTION_DIR}/scripts/install_camera_access.sh" >&2
  exit 1
fi

echo "PASS: all physical camera dependencies and permissions are ready."
