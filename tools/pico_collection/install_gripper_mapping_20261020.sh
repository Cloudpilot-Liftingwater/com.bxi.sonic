#!/usr/bin/env bash
# Workstation entry. Upload/configuration only; never starts/stops control or collection.
set -euo pipefail
MAPPING_ACTION=${1:---check}
[[ $# -le 1 && "$MAPPING_ACTION" =~ ^--(check|apply)$ ]] || { echo 'Use --check or --apply'; exit 2; }
[[ "$(hostname -s)" != elf3-test-01 ]] || { echo '请在工作站执行。'; exit 2; }
MAPPING_HOST=${ROBOT_MAPPING_HOST:-bxi@elf3-test-01.local}
[[ "$MAPPING_HOST" =~ ^bxi@[A-Za-z0-9.-]+$ ]] || exit 2
MAPPING_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/gripper_mapping_20261020" && pwd)
cd -- "$MAPPING_DIR"
sha256sum -c SHA256SUMS
MAPPING_SHA=$(sha256sum patch_mapping.py | cut -d ' ' -f 1)
echo "ROBOT_HOST=$MAPPING_HOST"
echo '仅改映射：左扳机/左侧数据=CAN5，右扳机/右侧数据=CAN6。'
echo 'apply前请保存采集、手动停止采集监听器，并正常结束自己的控制/遥操流程。'
MAPPING_STAGE=$(ssh -o StrictHostKeyChecking=yes -o ConnectTimeout=10 "$MAPPING_HOST" \
  'test "$(hostname -s)" = elf3-test-01 && mktemp -d /home/bxi/gripper_mapping_upload.XXXXXX')
[[ "$MAPPING_STAGE" =~ ^/home/bxi/gripper_mapping_upload\.[A-Za-z0-9]{6}$ ]] || exit 2
echo "REMOTE_UPLOAD_DIR=$MAPPING_STAGE"
scp -o StrictHostKeyChecking=yes "$MAPPING_DIR/patch_mapping.py" "$MAPPING_HOST:$MAPPING_STAGE/"
IFS= read -r -d '' MAPPING_REMOTE <<'REMOTE' || true
set -euo pipefail
test "$(hostname -s)" = elf3-test-01
test "$(id -un)" = bxi
cd "$1"
printf '%s  patch_mapping.py\n' "$3" | sha256sum -c -
python3 -B patch_mapping.py --check
if [[ "$2" == --apply ]]; then
  sudo python3 -B patch_mapping.py --apply
fi
REMOTE
printf -v MAPPING_QUOTED '%q' "$MAPPING_REMOTE"
ssh -t -o StrictHostKeyChecking=yes -o ConnectTimeout=10 "$MAPPING_HOST" \
  "bash -c $MAPPING_QUOTED gripper-mapping $MAPPING_STAGE $MAPPING_ACTION $MAPPING_SHA"
