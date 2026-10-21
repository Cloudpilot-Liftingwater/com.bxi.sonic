#!/usr/bin/env bash
# Collector-only upgrade. Never stops a service or starts robot control.
set -euo pipefail
PICO_ACTION=${1:---check}
[[ $# -le 1 && "$PICO_ACTION" =~ ^--(check|apply)$ ]] || { echo 'Use --check or --apply'; exit 2; }
[[ "$(hostname -s)" != elf3-test-01 ]] || { echo '请在工作站执行。'; exit 2; }
PICO_OUTPUTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PICO_RELEASE=sonic_pico_collection_20261019_r7
PICO_HOST=${PICO_COLLECTION_HOST:-bxi@192.168.88.45}
[[ "$PICO_HOST" =~ ^bxi@[a-zA-Z0-9.-]+$ ]] || exit 2
PICO_REPORT=$(mktemp -d /data/tmp/pico_collection_install.XXXXXX)
echo "REPORT_DIR=$PICO_REPORT"
cd -- "$PICO_OUTPUTS"
sha256sum -c "$PICO_RELEASE.tgz.sha256"
PICO_SHA=$(sha256sum "$PICO_RELEASE.tgz" | cut -d ' ' -f 1)
echo 'r7：A+X切出POSE暂停，切回继续录同一条；仅更新采集监听器。'
echo '不自动停服务，不动机器人/夹爪控制。安装后必须看到 PICO_COLLECTION_READY 才测试按键。'
PICO_STAGE=$(ssh -o StrictHostKeyChecking=yes -o ConnectTimeout=10 "$PICO_HOST" \
    'test "$(hostname -s)" = elf3-test-01 && mktemp -d /home/bxi/pico_button_upload.XXXXXX')
[[ "$PICO_STAGE" =~ ^/home/bxi/pico_button_upload\.[a-zA-Z0-9]{6}$ ]] || exit 2
echo "REMOTE_UPLOAD_DIR=$PICO_STAGE"
scp -o StrictHostKeyChecking=yes "$PICO_RELEASE.tgz" "$PICO_HOST:$PICO_STAGE/"
IFS= read -r -d '' PICO_REMOTE <<'REMOTE' || true
set -euo pipefail
diagnose_failure() {
    local result=$?
    trap - ERR
    echo 'INSTALL_FAILED: 不要继续按键测试；下面为只读服务诊断。'
    systemctl show elf3_pico_collection.service --property=ActiveState --property=SubState \
        --property=ExecStart --property=Result --property=ExecMainStatus --property=FragmentPath || true
    journalctl -u elf3_pico_collection.service -n 40 --no-pager || true
    exit "$result"
}
trap diagnose_failure ERR
test "$(hostname -s)" = elf3-test-01
test "$(id -un)" = bxi
systemctl show elf3_pico_collection.service --property=ActiveState --property=ExecStart || true
cd "$1"
printf '%s  %s\n' "$4" "$3.tgz" | sha256sum -c -
tar -xzf "$3.tgz" --no-same-owner
test ! -L "/home/bxi/$3"
if [ ! -e "/home/bxi/$3" ]; then mv -T -n "$3" "/home/bxi/$3"; fi
cd "/home/bxi/$3"
sha256sum --quiet -c SHA256SUMS
/usr/bin/python3 -B install_service.py --check
if [ "$2" = --apply ]; then
    sudo /usr/bin/python3 -B install_service.py --apply
    systemctl show elf3_pico_collection.service --property=ActiveState --property=MainPID --property=ExecStart
    echo '安装成功。录10秒→A+X暂停5秒→A+X恢复再录10秒→左握把+A保存；session始终应相同。'
fi
REMOTE
printf -v PICO_REMOTE_QUOTED '%q' "$PICO_REMOTE"
trap 'echo "REPORT_DIR=$PICO_REPORT"' EXIT
ssh -t -o StrictHostKeyChecking=yes -o ConnectTimeout=10 "$PICO_HOST" \
    "bash -c $PICO_REMOTE_QUOTED pico-install $PICO_STAGE $PICO_ACTION $PICO_RELEASE $PICO_SHA" \
    2>&1 | tee "$PICO_REPORT/install.log"
