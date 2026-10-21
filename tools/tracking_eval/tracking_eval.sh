#!/usr/bin/env bash
# User-run workstation entry. Never stops/starts robot control or cameras.
set -euo pipefail
TRACKING_HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRACKING_RELEASE=sonic_tracking_20261019_v4
TRACKING_PYTHON=${SONIC_EVAL_PYTHON:-/usr/bin/python3}
export PYTHONDONTWRITEBYTECODE=1
TRACKING_ACTION=${1:---help}
case "$TRACKING_ACTION" in
  --analyze)
    [[ $# == 2 ]] || { echo '用法：--analyze 本地记录目录'; exit 2; }
    "$TRACKING_PYTHON" "$TRACKING_HERE/recording/analyze.py" "$2"
    exec "$TRACKING_PYTHON" "$TRACKING_HERE/evaluation/evaluate.py" "$2" ;;
  --check|--install|--remove) [[ $# == 1 ]] || exit 2 ;;
  --record)
    [[ $# -le 2 ]] || exit 2
    TRACKING_SECONDS=${2:-60}
    [[ "$TRACKING_SECONDS" =~ ^[0-9]{1,3}$ ]] || exit 2
    (( 10#$TRACKING_SECONDS >= 1 && 10#$TRACKING_SECONDS <= 600 )) || exit 2 ;;
  *)
    echo 'SONIC评测工具：tracking_eval.sh --check（只检查） | --install（安装观察点） | --record [秒，默认60] | --analyze 本地记录目录 | --remove（移除观察点）'
    exit 0 ;;
esac
[[ "$(hostname -s)" != elf3-test-01 ]] || { echo '请在工作站运行这个入口脚本。'; exit 2; }
TRACKING_HOST=${SONIC_EVAL_HOST:-bxi@192.168.88.149}
TRACKING_HOSTKEY=${SONIC_EVAL_HOSTKEY_ALIAS:-elf3-test-01.local}
[[ "$TRACKING_HOST" =~ ^bxi@[A-Za-z0-9.-]+$ && "$TRACKING_HOSTKEY" =~ ^[A-Za-z0-9.-]+$ ]] || exit 2
TRACKING_SSH=(-o StrictHostKeyChecking=yes -o "HostKeyAlias=$TRACKING_HOSTKEY")
echo "ROBOT_HOST=$TRACKING_HOST HOST_KEY_VERIFIED_AS=$TRACKING_HOSTKEY"
cd -- "$TRACKING_HERE/packages"
sha256sum -c "$TRACKING_RELEASE.tgz.sha256"
TRACKING_SHA=$(sha256sum "$TRACKING_RELEASE.tgz" | cut -d ' ' -f 1)
TRACKING_REPORT=$(mktemp -d /data/tmp/sonic_tracking_review.XXXXXX)
echo "REPORT_DIR=$TRACKING_REPORT"
if [[ "$TRACKING_ACTION" == --install || "$TRACKING_ACTION" == --remove ]]; then
  echo '仅安装/移除观察点。机器人空闲、正常结束你自己的控制和采集流程后执行；脚本不会停进程。'
else
  echo '只读观察；不依赖相机、不发电机命令、不改夹爪和身体控制参数。'
fi
TRACKING_STAGE=$(ssh "${TRACKING_SSH[@]}" -o ConnectTimeout=10 "$TRACKING_HOST" \
  'test "$(hostname -s)" = elf3-test-01 && mktemp -d /home/bxi/sonic_tracking_upload.XXXXXX')
[[ "$TRACKING_STAGE" =~ ^/home/bxi/sonic_tracking_upload\.[A-Za-z0-9]{6}$ ]] || exit 2
echo "REMOTE_UPLOAD_DIR=$TRACKING_STAGE"
scp "${TRACKING_SSH[@]}" "$TRACKING_RELEASE.tgz" "$TRACKING_HOST:$TRACKING_STAGE/"
IFS= read -r -d '' TRACKING_REMOTE <<'REMOTE' || true
set -euo pipefail
test "$(hostname -s)" = elf3-test-01
cd "$1"
printf '%s  %s\n' "$4" "$3.tgz" | sha256sum -c -
tar -xzf "$3.tgz" --no-same-owner
cd "$3"
sha256sum --quiet -c SHA256SUMS
if [[ "$2" == --record ]]; then
  /usr/bin/python3 -B install.py --require-installed
  # ROS setup reads optional unset variables; keep errexit/pipefail enabled.
  set +u
  source /home/bxi/sonic_gripper_fastfollow_20260917/environment.sh
  set -u
  mkdir -p /home/bxi/sonic_tracking_runs
  TRACKING_RUN=$(mktemp -d /home/bxi/sonic_tracking_runs/run.XXXXXX)
  echo "REMOTE_RUN_DIR=$TRACKING_RUN"
  set +e
  /usr/bin/python3 -B -u record.py --output "$TRACKING_RUN" --seconds "$5"
  TRACKING_RC=$?
  set -e
  echo "RECORD_EXIT=$TRACKING_RC"
  echo "REMOTE_RUN_DIR=$TRACKING_RUN"
  exit "$TRACKING_RC"
else
  TRACKING_EXTRA=()
  [[ "$2" != --remove ]] || TRACKING_EXTRA=(--remove)
  /usr/bin/python3 -B install.py "${TRACKING_EXTRA[@]}"
  if [[ "$2" != --check ]]; then
    sudo /usr/bin/python3 -B install.py --apply "${TRACKING_EXTRA[@]}"
  fi
fi
REMOTE
printf -v TRACKING_QUOTED '%q' "$TRACKING_REMOTE"
TRACKING_RC=0
ssh -t "${TRACKING_SSH[@]}" -o ConnectTimeout=10 -o ServerAliveInterval=10 \
  "$TRACKING_HOST" \
  "bash -c $TRACKING_QUOTED sonic-tracking $TRACKING_STAGE $TRACKING_ACTION $TRACKING_RELEASE $TRACKING_SHA ${TRACKING_SECONDS:-60}" \
  2>&1 | tee "$TRACKING_REPORT/operation.log" || TRACKING_RC=$?
if [[ "$TRACKING_ACTION" == --record ]]; then
  TRACKING_RUN=$("$TRACKING_PYTHON" - "$TRACKING_REPORT/operation.log" <<'PY'
import pathlib,re,sys
text=pathlib.Path(sys.argv[1]).read_text()
paths=re.findall(r'^REMOTE_RUN_DIR=(/home/bxi/sonic_tracking_runs/run\.[A-Za-z0-9]{6})\r?$',text,re.M)
if paths:print(paths[-1])
PY
)
  [[ -n "$TRACKING_RUN" ]] || { echo "记录未启动；查看上面的检查错误。REPORT_DIR=$TRACKING_REPORT"; exit 3; }
  scp -r "${TRACKING_SSH[@]}" "$TRACKING_HOST:$TRACKING_RUN" "$TRACKING_REPORT/"
  "$TRACKING_PYTHON" "$TRACKING_HERE/recording/analyze.py" "$TRACKING_REPORT/$(basename "$TRACKING_RUN")"
  "$TRACKING_PYTHON" "$TRACKING_HERE/evaluation/evaluate.py" "$TRACKING_REPORT/$(basename "$TRACKING_RUN")"
  echo "HTML_REPORT=$TRACKING_REPORT/$(basename "$TRACKING_RUN")/analysis/tracking_reference.html"
fi
echo "REPORT_DIR=$TRACKING_REPORT"
exit "$TRACKING_RC"
