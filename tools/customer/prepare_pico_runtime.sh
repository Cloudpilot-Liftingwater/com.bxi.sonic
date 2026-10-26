#!/usr/bin/env bash
# Explicit dependency installation only; no ROS, XR service, camera or motor startup.
set -euo pipefail
# Keep this installer isolated from an activated Conda shell.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
export PATH=/usr/bin:/bin
PICO_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PICO_PLATFORM=$(/usr/bin/python3 -c 'import sysconfig; print(sysconfig.get_platform().lower())')
PICO_ENV=${1:-"$PICO_REPO/.runtime/$PICO_PLATFORM/pico"}
[[ "$PICO_ENV" = /* ]] || { echo 'Use an absolute environment path'; exit 2; }
if [[ -e "$PICO_ENV" && ! -f "$PICO_ENV/pyvenv.cfg" ]]; then
  echo "STOP: existing directory is not a venv: $PICO_ENV" >&2
  exit 2
fi
if [[ ! -x "$PICO_ENV/bin/python" ]]; then
  /usr/bin/python3 -m venv --without-pip "$PICO_ENV"
fi
/usr/bin/python3 -B - "$PICO_REPO" "$PICO_ENV/bin/python" <<'PYCHECK'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / 'pico'))
from python_runtime import _probe
ok, reason = _probe(Path(sys.argv[2]), ())
if not ok:
    raise SystemExit(reason)
PYCHECK
unset PYTHONPATH PYTHONHOME PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null
# Use the already bundled pip wheel; never install into system Python.
"$PICO_ENV/bin/python" -B - "$PICO_REPO/tools/gr00t_dataset/bootstrap/pip-26.1-py3-none-any.whl" <<'PYPIP'
import sys
sys.path.insert(0, sys.argv[1])
from pip._internal.cli.main import main
raise SystemExit(main(['install', '--no-index', '--no-deps', sys.argv[1]]))
PYPIP
"$PICO_ENV/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  --index-url https://pypi.org/simple \
  -r "$PICO_REPO/requirements-pico.txt" -c "$PICO_REPO/tools/customer/requirements-field.txt"
SONIC_PICO_PYTHON="$PICO_ENV/bin/python" /usr/bin/python3 -B - "$PICO_REPO" <<'PYPROBE'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / 'pico'))
from service_runtime import prepare_service_environment
from python_runtime import select_python
from manager_launcher import MANAGER_IMPORTS
prepare_service_environment()
selected = select_python('PICO dependency check (no services started)', MANAGER_IMPORTS)
print('PICO_VENV_READY=' + str(selected.executable))
PYPROBE
