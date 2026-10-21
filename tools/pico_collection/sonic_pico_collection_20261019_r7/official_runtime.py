"""Select the isolated dataset Python without modifying ROS/control dependencies."""
import os
from pathlib import Path
import subprocess
from delivery_paths import REPO


def python():
    path = Path(os.environ.get('SONIC_COLLECTION_PYTHON', REPO/'.runtime/collection-writer-v13/bin/python'))
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError('Official writer environment missing: run tools/gr00t_dataset/prepare_online_runtime.sh')
    return str(path)


def check_ros():
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    # Imports only: does not initialize rclpy or construct any ROS node.
    subprocess.run([python(), '-B', '-c',
        'import rclpy, communication.msg, sensor_msgs.msg, yaml, zmq; '
        'from official_session import OfficialSession; '
        'import official_writer; official_writer.dependencies()'], env=env, check=True, timeout=45)
