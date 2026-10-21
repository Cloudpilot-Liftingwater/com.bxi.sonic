"""Encode/save/discard a tiny real official dataset. No ROS or hardware startup."""
import importlib.metadata
import json
from pathlib import Path
import tempfile
import numpy as np
import official_writer
from contract import VECTOR_NAMES, CAMERAS


def check():
    source = json.loads(importlib.metadata.distribution('lerobot').read_text('direct_url.json') or '{}')
    if source.get('vcs_info', {}).get('commit_id') != 'a445d9c9da6bea99a8972daa4fe1fdd053d711d2':
        raise RuntimeError('Wrong LeRobot source revision; use prepare_online_runtime.sh')
    for name, version in [('torch','2.6.0'),('torchvision','0.21.0'),('numpy','1.26.4'),('datasets','3.6.0')]:
        if importlib.metadata.version(name).split('+')[0] != version:
            raise RuntimeError('Wrong writer dependency version: '+name)
    with tempfile.TemporaryDirectory(prefix='official-writer-check-') as temp:
        writer = official_writer.create(Path(temp)/'dataset', 'SYNTHETIC DEPENDENCY CHECK',
            camera_shapes={key:(32,32,3) for key in CAMERAS}, online=True)
        try:
            frame = {key:np.zeros(len(names),np.float32) for key,names in VECTOR_NAMES.items()}
            frame.update({'observation.images.'+key:np.zeros((32,32,3),np.uint8) for key in CAMERAS})
            for _ in range(3): writer.add_frame(frame)
            writer.save_episode_as_discarded()
        finally:
            official_writer.close(writer)
        import av
        for path in (Path(temp)/'dataset').rglob('*.mp4'):
            with av.open(str(path)) as stream:
                if sum(1 for _ in stream.decode(video=0)) != 3:
                    raise RuntimeError('Official video flush failed')
        if writer.meta.info['discarded_episode_indices'] != [0]:
            raise RuntimeError('Official discard did not persist')
    print('OFFICIAL_ONLINE_WRITER_DEPENDENCIES_PASS')


if __name__ == '__main__':
    check()
