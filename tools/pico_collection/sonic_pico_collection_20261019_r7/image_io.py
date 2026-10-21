"""Read old JPEG recordings and official MP4 recordings without changing the data."""
from pathlib import Path
import cv2


class EpisodeImages:
    def __init__(self, episode):
        self.episode = Path(episode).resolve()
        self.videos = {}

    def read(self, reference):
        name = reference['path'] if isinstance(reference, dict) else reference
        path = (self.episode / name).resolve()
        if not path.is_relative_to(self.episode):
            raise ValueError('Image path outside episode')
        if not isinstance(reference, dict):
            return cv2.imread(str(path))
        index = reference['frame_index']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid video frame index')
        if path not in self.videos:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-50) > .001:
                cap.release()
                raise ValueError('Cannot read 50fps official video: ' + str(path))
            self.videos[path] = [cap, -1, None]
        item = self.videos[path]
        if index <= item[1]:
            raise ValueError('Repeated/backward video frame reference')
        while item[1] < index:
            ok, frame = item[0].read()
            if not ok:
                raise ValueError('Video ended before referenced frame: ' + str(path))
            item[1] += 1
            item[2] = frame
        return item[2]

    def verify_end(self):
        for path, (cap, _, _) in self.videos.items():
            if cap.read()[0]:
                raise ValueError('Video has extra frames beyond recorded rows: ' + str(path))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for cap, _, _ in self.videos.values():
            cap.release()


def verify_official_table(episode, meta):
    """Numerics and official outcome must agree with the ROS provenance sidecar."""
    import json
    import numpy as np
    import pyarrow.parquet as pq
    root = Path(episode) / 'lerobot'
    info = json.loads((root/'meta/info.json').read_text())
    discarded = info.get('discarded_episode_indices', [])
    if meta.get('discarded') or meta.get('incomplete'):
        if 0 not in discarded:
            raise ValueError('Official dataset lost the discarded/interrupted marker')
    elif discarded:
        raise ValueError('Saved episode unexpectedly marked discarded')
    if info['total_episodes'] != 1 or info['total_frames'] != meta['num_frames']:
        raise ValueError('Official dataset count mismatch')
    table = pq.read_table(root/'data/chunk-000/episode_000000.parquet').to_pydict()
    with (Path(episode)/'frames.jsonl').open() as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            expected = {
                'observation.state': row['observation.state'][:29]+row['gripper']['state'],
                'observation.projected_gravity': row['observation.projected_gravity'],
                'action.motion_token': row['action.motion_token'],
                'action.wbc': row['action'][:29]+row['gripper']['command'],
            }
            for key, value in expected.items():
                if not np.allclose(table[key][i], value, atol=1e-6, rtol=0):
                    raise ValueError(f'Official Parquet differs from source: {key}, row {i}')
            if table['frame_index'][i] != i or abs(table['timestamp'][i]-i/50) > 1e-4:
                raise ValueError('Official row time/index differs')
