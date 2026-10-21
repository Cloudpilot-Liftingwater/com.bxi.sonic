"""ELF3 ROS field adapter -> NVIDIA SONIC Gr00tDataExporter, directly online.

No JPEG queue/encoder or alternative Parquet/episode writer here. Validation and
source-clock sidecars remain shared with the legacy DM diagnostic reader.
"""
from pathlib import Path
import sys

import numpy as np

from .recorder import RealEpisodeRecorder

DATA_TOOLS = Path(__file__).resolve().parents[4] / 'gr00t_dataset'
sys.path.insert(0, str(DATA_TOOLS))
import official_writer
from contract import CAMERAS


class OfficialEpisodeRecorder(RealEpisodeRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.depth_enabled or set(self.rgb_camera_names) != set(CAMERAS.values()):
            raise ValueError('Official online ELF3 profile requires the existing three RGB cameras')
        self.queue = None  # Production never uses the legacy JPEG queue.
        self.exporter = None
        self._drained = False
        self._outcome = 'saved'

    def _start_writer(self):
        self._drained = False
        self._outcome = 'saved'
        sizes = {key: (self.camera_metadata[raw]['height'],
                       self.camera_metadata[raw]['width'], 3) for key, raw in CAMERAS.items()}
        self.exporter = official_writer.create(
            self.inprogress_dir / 'lerobot', self.meta['task_name'],
            camera_shapes=sizes, online=True)
        self.meta['storage_backend'] = 'nvidia_sonic_lerobot_v1'
        self.meta['lerobot_path'] = 'lerobot'
        self.meta['camera_contract']['rgb_storage'] = 'NVIDIA SONIC VideoWriter H264 MP4; one frame per actual inference row'
        self.meta['camera_contract']['encoder'] = {'codec': 'h264', 'preset': 'ultrafast', 'crf': '18', 'threads_per_camera': 1}
        # Base metadata setup creates historic empty JPEG directories; do not keep them.
        for name in self.rgb_camera_names:
            (self.inprogress_dir / name / 'rgb').rmdir()
            (self.inprogress_dir / name).rmdir()

    def _prepare_rgb(self, image):
        # Upstream add_frame deep-copies the whole frame before enqueueing.
        # ROS ImageSample arrays are never mutated; no second full RGB copy needed.
        return image

    def _submit_frame(self, row, rgb, depth):
        if depth is not None:
            raise ValueError('Depth is not part of the native66 RGB training contract')
        frame = {
            'observation.state': np.asarray(row['observation.state'][:29] + row['gripper']['state'], dtype=np.float32),
            'observation.projected_gravity': np.asarray(row['observation.projected_gravity'], dtype=np.float32),
            'action.motion_token': np.asarray(row['action.motion_token'], dtype=np.float32),
            'action.wbc': np.asarray(row['action'][:29] + row['gripper']['command'], dtype=np.float32),
        }
        images = {}
        for key, raw in CAMERAS.items():
            video_key = 'observation.images.' + key
            frame[video_key] = rgb[raw]
            images[raw] = {'path': 'lerobot/' + str(self.exporter.meta.get_video_file_path(0, video_key)),
                           'frame_index': row['idx']}
        self.exporter.add_frame(frame)  # Official per-camera workers + blocking queue.put.
        row['observation.images'] = images
        self.rows.append(row)  # Provenance only; numerical training rows belong to exporter.

    def writer_queue_size(self):
        return max((w.queue.qsize() for w in self.exporter.video_writers.values()), default=0) if self.exporter else 0

    def _drain_writer(self):
        if self._drained or self.exporter is None:
            return
        self._drained = True
        try:
            if self.frame_count:
                if self._outcome == 'saved':
                    self.exporter.save_episode()
                else:
                    # Official API retains the data AND writes discarded_episode_indices.
                    self.exporter.save_episode_as_discarded()
        except Exception as exc:
            self.errors.append('official_exporter: ' + str(exc))
        finally:
            try:
                official_writer.close(self.exporter)
            except Exception as exc:
                self.errors.append('official_video_writer: ' + str(exc))
        self.meta['official_writer_finalized'] = not bool(self.errors)

    def preserve_interrupted_episode(self, reason, *, min_frames, discarded=False):
        self._outcome = 'discarded' if discarded else 'interrupted'
        return super().preserve_interrupted_episode(reason, min_frames=min_frames, discarded=discarded)

    def discard_episode(self, reason):
        return self.preserve_interrupted_episode(reason, min_frames=0, discarded=True)
