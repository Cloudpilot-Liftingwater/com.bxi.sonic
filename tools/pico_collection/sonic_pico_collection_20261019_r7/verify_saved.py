#!/usr/bin/env python3
"""Streamed readback of all three JPEGs, raw signals and source-time alignment."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'collector'))
from real_robot_collection.wrist_diagnostic import SCHEMA, WRISTS, validate_raw_row
from real_robot_collection.contracts import validate_motion_token, sonic_projected_gravity
from real_robot_collection.gripper_history import blocked_by_intervals
from real_robot_collection.gripper_identity import validate_identities
from session_protocol import capture_segment, validate_pauses
from delivery_paths import site
from image_io import EpisodeImages, verify_official_table


def vector(value, size):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,) or not np.isfinite(arr).all():
        raise ValueError(f'invalid {size}-dimensional vector')
    return arr


def verify(episode, *, allow_incomplete=False, expected_ports=None):
    episode = Path(episode).resolve(strict=True)
    meta = json.loads((episode/'meta.json').read_text())
    if meta.get('schema_version') != SCHEMA or meta.get('training_export_allowed') is not False:
        raise ValueError('not a three-RGB DM/BXI v2 diagnostic source episode')
    latest_received = (meta.get('metadata', {}).get('sample_selection') == 'latest_received_v1'
                       and meta.get('metadata', {}).get('capture_profile') == 'sonic50_latest_rgb')
    latest_rgb = meta.get('metadata', {}).get('capture_profile') == 'sonic50_latest_rgb'
    method = 'sonic_inference_latest_received' if latest_received else 'sonic_inference_latest_rgb'
    if latest_rgb and (meta.get('fps') != 50 or meta.get('synchronization', {}).get('method')
                       != method):
        raise ValueError('inconsistent 50Hz/latest-RGB source profile')
    expected_grippers = validate_identities(meta.get('gripper_identity_contract'))
    if validate_identities(meta.get('metadata', {}).get('gripper_identities')) != expected_grippers:
        raise ValueError('gripper identity metadata mismatch')
    complete = meta.get('termination', {}).get('success') is True
    discarded = meta.get('discarded') is True
    if discarded and (complete or meta.get('excluded_from_training') is not True
                      or meta.get('termination', {}).get('writer_errors')):
        raise ValueError('invalid discard metadata or writer failure')
    if not complete and not (allow_incomplete and meta.get('incomplete') is True):
        raise ValueError('episode was not finalized successfully')
    ports = site()['wrist_usb_ports'] if expected_ports is None else expected_ports
    if meta.get('metadata', {}).get('physical_ports') != ports:
        raise ValueError('wrong physical camera roles')
    ingest = meta.get('synchronization', {}).get('gripper_ingest', {})
    pauses = meta.get('synchronization', {}).get('pose_pauses', [])
    validate_pauses(pauses, float('inf'))
    segments = []
    intervals = ingest.get('invalid_intervals', [])
    for barrier in intervals:
        if (not np.isfinite(barrier['start_unix']) or not np.isfinite(barrier['recover_after_unix'])
                or barrier['start_unix'] > barrier['recover_after_unix']):
            raise ValueError('invalid gripper interval metadata')
    stamps, deltas, ages = [], [], []
    source_ages = {}
    last_sequence = {n:-1 for n in WRISTS}
    last_camera_stamp = {n:0. for n in WRISTS}
    last_inference, count, previews = -1, 0, {}
    seen_paths = set()
    reused_counts = {n:0 for n in WRISTS}
    with EpisodeImages(episode) as image_reader, (episode/'frames.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['idx'] != count or row.get('diagnostic_only') is not True:
                raise ValueError('bad row index/diagnostic flag')
            if row.get('dataset.observation.state') is not None or row.get('dataset.action.wbc') is not None:
                raise ValueError('raw records must not populate legacy simulated-coordinate columns')
            aux, source = row['auxiliary'], row['source_timestamps_unix']
            anchor = float(aux['anchor_stamp_unix'])
            if not np.isfinite(anchor) or anchor <= 0 or stamps and anchor <= stamps[-1]:
                raise ValueError('non-increasing/invalid source anchor')
            if not np.isclose(row['t_episode'], anchor-(stamps[0] if stamps else anchor), atol=1e-7, rtol=0):
                raise ValueError('episode timing is not real source timing')
            segment = capture_segment(anchor, pauses)
            if segment is None or aux.get('capture_segment', 0) != segment:
                raise ValueError('row inside pause or wrong segment')
            if not latest_received and any(capture_segment(float(stamp), pauses) != segment for stamp in source.values()):
                raise ValueError('source sample crosses pause boundary')
            if not latest_received and 'smpl_ref' in aux.get('received_monotonic', {}):
                if capture_segment(aux['received_monotonic']['smpl_ref']+aux['clock_offset_s'], pauses) != segment:
                    raise ValueError('SMPL packet crosses pause boundary')
            segments.append(segment)
            stamps.append(anchor)
            validate_raw_row(row['gripper'], expected_grippers)
            validate_motion_token(row['action.motion_token'])
            inf = row['sonic.inference']
            if inf['sequence'] <= last_inference or inf.get('policy_status') != 'live_reference':
                raise ValueError('reused/non-live inference')
            last_inference = inf['sequence']
            if latest_rgb and abs(float(source['sonic_inference'])-anchor) > 1e-6:
                raise ValueError('row is not anchored to its actual inference timestamp')
            if inf['model_sha256'] != meta['metadata']['motion_token']['sha256']:
                raise ValueError('motion token model mismatch')
            target, action = vector(inf['target_position'],31), vector(row['action'],31)
            if not latest_received and not np.allclose(target[:29], action[:29], atol=1e-5, rtol=0):
                raise ValueError('token and final body action do not match')
            state = vector(row['observation.state'],31)
            if not np.asarray(row['observation.joint_present'], bool)[:29].all():
                raise ValueError('missing body joint feedback')
            imu = row['observation.imu']
            q = vector(imu['orientation_wxyz'],4)
            vector(imu['angular_velocity'],3)
            vector(imu['linear_acceleration'],3)
            gravity = vector(row['observation.projected_gravity'],3)
            if not np.allclose(gravity, sonic_projected_gravity(q,state[:29]), atol=1e-5, rtol=0):
                raise ValueError('projected gravity does not match saved IMU/body')
            if set(row['observation.images']) != set(WRISTS):
                raise ValueError('missing/extra RGB camera')
            for key in ('actuator_state','actuator_command','imu','sonic_inference','gripper_state','gripper_command'):
                if not np.isfinite(source[key]) or (not latest_received and abs(source[key]-anchor) > .030001):
                    raise ValueError('source alignment exceeds 30ms: '+key)
            raw = row['gripper']['native_raw_sample']
            if aux.get('gripper_ingest_method') == 'source_history_v2':
                if ingest.get('method') != 'source_history_v2' or 'invalid_intervals' not in ingest:
                    raise ValueError('missing gripper source-history audit')
                latency = aux['received_monotonic']['gripper']-float(raw['monotonic'])
                if (not np.isfinite(latency) or not 0 <= latency <= .75
                        or not np.isclose(aux['gripper_publication_to_callback_s'],latency,atol=1e-7,rtol=0)
                        or abs(source['gripper_state']-float(raw['monotonic'])-aux['clock_offset_s']) > .02):
                    raise ValueError('invalid gripper transport/clock audit')
            if not latest_received and blocked_by_intervals(source['gripper_state'], anchor, intervals):
                raise ValueError('row crosses invalid gripper source interval')
            for item in raw['samples']:
                for suffix, mono in (('feedback',item['feedback_monotonic']),('command',item['command']['sent_monotonic'])):
                    key = item['side']+'_gripper_'+suffix
                    expected = source['gripper_state']+(float(mono)-float(raw['monotonic']))
                    if not np.isfinite(source[key]) or (not latest_received and abs(source[key]-anchor) > .050001) or abs(source[key]-expected)>1e-6:
                        raise ValueError('invalid gripper feedback/command time: '+key)
            pico, smpl = row['pico.raw'], row['pico.smpl_ref']
            if not latest_received and int(np.asarray(pico['frame_index']).reshape(-1)[-1]) != int(np.asarray(smpl['frame_index']).reshape(-1)[-1]):
                raise ValueError('PICO/SMPL frame identity mismatch')
            pico_stamp = float(np.asarray(pico['timestamp_realtime']).reshape(-1)[-1])
            if source['pico'] != pico_stamp or (not latest_received and (not 0 <= anchor-pico_stamp <= .200001 or source['smpl_ref'] != pico_stamp)):
                raise ValueError('PICO source stamp/join mismatch')
            if latest_received:
                if aux.get('sample_selection') != 'latest_received_v1':
                    raise ValueError('missing latest-received provenance')
                selection = float(aux['selection_monotonic'])
                if not np.isfinite(selection):
                    raise ValueError('invalid selection time')
                for key, received in aux['received_monotonic'].items():
                    age = selection - float(received)
                    if not np.isfinite(age) or age < -1e-6:
                        raise ValueError('sample was not received at selection: '+key)
                    source_ages.setdefault(key, []).append(age)
                if abs(source['smpl_ref']-aux['received_monotonic']['smpl_ref']-aux['clock_offset_s']) > 1e-6:
                    raise ValueError('incorrect SMPL receipt-clock provenance')
            for name in WRISTS:
                source_stamp = float(source[name])
                delta = source_stamp-anchor
                sequence = aux['selected_camera_sequences'][name]
                if latest_rgb:
                    reused = sequence == last_sequence[name]
                    if (not np.isfinite(delta) or (not latest_received and not -.100001 <= delta <= 1e-7)
                            or sequence < last_sequence[name]
                            or source_stamp < last_camera_stamp[name]
                            or (reused and source_stamp != last_camera_stamp[name])
                            or (not reused and source_stamp <= last_camera_stamp[name])
                            or aux.get('camera_reused', {}).get(name) is not reused
                            or not np.isclose(aux.get('camera_source_age_s', {}).get(name, -1),
                                              -delta, atol=1e-7, rtol=0)):
                        raise ValueError('invalid latest camera stamp/sequence/reuse audit: '+name)
                    reused_counts[name] += int(reused)
                elif (not np.isfinite(delta) or abs(delta) > .025001
                      or source_stamp <= last_camera_stamp[name] or sequence <= last_sequence[name]):
                    raise ValueError('invalid/reused camera source stamp/sequence: '+name)
                last_camera_stamp[name] = source_stamp
                deltas.append(abs(delta))
                last_sequence[name] = sequence
                reference = row['observation.images'][name]
                name_path = reference['path'] if isinstance(reference, dict) else reference
                p = (episode/name_path).resolve()
                frame_key = (p, reference['frame_index']) if isinstance(reference, dict) else p
                if not p.is_relative_to(episode) or frame_key in seen_paths:
                    raise ValueError('outside/reused image frame')
                seen_paths.add(frame_key)
                if meta.get('storage_backend') == 'nvidia_sonic_lerobot_v1':
                    if not isinstance(reference, dict) or reference['frame_index'] != count:
                        raise ValueError('official video must match numerical row')
                image = image_reader.read(reference)
                camera = meta['camera_contract']['mounts'][name]
                if image is None or image.shape != (camera['height'],camera['width'],3):
                    raise ValueError('RGB/JPEG/video decode/size failed: '+str(p))
                age = aux['received_monotonic'][name]+aux['clock_offset_s']-source_stamp
                if not np.isfinite(age) or age < -.02:
                    raise ValueError('invalid source/receipt clock relation')
                ages.append(age)
                if count in (0,meta['num_frames']//2,meta['num_frames']-1):
                    previews[f'{name}_{count}'] = str(p)
            count += 1
        image_reader.verify_end()
    if meta.get('storage_backend') == 'nvidia_sonic_lerobot_v1' and count:
        if meta.get('official_writer_finalized') is not True:
            raise ValueError('Official writer did not finalize')
        verify_official_table(episode, meta)
    if count < (100 if complete else 0 if discarded else 1) or count != meta['num_frames']:
        raise ValueError('too few/mismatched saved rows')
    if discarded and count == 0:
        return dict(episode=str(episode), saved_rows=0, decoded_images=0,
                    readback='METADATA_ONLY', discarded=True, excluded_from_training=True,
                    episode_complete=False, timing_review='NOT_APPLICABLE',
                    training_export_allowed=False)
    gaps = np.diff(stamps)
    duration = stamps[-1]-stamps[0]
    rate = (count-1)/duration if duration > 0 else 0.
    max_gap = float(max(gaps)) if len(gaps) else None
    if not np.isclose(meta['duration_s'],duration,atol=1e-6,rtol=0):
        raise ValueError('metadata duration differs from source timestamps')
    sync = meta.get('synchronization', {})
    same_segment = np.diff(segments) == 0
    active_gaps = gaps[same_segment]
    active_rate = len(active_gaps)/sum(active_gaps) if len(active_gaps) else 0.
    active_max_gap = float(max(active_gaps)) if len(active_gaps) else None
    timing_ok = (complete and (45 <= active_rate <= 55 if latest_rgb else 25 <= active_rate <= 32) and active_max_gap is not None
                 and active_max_gap <= .2 and not sync.get('clock_jump', True))
    # Full invalid-interval audit stays in meta.json; keep terminal output compact.
    sync_display = dict(sync)
    if 'gripper_ingest' in sync:
        sync_display['gripper_ingest'] = {k:v for k,v in sync['gripper_ingest'].items()
                                         if k != 'invalid_intervals'}
    return dict(episode=str(episode), saved_rows=count, decoded_images=len(seen_paths),
        actual_row_rate_hz=rate, source_duration_s=duration, maximum_row_gap_s=max_gap,
        uninterrupted_row_rate_hz=active_rate, maximum_uninterrupted_gap_s=active_max_gap,
        camera_reused_rows=reused_counts, recorded_segments=len(set(segments)), pause_count=len(pauses),
        segment_boundaries=[i+1 for i, same in enumerate(same_segment) if not same],
        maximum_camera_to_anchor_delta_ms=1000*max(deltas),
        median_camera_header_to_receipt_ms=1000*float(np.median(ages)),
        source_receipt_age_ms={k:dict(max=1000*max(v), median=1000*float(np.median(v)))
                               for k,v in source_ages.items()},
        synchronization=sync_display, physical_ports=ports,
        readback='PASS', episode_complete=complete, gripper_identities=expected_grippers,
        discarded=discarded, excluded_from_training=discarded or not complete,
        timing_review=('PASS' if timing_ok else 'REVIEW') if complete else 'INCOMPLETE',
        training_export_allowed=False, preview_files=previews)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=Path)
    parser.add_argument('--allow-incomplete',action='store_true',help='Integrity audit only; never marks a partial trial complete.')
    args = parser.parse_args()
    try:
        result = verify(args.episode,allow_incomplete=args.allow_incomplete)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result['timing_review'] == 'PASS' else 2)
    except Exception as exc:
        print('FAIL: '+str(exc), file=sys.stderr)
        raise SystemExit(1)
