#!/usr/bin/env python3
"""Official SONIC cleanup/merge, with ELF3 task labels and gripper-mm v2 metadata."""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np

from contract import CONTRACT_ID, COORDINATE, POSITION_UNIT, ABI, CAMERAS, FPS, HORIZON, JOINTS, MODEL_SHA, SCHEMA, VECTOR_NAMES, modality
from vendor.sonic_data import process_dataset as official
from vendor.gr00t_stats import generate_stats


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def root_of(path):
    path = Path(path).resolve(strict=True)
    return path / 'lerobot' if (path / 'lerobot/meta/info.json').is_file() else path


def task_valid(task):
    return (isinstance(task, str) and len(task.strip()) >= 3
            and not task.lower().startswith(('diagnostic', 'replace_', 'unlabelled'))
            and task.strip().lower() not in ('test', '测试', 'todo'))


def export(episodes, output, task_override=None, *, inspection_only=False, include_discarded=False):
    if include_discarded and not inspection_only:
        raise ValueError("Discarded data may only be exported with inspection_only=True")
    roots = [root_of(p) for p in episodes]
    if any(not (p / 'meta/info.json').is_file() for p in roots):
        if any((p / 'meta/info.json').is_file() for p in roots):
            raise ValueError('Export legacy JPEG and online LeRobot sources separately')
        raise ValueError('Legacy JPEG/radian sources require the frozen v1 exporter; no implicit conversion to mm')
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    processed, tasks, source_map, cleanup = [], [], [], []
    reference_info = None
    for root in roots:
        # Require physical-unit ABI explicitly; v1 rad and v2 mm may never be merged.
        info = official.load_info(root)
        if (info.get('script_config', {}).get('gripper_coordinate') != COORDINATE
                or info.get('script_config', {}).get('gripper_position_unit') != POSITION_UNIT
                or info.get('script_config', {}).get('contract_id') != CONTRACT_ID
                or info.get('fps') != FPS):
            raise ValueError(f'Wrong coordinate/timebase: {root}')
        for key, names in VECTOR_NAMES.items():
            if info['features'][key]['shape'] != [len(names)] or info['features'][key]['names'] != names:
                raise ValueError(f'Joint/token layout differs: {root}: {key}')
        if reference_info is not None:
            for key in [*VECTOR_NAMES, *('observation.images.' + k for k in CAMERAS)]:
                if info['features'][key] != reference_info['features'][key]:
                    raise ValueError(f'Cannot merge different camera/data schemas: {key}')
        reference_info = reference_info or copy.deepcopy(info)
        stats, eps, _ = official.process_single_dataset(root, remove_stale_smpl=True, remove_discarded=not include_discarded)
        cleanup.append(dict(source=str(root), **stats))
        labels = {x['task_index']: x['task'] for x in official.load_tasks_meta(root)}
        for ep in eps:
            old_index = int(ep['episode_meta']['episode_index'])
            df = ep['df']
            remap = {}
            for old in df['task_index'].unique():
                task = task_override or labels[int(old)]
                if not inspection_only and not task_valid(task):
                    raise ValueError('Supply the actual instruction with --task (recording may be labelled afterward)')
                if task not in tasks:
                    tasks.append(task)
                remap[old] = tasks.index(task)
            df['task_index'] = df['task_index'].map(remap)
            ep['episode_meta'] = dict(ep['episode_meta'], tasks=[tasks[i] for i in sorted(set(remap.values()))])
            # Raw diagnostic SMPL drives the official cleanup, not a VLA modality.
            if 'teleop.smpl_pose' in df:
                df.drop(columns=['teleop.smpl_pose'], inplace=True)
            for key, names in VECTOR_NAMES.items():
                arr = np.stack(df[key]).astype(np.float32)
                if arr.shape != (len(df), len(names)) or not np.isfinite(arr).all():
                    raise ValueError(f'Invalid real numeric data: {root}/{old_index}/{key}')
            for key, path in ep['source_video_paths'].items():
                if not path.is_file():
                    raise FileNotFoundError(path)
            idx = ep['valid_indices']
            source_map.append(dict(source=str(root), source_episode=old_index,
                output_episode=len(processed), source_discarded=old_index in info.get('discarded_episode_indices', []), source_rows=None if idx is None else idx.tolist(),
                retained_rows=len(df), video_copy_without_reencoding=idx is None))
            processed.append(ep)
    if not processed:
        raise ValueError('No saved non-stale episodes; discarded data remains in the source')
    reference_info['features'].pop('teleop.smpl_pose', None)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.export-', dir=output.parent))
    try:
        count = official.write_output_dataset(staging, processed, reference_info,
            [dict(task_index=i, task=t) for i, t in enumerate(tasks)], reference_info.get('script_config'))
        official.copy_modality_json(roots, staging)
        # Upstream writer does not refresh these aggregate fields after merging.
        info = json.loads((staging / 'meta/info.json').read_text())
        info.update(total_tasks=len(tasks), total_videos=len(processed)*len(CAMERAS),
                    total_chunks=(len(processed)-1)//info.get('chunks_size',1000)+1,
                    splits={'train':f'0:{len(processed)}'})
        if include_discarded:
            info['discarded_episode_indices'] = [s['output_episode'] for s in source_map if s['source_discarded']]
            info['inspection_only'] = True
        dump(staging / 'meta/info.json', info)
        dump(staging / 'meta/modality.json', modality())
        generate_stats(staging)
        dump(staging / 'meta/elf3_contract.json', dict(
            schema=SCHEMA, joint_dimension=31, joint_names=JOINTS,
            state_model_dimension=34, action_model_dimension=66, action_horizon=HORIZON,
            embodiment_tag=ABI['embodiment_tag'], projector_slot=ABI['projector_slot'],
            fps=FPS, sonic_control_hz=50, tokenizer_sha256=MODEL_SHA,
            gripper_coordinate=COORDINATE, gripper_position_unit=POSITION_UNIT, contract_id=CONTRACT_ID,
            gripper_reference_mm={'closed':0,'open':70},
            image_transform='none_at_export; official_GR00T_Processor_at_training_and_inference',
            pause_policy='official_stale_SMPL_removed_same_episode', inspection_only=inspection_only,
            training_ready=not inspection_only, includes_discarded_for_inspection=include_discarded, output_time_basis='retained_row_index_divided_by_50',
            missing_source_rows_reconstructed=False, export_release='20261022-export-06-opening-mm'))
        dump(staging / 'meta/native_1d_derivation.json', dict(source_kind='real_robot_official_lerobot',
            gripper_coordinate=COORDINATE, unit=POSITION_UNIT, contract_id=CONTRACT_ID, simulated_coordinate_conversion=False))
        dump(staging / 'meta/source_map.json', dict(sequences=source_map, official_cleanup=cleanup))
        # Keep physical source clocks for inspection; they are not a VLA feature.
        for i, root in enumerate(roots):
            source = root.parent / 'source.jsonl'
            if source.is_file():
                import shutil
                dest=staging / 'provenance' / f'source_{i:04d}.jsonl'
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(source, dest)
        report = dict(status='INSPECTION_EXPORT_ONLY' if inspection_only else 'TRAINING_DATASET_READY',
            dataset=str(output), frames=count, training_sequences=len(processed), training_ready=not inspection_only,
            short_episode_indices=[i for i,e in enumerate(processed) if len(e['df'])<HORIZON],
            official_loader_run=False, includes_discarded_for_inspection=include_discarded, cleanup=cleanup, hardware_commands_sent=0)
        dump(staging/'EXPORT_RESULT.json', report)
        staging.rename(output)
        return report
    except BaseException as exc:
        dump(staging/'EXPORT_FAILED.json', dict(error=str(exc), training_ready=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',type=Path,nargs='*')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--task')
    parser.add_argument('--inspection-only',action='store_true')
    parser.add_argument('--include-discarded',action='store_true', help='Inspection exports only; keeps original discard flags')
    parser.add_argument('--check-dependencies',action='store_true')
    a=parser.parse_args()
    if a.check_dependencies:
        import official_writer
        official_writer.dependencies()
        print('OFFICIAL_CLEANUP_IMPORT_PASS')
        return
    if not a.episodes or a.output is None:parser.error('episodes and --output required')
    print(json.dumps(export(a.episodes,a.output,a.task,inspection_only=a.inspection_only,include_discarded=a.include_discarded),ensure_ascii=False))


if __name__=='__main__':main()
