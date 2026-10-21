#!/usr/bin/env python3
"""Workstation-only readback of official online LeRobot files; never controls hardware."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from contract import CAMERAS, VECTOR_NAMES


def verify(path, session_path):
    path=Path(path); session_path=Path(session_path)
    result=json.loads((session_path/'session_result.json').read_text())
    header=json.loads((session_path/'capture.json').read_text())
    n=result['num_frames'];outcome=result['outcome'];events=result['events']
    if n==0:
        return dict(category='EMPTY_RECORDING',num_frames=0,readback='PASS',pause_resume_verified=False,
                    outcome=outcome,collector_release=header['collector_release'],events=events)
    root=path/'lerobot';info=json.loads((root/'meta/info.json').read_text())
    if info['script_config']['gripper_coordinate']!='absolute_motor_radians':raise ValueError('Wrong motor coordinate')
    metas=[json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines()]
    if len(metas)!=1 or metas[0]['length']!=n:raise ValueError('Episode metadata does not match saved rows')
    df=pd.read_parquet(next(root.glob('data/*/*.parquet')))
    if len(df)!=n:raise ValueError('Incomplete Parquet')
    for key,names in VECTOR_NAMES.items():
        value=np.stack(df[key])
        if value.shape!=(n,len(names)) or not np.isfinite(value).all():raise ValueError('Bad numeric column '+key)
    videos=list(root.rglob('*.mp4'))
    if len(videos)!=3:raise ValueError('Expected three videos')
    for video in videos:
        cap=cv2.VideoCapture(str(video))
        try:
            feature=info['features'][video.parent.name]
            count=0
            while True:
                ok,image=cap.read()
                if not ok:break
                if list(image.shape)!=feature['shape']:raise ValueError('Camera dimensions differ')
                count+=1
            if count!=n:raise ValueError('Incomplete video '+str(video))
        finally:cap.release()
    discarded=0 in info.get('discarded_episode_indices',[])
    if discarded!=(outcome!='saved'):raise ValueError('Operator result differs from official discard flag')
    source=[json.loads(line) for line in (path/'source.jsonl').read_text().splitlines()]
    if len(source)!=n:raise ValueError('Source receipt map incomplete')
    pairs=[];start=None
    for event in events:
        if event['event']=='pause':start=event
        elif start is not None:
            pairs.append((start,event));start=None
    resumed=any(0<a['row']<=b['row']<n for a,b in pairs)
    rates=[]
    stamps=[x.get('inference_stamp_unix') for x in source]
    if all(x is not None for x in stamps):
        gaps=np.diff(stamps);rates=[float(1/np.median(gaps[gaps>0]))] if np.any(gaps>0) else []
    return dict(category={'saved':'SAVED_PASS','discarded':'DISCARDED_EXCLUDED','interrupted':'INTERRUPTED_PRESERVED'}[outcome],
        num_frames=n,readback='PASS',decoded_video_frames=3*n,events=events,pause_resume_verified=resumed,
        outcome=outcome,collector_release=header['collector_release'],median_source_hz=rates[0] if rates else None,
        measured_motor_units='rad',commanded_motor_units='rad')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--session',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.episode,a.session),ensure_ascii=False))
