"""Bounded official GR00T data-path checks; no models, training or hardware."""
import argparse
import json
from pathlib import Path
import sys


def run(report, tools):
    sys.path.insert(0,str(tools))
    import numpy as np
    from elf3_native_1d_config import elf3_native_1d_config as config
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    entries=json.loads((report/'exports.json').read_text())['entries']
    results=[]
    for entry in entries:
        if entry['export_status']!='EXPORTED_FOR_REVIEW':continue
        path=report/entry['export_path']
        result=dict(dataset=str(path),session=entry['session'],status='PENDING',finetuning_run=False,training_ready=False)
        try:
            loader=LeRobotEpisodeLoader(path,config,decoder_kwargs={'num_ffmpeg_threads':1})
            tag=EmbodimentTag.ELF3_NATIVE_1D
            processor=StateActionProcessor({tag.value:config},{tag.value:loader.get_dataset_statistics()},use_relative_action=False)
            samples=[]
            for ep in range(len(loader)):
                frame=loader._load_parquet_data(ep)
                length=len(frame)
                if length<40:raise ValueError(f'episode {ep} 只有{length}行，不足无padding的40步动作窗口；数据仍保留')
                # Full raw videos were read back by review.py. Here exercise the
                # training loader/40-step chunks at beginning, middle and end.
                starts=sorted({0,max(0,length//2-20),length-40})
                for start in starts:
                    batch=frame.iloc[start:start+40].reset_index(drop=True).copy()
                    videos=loader._load_video_data(ep,np.arange(start,start+40))
                    assert len(videos)==3
                    for key, images in videos.items():
                        assert len(images)==40
                        batch['video.'+key]=list(images)
                    step=extract_step_data(batch,0,config,tag,allow_padding=False)
                    assert sum(x.shape[-1] for x in step.states.values())==34
                    assert sum(x.shape[-1] for x in step.actions.values())==66
                    assert all(x.shape[0]==40 for x in step.actions.values())
                    states,actions=processor.apply(step.states,step.actions,tag.value)
                    assert all(np.isfinite(x).all() for x in [*states.values(),*actions.values()])
                    back_s,back_a=processor.unapply(states,actions,tag.value)
                    for key in ('left_gripper','right_gripper'):
                        np.testing.assert_allclose(back_s[key],step.states[key],atol=1e-5)
                        np.testing.assert_allclose(back_a[key],step.actions[key],atol=1e-5)
                    del batch,videos,step,states,actions
                samples.append(dict(episode=ep,rows=length,sampled_40_step_starts=starts))
            result.update(status='OFFICIAL_LOADER_SAMPLE_PASS',episodes=samples,state_shape=[1,34],action_shape=[40,66],gripper_unit='mm',
                          scope='Official parquet/video loader, first/middle/last 40-step windows, normalization round trip; not model finetuning')
        except Exception as exc:
            result.update(status='LOADER_CHECK_NOT_PASSED',error=str(exc))
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        (report/'loader_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    if not results:(report/'loader_results.json').write_text('[]\n')
    print('LOADER_RESULTS='+str(report/'loader_results.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    p.add_argument('--tools',type=Path,default=Path(__file__).resolve().parents[1]/'gr00t_dataset')
    a=p.parse_args()
    run(a.report.resolve(),a.tools.resolve())
