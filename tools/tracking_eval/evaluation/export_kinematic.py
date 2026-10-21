"""Export the archived trained g1_kin only; workstation offline build utility."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ARCHIVE=Path('/data/ros2_ws/备份/SONIC_精灵3_step28800_训练配置与部署参考_20260923/资料')
POLICY=Path('/data/ros2_ws/zhou/bxi_vla_ws/bxi_rl_controller_ros2_example/src/bxi_example_py_elf3/mods/com.bxi.sonic/assets/sonic_action_token.onnx')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def export(out):
    import torch
    import onnx
    import onnxruntime as ort
    import yaml
    from omegaconf import OmegaConf
    from onnx import numpy_helper
    torch.set_num_threads(2)
    ckpt=ARCHIVE/'01_training_snapshot/model_step_028800.pt'
    config_path=ARCHIVE/'01_training_snapshot/config.yaml'
    source=ARCHIVE/'04_source_reference/gear_sonic/trl/modules/base_module.py'
    config=yaml.safe_load(config_path.read_text())
    cfg=config['algo']['config']['actor']['backbone']['decoders']['g1_kin']
    assert cfg['inputs']==['token']
    assert cfg['outputs']==['command_multi_future_nonflat','motion_anchor_ori_b_mf_nonflat']
    # Discard training bookkeeping using inert local types. No trainer imports,
    # checkpoint-defined callables, or weights_only=False deserialization.
    class MetadataOnly:
        def __init__(self,*args,**kwargs):pass
    def queue_metadata(*args):return list(args[0]) if args else []
    allowed=[]
    for name in torch.serialization.get_unsafe_globals_in_checkpoint(ckpt):
        if name=='collections.deque':allowed.append((queue_metadata,name))
        elif name.startswith(('trl.','transformers.','accelerate.')):allowed.append((MetadataOnly,name))
        else:raise ValueError('unexpected checkpoint global '+name)
    with torch.serialization.safe_globals(allowed):
        state=torch.load(ckpt,map_location='cpu',weights_only=True,mmap=True)['policy_state_dict']
    # Reuse the archived BaseModule class verbatim, without importing its unused
    # torchvision backends or starting an Isaac environment.
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseModule')
    namespace={'nn':torch.nn}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    module=namespace['BaseModule'](input_dim=32,output_dim=64,
        num_input_temporal_dims=2,num_output_temporal_dims=10,
        module_config_dict=OmegaConf.create(cfg['params']['module_config_dict']))
    prefix='actor_module.decoders.g1_kin.'
    module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
    module.eval()
    # Establish a strict policy-model allowlist from exact encoder/dyn weights.
    initializers={v.name:numpy_helper.to_array(v) for v in onnx.load(str(POLICY)).graph.initializer}
    matches=[]
    for name,a in initializers.items():
        if name.startswith('module.'):
            key='actor_module.'+name[len('module.'):]
            assert np.array_equal(a,state[key].numpy()),key
            matches.append(key)
    for i,n in enumerate(range(338,345)):
        key=f'actor_module.decoders.g1_dyn.module.{2*i}.weight'
        assert np.array_equal(initializers[f'onnx::MatMul_{n}'],state[key].numpy().T),key
        matches.append(key)
    assert len(matches)==24
    target=out/'g1_kin_step28800.onnx'
    generator=torch.Generator().manual_seed(20261015)
    sample=torch.randint(-16,16,(17,2,32),generator=generator).float()/16
    torch.onnx.export(module,sample[:1],str(target),input_names=['motion_token'],
        output_names=['motion_reconstruction'],opset_version=17,dynamo=False,
        dynamic_axes={'motion_token':{0:'batch'},'motion_reconstruction':{0:'batch'}})
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
    with torch.inference_mode():expected=module(sample).numpy()
    actual=session.run(None,{'motion_token':sample.numpy()})[0]
    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-5)
    np.savez_compressed(out/'tests/kinematic_golden.npz',token=sample.numpy(),raw=expected)
    cmd=config['manager_env']['commands']['motion']
    manifest=dict(schema=1,reference_kind='learned_kinematic_reconstruction',
        checkpoint_sha256=sha(ckpt),training_config_sha256=sha(config_path),
        base_module_sha256=sha(source),kinematic_onnx_sha256=sha(target),
        compatible_policy_sha256=[sha(POLICY)],matched_policy_parameters=matches,
        joint_names=config['elf3_controlled_joints'],units='rad',
        current_slot=0,robot_window_frames=10,robot_window_dt_s=cmd['dt_future_ref_frames'],
        smpl_window_dt_s=cmd['smpl_dt_future_ref_frames'],
        packing='raw[..., :58].reshape(batch,580): positions[:290], velocities[290:]; rotation=raw[...,58:64]',
        reference_angle_transform='absolute joint coordinates; no action scaling or default-pose addition',
        onnx_vs_archived_module_max_abs=float(np.max(np.abs(actual-expected))),
        validation='17 synthetic FSQ-grid tokens, not robot performance')
    (out/'kinematic_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('KINEMATIC_EXPORT='+str(target));print('MAX_ABS_ERROR='+str(manifest['onnx_vs_archived_module_max_abs']))

if __name__=='__main__':export(Path(__file__).resolve().parent)
