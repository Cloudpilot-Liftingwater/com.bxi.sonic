"""SONIC/SMPLSim position metric definitions; strict, dependency-light adapters.

MPJPE_L removes translation only. Never remove heading or fit scale/rotation
under this name. Formula provenance and source fingerprints: provenance.json.
"""
import numpy as np

BODIES = ('torso_link','l_hip_x_link','l_knee_y_link','l_ankle_x_link',
          'r_hip_x_link','r_knee_y_link','r_ankle_x_link','waist_z_link',
          'l_shoulder_x_link','l_elbow_y_link','l_wrist_z_link',
          'r_shoulder_x_link','r_elbow_y_link','r_wrist_z_link')


def position_errors(predicted, reference, root_index=0):
    pred, ref = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    if pred.shape != ref.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError('positions must be matching [time, body, xyz] arrays in metres')
    if not 0 <= root_index < pred.shape[1]:raise ValueError('invalid root index')
    global_mm = np.linalg.norm(pred-ref, axis=-1)*1000.
    local_mm = np.linalg.norm((pred-pred[:, [root_index]])-(ref-ref[:, [root_index]]), axis=-1)*1000.
    return global_mm, local_mm


def summary(values, unit):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return dict(n=0,unit=unit,mean=None,rmse=None,p95=None,max=None)
    return dict(n=int(len(x)),unit=unit,mean=float(x.mean()),rmse=float(np.sqrt(np.mean(x*x))),
                p95=float(np.percentile(x,95)),max=float(x.max()))


def sixd_error_degrees(sixd):
    """Policy stores R[:, :2] flattened row-major. Error angle to identity.

    This is the already-yaw-aligned reference vs robot waist-anchor rotation,
    not an independent world-localization measurement.
    """
    x=np.asarray(sixd,dtype=float).reshape(-1,3,2)
    a,b=x[:,:,0],x[:,:,1]
    na,nb=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
    good=np.isfinite(x).all(axis=(1,2))&(np.abs(na-1)<.001)&(np.abs(nb-1)<.001)&(np.abs((a*b).sum(-1))<.001)
    out=np.full(len(x),np.nan)
    a=a[good]/na[good,None];b=b[good]-(a*b[good]).sum(-1)[:,None]*a
    b/=np.linalg.norm(b,axis=-1)[:,None]
    r=np.stack((a,b,np.cross(a,b)),axis=-1)
    out[good]=np.degrees(np.arccos(np.clip((np.trace(r,axis1=1,axis2=2)-1)/2,-1,1)))
    return out


def quaternion_error_degrees(predicted_xyzw, reference_xyzw):
    a,b=np.asarray(predicted_xyzw,dtype=float),np.asarray(reference_xyzw,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=4:raise ValueError('root quaternions must be [time,4], xyzw')
    na,nb=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1)
    good=np.isfinite(a).all(1)&np.isfinite(b).all(1)&(na>1e-8)&(nb>1e-8)
    out=np.full(len(a),np.nan)
    dot=np.abs((a[good]*b[good]).sum(1)/(na[good]*nb[good]))
    out[good]=np.degrees(2*np.arccos(np.clip(dot,0,1)))
    return out


def world_metrics(path):
    """Explicit, already synchronized same-robot/world data. No auto alignment."""
    with np.load(path,allow_pickle=False) as data:
        required=('time_s','predicted_world_m','reference_world_m','body_names',
            'coordinate_frame','reference_kind','actual_kind','synchronization',
            'robot_model_sha256','reference_robot_model_sha256')
        missing=[k for k in required if k not in data]
        if missing:raise ValueError('world data missing: '+', '.join(missing))
        text={k:str(data[k].item()) for k in required[4:]}
        if text['reference_kind']!='retargeted_robot':raise ValueError('reference must be independent retargeted_robot, not decoder or SMPL points')
        if text['actual_kind'] not in ('odometry_fk','external_mocap','simulator'):
            raise ValueError('actual_kind must identify measurement source')
        if any(not text[k].strip() for k in ('coordinate_frame','synchronization')):raise ValueError('world frame and time matching must be declared')
        h=text['robot_model_sha256']
        if len(h)!=64 or any(c not in '0123456789abcdef' for c in h) or h!=text['reference_robot_model_sha256']:
            raise ValueError('reference and measured robot model hashes must match')
        names=tuple(data['body_names'].tolist())
        if names!=BODIES:raise ValueError('world data body names/order differ from ELF3 evaluation contract')
        t=np.asarray(data['time_s'],dtype=float)
        pred=np.asarray(data['predicted_world_m'],dtype=float);ref=np.asarray(data['reference_world_m'],dtype=float)
        if t.ndim!=1 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('time_s must be finite and strictly increasing')
        if pred.shape!=(len(t),len(BODIES),3) or ref.shape!=pred.shape:raise ValueError('world positions shape mismatch')
        valid=np.isfinite(pred).all(axis=(1,2))&np.isfinite(ref).all(axis=(1,2))
        if 'valid' in data:
            v=np.asarray(data['valid'])
            if v.shape!=t.shape or v.dtype!=np.bool_:raise ValueError('valid must be boolean [time]')
            valid &= v
        g,l=position_errors(pred,ref)
        root=np.linalg.norm(pred[:,0]-ref[:,0],axis=-1)*1000.
        orientation=dict(n=0,unit='deg',mean=None,reason='both measured/reference root quaternions required')
        keys=('predicted_root_xyzw','reference_root_xyzw')
        if any(k in data for k in keys):
            if not all(k in data for k in keys):raise ValueError('need both root quaternion arrays')
            if data[keys[0]].shape!=(len(t),4) or data[keys[1]].shape!=(len(t),4):raise ValueError('root quaternion length mismatch')
            ori=quaternion_error_degrees(data[keys[0]],data[keys[1]])
            orientation=summary(ori[valid],'deg')
        return dict(status='computed' if valid.any() else 'no_valid_samples',
            source_contract=text,body_names=names,local_root='torso_link',received_frames=len(t),valid_frames=int(valid.sum()),
            mpjpe_g_mm=summary(g[valid],'mm'),mpjpe_l_mm=summary(l[valid],'mm'),
            root_position_error_mm=summary(root[valid],'mm'),root_orientation_error_deg=orientation,
            limitation='Coordinates, synchronization and calibration are caller-declared; no automatic frame fitting. Not an official benchmark success-rate result.')
