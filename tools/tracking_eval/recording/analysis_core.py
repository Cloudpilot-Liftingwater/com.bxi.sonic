"""Timestamp alignment and metrics. Missing samples are never replaced by zero."""
import json
import numpy as np


def packed(packet, topic):
    prefix = topic.encode()
    if not packet.startswith(prefix) or len(packet)<len(prefix)+1280:
        raise ValueError('short/bad packed message')
    packet = packet[len(prefix):]
    h = json.loads(packet[:1280].split(b'\0',1)[0])
    if h.get('endian','le')!='le':raise ValueError('unsupported endian')
    dtypes = dict(f32='<f4',f64='<f8',i32='<i4',i64='<i8',u8='u1',bool='?')
    offset = 1280;out = {}
    for f in h['fields']:
        shape = tuple(int(x) for x in f['shape'])
        if any(x<0 or x>100000 for x in shape):raise ValueError('bad shape')
        count = int(np.prod(shape)) if shape else 1
        dtype = np.dtype(dtypes[f['dtype']]);size = count*dtype.itemsize
        if count>1000000 or offset+size>len(packet):raise ValueError('bad payload size')
        if f['name'] in out:raise ValueError('duplicate field')
        out[f['name']] = np.frombuffer(packet,dtype=dtype,count=count,offset=offset).copy().reshape(shape)
        offset += size
    if offset!=len(packet):raise ValueError('trailing packed bytes')
    return out


def named(values, names, expected):
    if len(values)!=len(names) or len(set(names))!=len(names):raise ValueError('joint name/size mismatch')
    lookup = dict(zip(names,values))
    # Missing and nonfinite names remain explicit NaN, never default pose/zero.
    return np.array([lookup.get(k,np.nan) for k in expected],dtype=float)


def source_time(event):
    stamp = int(event['value']['stamp_ns'])
    now = int(event['ros_now_ns'])
    age = (now-stamp)*1e-9
    if stamp<=0 or not -.01<=age<=1.:
        raise ValueError('invalid/stale ROS source stamp')
    return float(event['received_monotonic'])-age


def causal_indices(times, queries, max_age=.05):
    times, queries = np.asarray(times),np.asarray(queries)
    if not len(times):return np.full(len(queries),-1,dtype=int)
    indices = np.searchsorted(times,queries,side='right')-1
    ages = queries-times[np.maximum(indices,0)]
    return np.where((indices>=0)&(ages>=-1e-9)&(ages<=max_age),indices,-1)


def metrics(error):
    x = np.asarray(error,dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):return dict(n=0,mae_deg=None,rmse_deg=None,bias_deg=None,p95_abs_deg=None,max_abs_deg=None)
    return dict(n=int(len(x)),mae_deg=float(np.mean(np.abs(x))),rmse_deg=float(np.sqrt(np.mean(x*x))),
        bias_deg=float(np.mean(x)),p95_abs_deg=float(np.percentile(np.abs(x),95)),max_abs_deg=float(np.max(np.abs(x))))


def estimate_lag(times, reference, measured, valid, *, max_lag=.25, step=.005, epochs=None):
    """Positive lag: measured(t) best matches reference(t-lag). Keep zero-lag metrics separately.

    All lag candidates use the same valid samples; no interpolation across gaps
    or inactive periods. This is a trajectory estimate, not clock/transport truth.
    """
    times = np.asarray(times);reference=np.asarray(reference);measured=np.asarray(measured)
    valid = np.asarray(valid)&np.isfinite(reference)&np.isfinite(measured)
    if valid.sum()<100:return dict(status='insufficient_samples',estimated_lag_ms=None)
    if np.ptp(reference[valid])<5.:
        return dict(status='insufficient_motion_under_5deg',estimated_lag_ms=None)
    dt = np.diff(times)
    breaks = (~valid).astype(int)
    breaks[1:] |= (dt<=0)|(dt>.05)|(~valid[:-1])
    if epochs is not None:
        breaks[1:] |= np.diff(np.asarray(epochs))!=0
    segment = np.cumsum(breaks)
    lags = np.arange(0,max_lag+step/2,step)
    indices = [causal_indices(times,times-lag,max_age=.03) for lag in lags]
    good = valid.copy()
    for ind in indices:
        good &= (ind>=0)&valid[np.maximum(ind,0)]&(segment==segment[np.maximum(ind,0)])
    if good.sum()<100 or np.ptp(times[good])<2.:
        return dict(status='insufficient_contiguous_motion',estimated_lag_ms=None)
    scores = [float(np.sqrt(np.mean((measured[good]-reference[ind[good]])**2))) for ind in indices]
    best = int(np.argmin(scores))
    return dict(status='search_boundary' if best==len(lags)-1 else 'trajectory_estimate',
        estimated_lag_ms=float(lags[best]*1000),lag_compensated_rmse_deg=scores[best],
        zero_lag_rmse_on_same_samples_deg=scores[0],samples=int(good.sum()),step_ms=step*1000,
        search_max_ms=max_lag*1000)
