"""Batch URDF FK in a fixed torso frame. This is an execution diagnostic."""
import numpy as np
from eval_metrics import BODIES


def body_positions(geometry, joint_names, angles, chunk_size=512):
    q=np.asarray(angles,dtype=float)
    if q.ndim!=2 or q.shape[1]!=len(joint_names):raise ValueError('joint angle shape mismatch')
    if geometry.roots!={'torso_link'}:raise ValueError('expected torso_link URDF root')
    index={name:i for i,name in enumerate(joint_names)}
    out=np.full((len(q),len(BODIES),3),np.nan)
    for start in range(0,len(q),chunk_size):
        current=q[start:start+chunk_size];n=len(current)
        good=np.isfinite(current).all(axis=1)
        links={'torso_link':np.tile(np.eye(4),(n,1,1))};pending=list(geometry.joints)
        while pending:
            remaining=[]
            for item in pending:
                name,kind,parent,child,origin,axis=item
                if parent not in links:remaining.append(item);continue
                transform=links[parent]@origin
                value=current[:,index[name]] if name in index else np.zeros(n)
                if kind in ('revolute','continuous'):
                    axis=axis/np.linalg.norm(axis);x,y,z=axis
                    skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
                    rot=np.eye(3)+np.sin(value)[:,None,None]*skew+(1-np.cos(value))[:,None,None]*(skew@skew)
                    transform[:,:3,:3]=transform[:,:3,:3]@rot
                elif kind=='prismatic':transform[:,:3,3]+=np.einsum('nij,j->ni',transform[:,:3,:3],axis)*value[:,None]
                elif kind!='fixed':raise ValueError('unsupported joint type '+kind)
                links[child]=transform
            if len(remaining)==len(pending):raise ValueError('disconnected URDF')
            pending=remaining
        positions=np.stack([links[name][:,:3,3] for name in BODIES],axis=1)
        positions[~good]=np.nan
        out[start:start+n]=positions
    return out
