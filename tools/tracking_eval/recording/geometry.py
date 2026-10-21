"""Explicit geometric diagnostics, without equating SMPL rotvecs to robot DOFs."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT/'contract.json').read_text())
JOINTS = tuple(CONTRACT['joint_names'])
SMPL_BODY = ('left_hip','right_hip','spine1','left_knee','right_knee','spine2',
    'left_ankle','right_ankle','spine3','left_foot','right_foot','neck',
    'left_collar','right_collar','head','left_shoulder','right_shoulder',
    'left_elbow','right_elbow','left_wrist','right_wrist')
BENDS = ('left_elbow','right_elbow','left_knee','right_knee')
HUMAN_TRIPLES = ((16,18,20),(17,19,21),(1,4,7),(2,5,8))
ROBOT_TRIPLES = (
    ('l_shoulder_z_joint','l_elbow_y_joint','l_wrist_x_joint'),
    ('r_shoulder_z_joint','r_elbow_y_joint','r_wrist_x_joint'),
    ('l_hip_z_joint','l_knee_y_joint','l_ankle_y_joint'),
    ('r_hip_z_joint','r_knee_y_joint','r_ankle_y_joint'))


def bend(points):
    p = np.asarray(points, dtype=float)
    a, b = p[0]-p[1], p[2]-p[1]
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    if denominator < 1e-8 or not np.isfinite(p).all():return float('nan')
    return float(np.degrees(np.pi-np.arccos(np.clip(np.dot(a,b)/denominator,-1.,1.))))


def human_bends(positions):
    points = np.asarray(positions).reshape(24,3)
    return np.array([bend(points[list(indices)]) for indices in HUMAN_TRIPLES])


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    x,y,z = axis
    skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


class RobotGeometry:
    def __init__(self, path=ROOT/'elf3_reference.urdf'):
        self.joints = []
        for j in ET.parse(path).getroot().findall('joint'):
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
            rpy = np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
            transform = np.eye(4)
            transform[:3,:3] = rotation([0,0,1],rpy[2])@rotation([0,1,0],rpy[1])@rotation([1,0,0],rpy[0])
            transform[:3,3] = xyz
            axis = j.find('axis')
            self.joints.append((j.get('name'),j.get('type'),j.find('parent').get('link'),
                j.find('child').get('link'),transform,
                np.fromstring(axis.get('xyz','1 0 0') if axis is not None else '1 0 0',sep=' ')))
        parents = {j[2] for j in self.joints};children = {j[3] for j in self.joints}
        self.roots = parents-children

    def bends(self, q):
        values = dict(zip(JOINTS,np.asarray(q)))
        if not np.isfinite(q).all():return np.full(4,np.nan)
        links = {name:np.eye(4) for name in self.roots}
        points = {};pending = list(self.joints)
        while pending:
            remaining = []
            for name,kind,parent,child,origin,axis in pending:
                if parent not in links:remaining.append((name,kind,parent,child,origin,axis));continue
                transform = links[parent]@origin
                points[name] = transform[:3,3].copy()
                if kind in ('revolute','continuous'):
                    motion = np.eye(4);motion[:3,:3] = rotation(axis,values.get(name,0.))
                    transform = transform@motion
                elif kind == 'prismatic':
                    transform[:3,3] += transform[:3,:3]@axis*values.get(name,0.)
                elif kind != 'fixed':raise ValueError('unsupported URDF joint '+kind)
                links[child] = transform
            if len(remaining)==len(pending):raise ValueError('disconnected URDF joints')
            pending = remaining
        return np.array([bend([points[k] for k in names]) for names in ROBOT_TRIPLES])
