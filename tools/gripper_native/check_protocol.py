#!/usr/bin/env python3
"""Check the real shared library against official mm frames. No ROS/CAN/device access."""
import argparse
import ctypes as C
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def check(library):
    spec=importlib.util.spec_from_file_location('mm_protocol',ROOT/'bxi_gripper_protocol.py')
    protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
    cfg=protocol.settings();lib=C.CDLL(str(Path(library).resolve()))
    if lib.demo_abi()!=2:raise ValueError('requires mm bridge ABI2')
    arr=C.c_uint8*8
    lib.demo_pack.argtypes=[C.c_float]*7+[C.POINTER(C.c_uint8)]
    lib.demo_new.argtypes=[C.c_float]*7;lib.demo_new.restype=C.c_void_p
    lib.demo_step.argtypes=[C.c_void_p,C.c_float,C.c_float,C.c_int,C.POINTER(C.c_uint8)]
    lib.demo_free.argtypes=[C.c_void_p];lib.demo_free.restype=None
    data=arr()
    def pack(p,kp=2,kd=.05):
        if lib.demo_pack(p,0,kp,kd,0,cfg['velocity_limit_mm_s'],cfg['force_limit_n'],data):raise ValueError('codec failed')
        return bytes(data)
    assert pack(40)==bytes.fromhex('71c67ff6660cc7ff')
    assert pack(0)[:2]==b'\0\0'
    assert pack(70)[:2]==bytes.fromhex('c71b')
    assert pack(0,0,0)[3:7]==bytes.fromhex('f0000007')
    c=lib.demo_new(0,cfg['speed_mm_s'],cfg['acceleration_mm_s2'],cfg['kp'],cfg['kd'],cfg['velocity_limit_mm_s'],cfg['force_limit_n'])
    if not c:raise ValueError('invalid mm follower parameters')
    try:
        for target in (70,0,70):
            for _ in range(240):
                if lib.demo_step(c,target,.005,1,data):raise ValueError('follower failed')
                mm=((data[0]<<8)|data[1])*90/65535
                assert 0<=mm<=70.002
            assert abs(mm-target)<.02
    finally:lib.demo_free(c)
    print('BXI_MM_PROTOCOL_OFFLINE_PASS: ABI2, official golden frame, 0/70mm, STREAM reversal; hardware_commands=0')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,default=ROOT/'bxi_demo_native.so')
    check(p.parse_args().library)
