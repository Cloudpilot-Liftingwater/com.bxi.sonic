#!/usr/bin/env python3
"""Build and test the bundled usb_cam patch without opening devices or ROS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
source = root/'vendor/usb_cam-0.8.1'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True, help='New build directory outside the source repository')
p.add_argument('--baseline', type=Path, help='Optional previous library for reproducing the clock-step bug')
a = p.parse_args()
out = a.output.expanduser().absolute()
if out.is_relative_to(root.parents[1]):
    raise SystemExit('Build outside the source repository')
out.mkdir(parents=True, exist_ok=False)
flags = shlex.split(subprocess.check_output(['pkg-config','--cflags','libavcodec','libavutil','libswscale','opencv4'], text=True))
links = ['-lavcodec','-lavutil','-lswscale','-lopencv_imgproc','-lopencv_core']
subprocess.run(['g++','-std=c++17','-O2','-I',str(source/'include'),*flags,'-fPIC','-shared',
                '-Wl,-soname,libusb_cam.so','-Wl,--as-needed',str(source/'src/usb_cam.cpp'),
                '-o',str(out/'libusb_cam.so'),*links], check=True)
header = out/'test_include/usb_cam';header.mkdir(parents=True)
(header/'usb_cam.hpp').write_text((source/'include/usb_cam/usb_cam.hpp').read_text().replace('private:','public:'))
subprocess.run(['g++','-std=c++17','-O2','-I',str(header.parent),'-I',str(source/'include'),*flags,
                str(root/'test_clock_step.cpp'),'-L',str(out),'-lusb_cam',*links,
                '-Wl,--export-dynamic','-Wl,-rpath,$ORIGIN','-o',str(out/'test_clock_step')], check=True)
for lib, args in ([(a.baseline.parent, ['--expect-bug'])] if a.baseline else []) + [(out, [])]:
    subprocess.run([str(out/'test_clock_step'), *args], env=dict(os.environ, LD_LIBRARY_PATH=str(lib)), check=True)
print(json.dumps(dict(status='PASS', library=str(out/'libusb_cam.so'),
                     sha256=hashlib.sha256((out/'libusb_cam.so').read_bytes()).hexdigest(), hardware_commands=0)))
