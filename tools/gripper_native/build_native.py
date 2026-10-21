"""Build a pure in-process bridge, without linking/running the PCI driver."""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = 'c08c4d000f01019bcd79c236ceacb3b6569572e9a0327bf4a5a3d1494f95d487'

def build(output):
    backend = ROOT/'colleague_original/backend'
    if hashlib.sha256((backend/'gripper_backend.c').read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('colleague backend differs from the reviewed source')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['cc', '-std=c11', '-O2', '-shared', '-fPIC', '-fvisibility=hidden',
        '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections', '-Wl,-z,defs',
        '-I'+str(backend), '-I'+str(backend/'bxi_pci_drv'),
        str(ROOT/'bxi_demo_bridge.c'), '-o', str(output), '-lm', '-pthread'], check=True)
    symbols = subprocess.check_output(['nm', '-D', str(output)], text=True)
    for forbidden in ('bxi_pci_', 'motor_pwr_', 'canfd_send_packet', 'main', 'pthread_create'):
        if forbidden in symbols:
            raise ValueError('unexpected hardware/process symbol: '+forbidden)
    print('NATIVE_BUILD_ONLY='+str(output))
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
