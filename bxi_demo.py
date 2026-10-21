"""Colleague C controller on the existing ROS CAN link. No body state requests."""
import ctypes as C
import _ctypes
from dataclasses import dataclass
import math
from pathlib import Path
from threading import RLock
import time
from types import SimpleNamespace

from .gripper import DMMotor, MotorFeedback
from .state import SonicTeleopState

GRIPPER_RELEASE = '20261014-demo-03-speed1400'
LIBRARY_PATH = Path(__file__).with_name('bxi_demo_native.so')
OPEN_RAD = math.radians(300.)
BYTES = C.c_uint8 * 8


class Native:
    def __init__(self):
        self.lib = C.CDLL(str(LIBRARY_PATH))
        if not hasattr(self.lib, 'demo_abi') or self.lib.demo_abi() != 1:
            self.close()
            raise RuntimeError('wrong colleague bridge ABI')
        self.lib.demo_new.argtypes, self.lib.demo_new.restype = [C.c_float], C.c_void_p
        self.lib.demo_free.argtypes, self.lib.demo_free.restype = [C.c_void_p], None
        self.lib.demo_step.argtypes = [C.c_void_p, C.c_float, C.c_float, C.c_int, C.POINTER(C.c_uint8)]
        self.lib.demo_step.restype = C.c_int
        self.lib.demo_feedback.argtypes = [C.POINTER(C.c_uint8), C.POINTER(C.c_float)]
        self.lib.demo_feedback.restype = C.c_int

    def close(self):
        _ctypes.dlclose(self.lib._handle)

    def new(self, position):
        handle = self.lib.demo_new(position)
        if not handle:
            raise RuntimeError('cannot allocate colleague reference')
        return handle

    def step(self, handle, target, dt, follow=True):
        data = BYTES()
        if self.lib.demo_step(handle, target, dt, follow, data):
            raise ValueError('invalid colleague position command')
        return bytes(data)

    def feedback(self, data, now):
        values = (C.c_float * 5)()
        if self.lib.demo_feedback(BYTES.from_buffer_copy(bytes(data)), values):
            raise ValueError('not a node 1 feedback frame')
        return MotorFeedback(1, None, *values, now)


@dataclass
class Side:
    phase: str = 'wait_tx'
    handle: object = None
    feedback: object = None
    enabled_at: object = None
    trigger: object = None
    trigger_at: object = None
    trigger_seen: bool = False
    target: float = 300.
    holding: bool = False
    tx_count: int = 0
    rx_count: int = 0
    last_rx: object = None
    last_tx: object = None
    last_report: float = -math.inf


class BxiDemoState(SonicTeleopState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if {self._left_bus, self._right_bus} != {5, 6}:
            raise ValueError('left/right must select distinct CAN5/CAN6 buses')
        self._demo_lock = RLock()
        self._native = None
        self._sides = {}

    def _demo_log(self, event, bus=None, **fields):
        try:
            self._motion_log('warning' if event == 'failure' else 'info', 'SONIC夹爪DEMO ',
                             dict(release=GRIPPER_RELEASE, event=event, bus=bus,
                                  side=None if bus is None else 'left' if bus == self._left_bus else 'right',
                                  **fields))
        except Exception:
            pass  # A failed log sink must not turn into a body control exception.

    def on_enter(self, ctx):
        with self._demo_lock:
            self._end_demo()
            # Retain body entry/reset; never enter the parent's DM discovery path.
            enabled, self.hardware_gripper = self.hardware_gripper, False
            try:
                super().on_enter(ctx)
            finally:
                self.hardware_gripper = enabled
            if not enabled:
                return
            self._gripper_session_active = True
            self._sides = {bus: Side() for bus in (5, 6)}
            self._gripper_profiles = {bus: SimpleNamespace(bus=bus, can_id=1,
                protocol=SimpleNamespace(value='bxi'), position_range=(-12.5,12.5),
                velocity_range=(-45.,45.), torque_range=(-40.,40.)) for bus in (5,6)}
            self._demo_last_tick = time.monotonic()
            self.logger.info('SONIC夹爪软件版本='+GRIPPER_RELEASE)
            self._demo_log('entry', kp=25., kd=.25, open_degrees=300., closed_degrees=0.,
                           left_bus=self._left_bus, right_bus=self._right_bus,
                           speed_degrees_s=1400., acceleration_degrees_s2=28000.,
                           exit_action='stop_sending_only', body_zero_torque=False)
            try:
                self._native = Native()
            except Exception as exc:
                self._demo_log('failure', reason='native_load: '+str(exc))
                self._native = None
            self._update_gripper(ctx, .005)

    def _send(self, bus, data):
        self._gripper_publisher.publish(DMMotor.build_motor_packet(bus, 1, list(data), frame_flags=5))
        self._sides[bus].tx_count += 1
        self._sides[bus].last_tx = bytes(data).hex()

    def _fail_side(self, bus, reason):
        side = self._sides[bus]
        if side.phase == 'fault':
            return
        side.phase = 'fault'
        self._gripper_motion_ready = self._gripper_calibrated = False
        self._demo_log('failure', bus, reason=reason, body_action='none',
                       last_rx=side.last_rx, last_tx=side.last_tx)
        if side.enabled_at is not None:
            try:
                self._send(bus, bytes([255]*7+[253]))  # This gripper only, once.
            except Exception as exc:
                self._demo_log('disable_send_failed', bus, reason=str(exc))

    def _gripper_feedback_callback(self, msg):
        with self._demo_lock:
            if not self._gripper_session_active or self._native is None:
                return
            bus = int(msg.bus)
            side = self._sides.get(bus)
            if side is None or side.phase == 'fault':
                return
            side.rx_count += 1
            side.last_rx = dict(can_id=int(msg.frame.can_id), flags=int(msg.frame.flags),
                                length=int(msg.frame.len), data=list(msg.frame.data[:8]))
            if side.rx_count <= 3:
                self._demo_log('raw_rx', bus, **side.last_rx)
            if side.phase == 'wait_tx':
                return
            if ((int(msg.frame.can_id), int(msg.frame.flags), int(msg.frame.len)) != (17, 5, 8)
                    or int(msg.frame.data[0]) & 15 != 1):
                return
            try:
                f = self._native.feedback(msg.frame.data[:8], time.monotonic())
                side.feedback = f
                if max(f.mos_temperature_c, f.motor_temperature_c) >= 80.:
                    self._fail_side(bus, 'temperature >=80C')
                elif side.phase == 'wait_feedback':
                    side.handle = self._native.new(f.position)
                    side.phase = 'opening'
                    self._demo_log('first_feedback', bus, position_degrees=math.degrees(f.position))
            except Exception as exc:
                self._fail_side(bus, 'feedback: '+str(exc))

    def _trigger(self, bus, value):
        with self._demo_lock:
            side = self._sides.get(bus)
            if self._gripper_session_active and side is not None and side.phase == 'control':
                side.trigger_seen = True
                side.trigger = self._valid_trigger(value)
                side.trigger_at = time.monotonic() if side.trigger is not None else None

    def _left_trigger_callback(self, msg):
        self._trigger(self._left_bus, msg.data)

    def _right_trigger_callback(self, msg):
        self._trigger(self._right_bus, msg.data)

    def _update_gripper(self, ctx_or_dt, dt=None):
        with self._demo_lock:
            if not self.hardware_gripper or not self._gripper_session_active or self._native is None:
                return
            now = time.monotonic()
            step = max(.001, min(.02, now-self._demo_last_tick))
            self._demo_last_tick = now
            cycle = dict(monotonic=now, feedback={}, inputs={}, sides={}, exiting=False,
                         pose_allowed=True, policy_status=getattr(self.policy, 'last_status', None))
            for bus, side in self._sides.items():
                try:
                    if side.phase == 'wait_tx':
                        if self._gripper_publisher is not None and self._gripper_publisher.get_subscription_count() > 0:
                            side.phase, side.enabled_at = 'wait_feedback', now
                            self._send(bus, bytes([255]*7+[252]))
                            self._gripper_armed = True
                            self._demo_log('enable_tx', bus, can_id=1, flags=5, data='FFFFFFFFFFFFFFFC',
                                           physical_delivery_verified=False)
                    f = side.feedback
                    if side.phase in ('opening', 'control'):
                        if not 0 <= now-f.received_at <= .3:
                            self._fail_side(bus, 'feedback stale >300ms')
                            continue
                        if side.phase == 'opening' and abs(math.degrees(f.position)-300.) <= 2.:
                            side.phase = 'control'
                            self._demo_log('open_ready', bus, position_degrees=math.degrees(f.position))
                        fresh = side.trigger_at is not None and 0 <= now-side.trigger_at <= .25
                        follow = side.phase == 'opening' or not side.trigger_seen or fresh
                        if side.phase == 'control' and fresh:
                            side.target = 300.*(1.-side.trigger)
                        if not follow and not side.holding:
                            self._native.lib.demo_free(side.handle)
                            side.handle = None
                            side.handle = self._native.new(f.position)
                        side.holding = not follow
                        data = self._native.step(side.handle, side.target, step, follow)
                        self._send(bus, data)
                        self._collection_call('sent', bus, data, self._gripper_profiles[bus])
                        label = 'left' if bus == self._left_bus else 'right'
                        cycle['feedback'][bus] = f
                        cycle['inputs'][label] = (side.trigger, side.trigger_at, fresh)
                        cycle['sides'][bus] = dict(active=side.phase=='control' and fresh,
                                                  mode='tracking' if follow else 'input_hold')
                    if now-side.last_report >= 1.:
                        side.last_report = now
                        self._demo_log('status', bus, phase=side.phase, target_degrees=side.target,
                            position_degrees=None if f is None else math.degrees(f.position),
                            feedback_age_s=None if f is None else now-f.received_at,
                            tx_count=side.tx_count, rx_count=side.rx_count,
                            last_tx=side.last_tx, trigger=side.trigger,
                            velocity_degrees_s=None if f is None else math.degrees(f.velocity),
                            temperatures_c=None if f is None else [f.mos_temperature_c,f.motor_temperature_c])
                except Exception as exc:
                    self._fail_side(bus, 'control: '+str(exc))
            self._gripper_motion_ready = self._gripper_calibrated = all(s.phase=='control' for s in self._sides.values())
            self._gripper_control_snapshot = cycle

    def _soft_limits_for(self, bus, profile):
        return OPEN_RAD, 0.

    def prepare_exit(self, ctx, to_state, *, trigger, force=False):
        return True

    def _disable_grippers(self, repeats=1):
        pass  # Normal exit/unbind intentionally sends neither close nor disable.

    def _fail_gripper_session(self, reason, ctx=None):
        # Also prevent accidental inherited DM calls from requesting a body state.
        self._demo_log('failure', reason=str(reason), body_action='none')

    def _end_demo(self):
        for side in self._sides.values():
            if side.handle is not None:
                self._native.lib.demo_free(side.handle)
        self._sides.clear()
        if self._native is not None:
            self._native.close()
            self._native = None
        self._gripper_session_active = self._gripper_armed = False
        self._gripper_motion_ready = self._gripper_calibrated = False

    def on_exit(self, ctx):
        with self._demo_lock:
            self._demo_log('exit', action='stop_sending_only')
            self._end_demo()
            super().on_exit(ctx)

    def on_unbind(self, ctx):
        with self._demo_lock:
            self._end_demo()
            super().on_unbind(ctx)
