#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraMode:
    fourcc: str
    width: int
    height: int
    fps: float

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def __str__(self) -> str:
        return (
            f"{self.fourcc} | "
            f"{self.width}x{self.height} | "
            f"{self.fps:.2f} FPS"
        )


def get_camera_modes(device: str) -> list[CameraMode]:
    """
    使用 v4l2-ctl 获取摄像头支持的离散分辨率和帧率。
    """
    command = [
        "v4l2-ctl",
        "--device",
        device,
        "--list-formats-ext",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print(
            "错误：没有找到 v4l2-ctl，请执行：\n"
            "sudo apt install v4l-utils",
            file=sys.stderr,
        )
        return []
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if "Permission denied" in stderr or "Failed to open" in stderr:
            print(
                f"读取摄像头能力失败：{stderr}\n{format_device_permission_hint(device)}",
                file=sys.stderr,
            )
        else:
            print(
                f"读取摄像头能力失败：{stderr}",
                file=sys.stderr,
            )
        return []

    modes: list[CameraMode] = []

    current_fourcc: str | None = None
    current_width: int | None = None
    current_height: int | None = None

    for line in result.stdout.splitlines():
        # 示例：
        # [0]: 'MJPG' (Motion-JPEG, compressed)
        format_match = re.search(r"\[\d+\]:\s+'([^']+)'", line)
        if format_match:
            current_fourcc = format_match.group(1)
            current_width = None
            current_height = None
            continue

        # 示例：
        # Size: Discrete 1920x1080
        size_match = re.search(
            r"Size:\s+Discrete\s+(\d+)x(\d+)",
            line,
        )
        if size_match:
            current_width = int(size_match.group(1))
            current_height = int(size_match.group(2))
            continue

        # 示例：
        # Interval: Discrete 0.033s (30.000 fps)
        fps_match = re.search(r"\(([\d.]+)\s+fps\)", line)
        if (
            fps_match
            and current_fourcc is not None
            and current_width is not None
            and current_height is not None
        ):
            modes.append(
                CameraMode(
                    fourcc=current_fourcc,
                    width=current_width,
                    height=current_height,
                    fps=float(fps_match.group(1)),
                )
            )

    # 去除重复模式
    return list(dict.fromkeys(modes))


def format_preference(fourcc: str) -> int:
    """
    在其他条件相同时优先使用压缩格式。
    MJPG 通常比 YUYV 更容易实现高分辨率和高帧率。
    """
    fourcc = fourcc.upper()

    if fourcc in {"MJPG", "JPEG"}:
        return 3
    if fourcc in {"H264", "HEVC"}:
        return 2
    if fourcc in {"YUYV", "YUY2"}:
        return 1

    return 0


def select_mode(
    modes: list[CameraMode],
    priority: str,
) -> CameraMode:
    """
    resolution:
        优先最大分辨率，再选择该分辨率下的最高帧率。

    fps:
        优先最高帧率，再选择该帧率下的最大分辨率。

    bandwidth:
        按每秒像素量 width * height * fps 排序。
    """
    if priority == "resolution":
        return max(
            modes,
            key=lambda mode: (
                mode.pixels,
                mode.fps,
                format_preference(mode.fourcc),
            ),
        )

    if priority == "fps":
        return max(
            modes,
            key=lambda mode: (
                mode.fps,
                mode.pixels,
                format_preference(mode.fourcc),
            ),
        )

    return max(
        modes,
        key=lambda mode: (
            mode.pixels * mode.fps,
            mode.pixels,
            mode.fps,
            format_preference(mode.fourcc),
        ),
    )


def fourcc_to_string(value: float) -> str:
    value = int(value)

    return "".join(
        chr((value >> (8 * index)) & 0xFF)
        for index in range(4)
    ).strip("\x00")


def has_gui_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def format_device_permission_hint(device: str) -> str:
    return (
        f"没有权限访问 {device}。\n"
        "请检查：\n"
        "1. 当前用户是否在 video 组中；\n"
        f"2. 设备权限是否允许读写：ls -l {device}；\n"
        "3. 是否需要用 sudo 或调整 udev 规则；\n"
        "4. 远程服务器上该设备是否真的已透传/挂载到当前环境。"
    )


def print_modes(modes: list[CameraMode]) -> None:
    print("\n摄像头支持的离散模式：")

    sorted_modes = sorted(
        modes,
        key=lambda mode: (
            mode.pixels,
            mode.fps,
            format_preference(mode.fourcc),
        ),
        reverse=True,
    )

    for index, mode in enumerate(sorted_modes, start=1):
        print(f"{index:3d}. {mode}")


def open_camera(device: str, mode: CameraMode) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(format_device_permission_hint(device))

    # 设置较小的缓存，减少实时画面的延迟。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 编码格式应尽量在分辨率和帧率之前设置。
    fourcc = cv2.VideoWriter_fourcc(*mode.fourcc[:4])
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
    cap.set(cv2.CAP_PROP_FPS, mode.fps)

    return cap


def run_camera(device: str, mode: CameraMode, show_window: bool) -> None:
    cap = open_camera(device, mode)

    configured_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    configured_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    configured_fps = cap.get(cv2.CAP_PROP_FPS)
    configured_fourcc = fourcc_to_string(
        cap.get(cv2.CAP_PROP_FOURCC)
    )

    print("\n请求设置：")
    print(f"  设备：{device}")
    print(f"  编码：{mode.fourcc}")
    print(f"  分辨率：{mode.width}x{mode.height}")
    print(f"  帧率：{mode.fps:.2f} FPS")

    print("\n驱动实际接受的设置：")
    print(f"  编码：{configured_fourcc or '未知'}")
    print(f"  分辨率：{configured_width}x{configured_height}")
    print(f"  驱动报告帧率：{configured_fps:.2f} FPS")
    if show_window:
        print("\n按 Q 或 Esc 退出。")
    else:
        print("\n未检测到图形界面，已切换为无窗口模式，按 Ctrl+C 退出。")

    window_name = f"USB Camera - {device}"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_counter = 0
    measured_fps = 0.0
    measurement_start = time.perf_counter()

    consecutive_failures = 0

    try:
        while True:
            success, frame = cap.read()

            if not success or frame is None:
                consecutive_failures += 1
                print(
                    f"\r读取画面失败，连续失败次数："
                    f"{consecutive_failures}",
                    end="",
                    flush=True,
                )

                if consecutive_failures >= 30:
                    raise RuntimeError(
                        "连续读取摄像头画面失败，请检查摄像头连接或模式设置。"
                    )

                time.sleep(0.01)
                continue

            consecutive_failures = 0

            frame_counter += 1
            current_time = time.perf_counter()
            elapsed = current_time - measurement_start

            # 每约 1 秒更新一次实际 FPS。
            if elapsed >= 1.0:
                measured_fps = frame_counter / elapsed
                frame_counter = 0
                measurement_start = current_time

            frame_height, frame_width = frame.shape[:2]

            line1 = (
                f"Resolution: {frame_width}x{frame_height}  "
                f"Measured FPS: {measured_fps:.1f}"
            )
            line2 = (
                f"Requested: {mode.fourcc} "
                f"{mode.width}x{mode.height}@{mode.fps:.1f}"
            )

            cv2.putText(
                frame,
                line1,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line2,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if show_window:
                cv2.imshow(window_name, frame)

            print(
                f"\r实际画面：{frame_width}x{frame_height}，"
                f"实测帧率：{measured_fps:6.2f} FPS",
                end="",
                flush=True,
            )

            if show_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            else:
                time.sleep(0.001)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n摄像头已关闭。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="自动以 USB 摄像头的高分辨率/高帧率模式显示画面"
    )
    parser.add_argument(
        "--device",
        default="/dev/video0",
        help="摄像头设备，默认：/dev/video0",
    )
    parser.add_argument(
        "--priority",
        choices=["resolution", "fps", "bandwidth"],
        default="resolution",
        help=(
            "模式选择策略：resolution=分辨率优先；"
            "fps=帧率优先；bandwidth=每秒像素量优先"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="打印摄像头支持的所有模式",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="只打印支持模式，不打开摄像头",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="强制使用无窗口模式，适合远程服务器",
    )

    args = parser.parse_args()

    modes = get_camera_modes(args.device)

    if not modes:
        print(
            f"没有从 {args.device} 读取到离散摄像头模式。\n"
            "请执行以下命令检查：\n"
            f"v4l2-ctl -d {args.device} --list-formats-ext",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list or args.list_only:
        print_modes(modes)

    if args.list_only:
        return

    selected_mode = select_mode(modes, args.priority)
    show_window = has_gui_display() and not args.headless

    print(f"\n选择策略：{args.priority}")
    print(f"最终选择：{selected_mode}")
    if not show_window:
        print("提示：当前环境没有可用图形界面，脚本将以无窗口模式运行。")

    try:
        run_camera(args.device, selected_mode, show_window)
    except RuntimeError as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()