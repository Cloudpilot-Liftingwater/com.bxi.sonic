# 三摄像头人工检查（先看图，再测采集）

只检查头部彩色、左腕彩色、右腕彩色。无需进入SONIC，无需使能夹爪。下面由操作员执行；没有自动启停服务或机器人脚本。

## 第一步：机器人SSH终端查设备

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-path/*video-index0
fuser /dev/video*
```

2026-10-20本次枚举已确认：USB1对应video0，USB7对应video8，RealSense在USB2；video6是彩色候选节点，尚待格式输出确认。

下一步在机器人SSH终端执行：

```bash
v4l2-ctl -d /dev/video6 --list-formats-ext
fuser /dev/video0 /dev/video6 /dev/video8
```

`fuser`无`-k`参数，只显示占用者；不要杀进程。相机已经被现有驱动占用时，不要再直接打开同一设备；先确认是否有人使用，再由该操作者正常结束自己的相机预览/驱动，或查看现有图像流。

历史腕部对应为USB7左、USB1右；这与夹爪CAN5/6是两回事。换线或换相机后必须通过遮镜头重新确认。头部RealSense有多个video节点，不能任选video0，也不能把深度/metadata节点当彩色。

## 第二步：手动开预览窗口

以下需要**机器人的图形桌面终端**（物理屏幕或远程桌面）。普通SSH没有图形显示环境，直接运行ffplay会报显示错误，这不说明相机坏了。若用SSH X转发，画面流畅度还受网络影响，不适合作为摄像头帧率验收依据。

确认上一步确实存在下列设备后，分别在两个终端运行：

左腕：

```bash
ffplay -window_title LEFT_WRIST -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 -i /dev/v4l/by-path/pci-0000:00:14.0-usb-0:7:1.0-video-index0
```

右腕：

```bash
ffplay -window_title RIGHT_WRIST -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 30 -i /dev/v4l/by-path/pci-0000:00:14.0-usb-0:1:1.0-video-index0
```

头部先对**已由设备列表确认的彩色节点**列出格式（`/dev/videoN`须替换，不能原样运行）：

```bash
v4l2-ctl -d /dev/videoN --list-formats-ext
```

如果彩色节点列出的1920×1080@30模式为YUYV，用：

```bash
ffplay -window_title HEAD_RGB -f v4l2 -input_format yuyv422 -video_size 1920x1080 -framerate 30 -i /dev/videoN
```

如果该模式是MJPG，把`yuyv422`改为`mjpeg`。如果不支持1920×1080@30，先报告实际支持模式，不要硬套命令；当前r7要求这个头部输入尺寸。若只是在较低分辨率下看到图像，尚不能说明满足当前采集配置。

ffplay的V4L2输入、输入格式和分辨率参数依据[FFmpeg设备文档](https://ffmpeg.org/ffmpeg-devices.html#video4linux2_002c-v4l2)；退出按键依据[ffplay文档](https://ffmpeg.org/ffplay.html)。

## 通过标准

1. 三个窗口都能连续出图，分别遮挡镜头确认头/左/右对应正确。
2. 三路同时运行约一分钟，挥手能看到连续更新，无卡死、花屏、断开或设备报错。
3. 记录头部1920×1080@30、两腕640×480@30是否均可打开。
4. 每个窗口按`q`退出（或各终端Ctrl+C），释放预览占用，再进入采集测试。

这一步验证三路相机和同时取流能力；还没有验证ROS时间戳对齐、有效采集行、保存及A+X续采。三路通过后才做下一步，当前先不要启动新的采集条。
