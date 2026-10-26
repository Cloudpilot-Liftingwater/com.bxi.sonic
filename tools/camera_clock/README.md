# USB相机时钟换算补丁

usb_cam 0.8.1在启动时缓存墙钟与单调时钟差值。系统之后校时，旧差值会使ROS图像header过早或过晚。现场曾测出约565ms的错误偏移。

补丁仅在两处DQBUF后，对MONOTONIC缓冲区使用当前时钟差生成header；仍使用内核帧时间，不改成发布时间，不硬加565ms，不改变帧率、像素或缓冲个数。未识别的时间类型保留原行为。代码见CLOCK_REFRESH.patch及vendor/usb_cam-0.8.1，原许可证保留。

10月23日现场确认加载SHA256为513f7143ccbb376d92350ef28ab084e340e3c2bb840cafdf7c6874283a2e9c76的新库，左456、右455帧匹配，header换算偏差约-0.000715ms。内核帧时间到ROS接收仍约129ms，其中约119ms在取帧之前。补丁消除的是额外时钟误差，不等于消除了光学/USB/驱动延迟。历史数据不在本次发布中改时间轴。

运行库已放入tools/customer/runtime/ros_humble_x86_64/lib/libusb_cam.so，采集启动检查与交付清单同步。相机进程需要由操作者正常重启才能加载新库；发布本库本身不操作机器人。

Ubuntu22.04构建依赖：g++、pkg-config、FFmpeg开发包、OpenCV开发包。离线构建及真实read_frame路径测试（模拟ioctl和时钟，不打开设备）：

```bash
python3 tools/camera_clock/build.py --output /tmp/sonic-camera-build
```

输出目录必须不存在且位于库外。可传--baseline /旧运行库/libusb_cam.so复现旧偏移bug。构建结果不会自动安装；编译器差异可能改变二进制SHA256，发布前须同步交付指纹并验证ABI。
