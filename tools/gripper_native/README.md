# BXI demo03 原生控制核源码

来源：test-01归档中的 `gripper_demo_upload.n02upP/sonic_gripper_demo_20261014_v3`。同事原程序在 `colleague_original/`，C桥接在 `bxi_demo_bridge.c`。库根的 `bxi_demo.py` 通过现有ROS CAN通道调用此控制核。

KP=25、KD=0.25、最大速度1400°/s、加速度28000°/s²；绝对角度闭合0°、打开300°。这些BXI设置在C桥接中，不是mod.yaml里的DM参数。

只编译到临时目录，不连接机器人、不覆盖正在用的库：

```bash
python3 tools/gripper_native/build_native.py --output /data/tmp/bxi_demo_native.rebuilt.so
```

根目录 `bxi_demo_native.so` 是机器人原二进制（x86_64）。编译器/架构不同可能改变二进制哈希；不能把编译成功等同于真机验证。构建器检查不会链接PCI驱动、电源控制、发送CAN或创建控制线程。源码与原运行库的复建核对见 `docs/FIELD_BASELINE_20261020.md`。
