# BXI毫米协议控制核（ABI2）

同事原 `colleague_original/backend/gripper_backend.c` 保留不动，只复用单位无关的STREAM跟随器。`bxi_demo_bridge.c`提供官方毫米协议编解码；根目录Python负责原ROS通道和独立侧生命周期，不连接同事PCI驱动。

参数统一在库根 `bxi_gripper_mm.json`：0～70mm、KP2/KD0.05、326.6667mm/s、6533.3333mm/s²。说明见[新协议合同](../../docs/GRIPPER_MM_PROTOCOL_20261022.md)。

```bash
python3 tools/gripper_native/build_native.py --output /tmp/bxi_demo_native.mm.so
```

只编译，不发CAN、不运行硬件。输出应为ABI2。x86_64工作站库随本库提供；其他架构需本机编译后放在库根`bxi_demo_native.so`。动态库和Python/JSON必须配套更新；加载旧ABI1会拒绝使能。
