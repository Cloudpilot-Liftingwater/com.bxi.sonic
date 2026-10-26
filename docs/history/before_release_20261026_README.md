# SONIC / ELF3

当前本地版本已适配BXI官方毫米夹爪协议：**闭合0mm、打开70mm，KP2/KD0.05**。控制demo04、数采v16，已离线测试，尚未部署或真机验证。上一个角度控制demo03和数采v15有现场使用记录，冻结包保留。

| 功能 | 当前行为 |
|---|---|
| SONIC输入 | 完整连续10帧；成功推理后推进；断流保持参考 |
| A+X | 切出POSE保持参考，切回恢复；录制前后属于同一episode |
| BXI夹爪 | 左扳机→CAN5、右扳机→CAN6；使能后打开70mm，独立控制 |
| 正常退出 | 夹爪停止发送；BXI故障不主动请求身体零力矩 |
| 数据采集 | 官方MP4/Parquet写入与清理；保留v15非阻塞接收；夹爪实测/目标均为mm |
| 训练合同 | 状态34维、动作66维；身体rad、夹爪mm；ELF3_NATIVE_1D/slot14 |
| 跟踪评测 | 原v4记录观察点和参考重建工具保留，已有真机链路验证 |

- [新协议、参数与数据合同](docs/GRIPPER_MM_PROTOCOL_20261022.md)
- [客户部署入口](tools/customer/README.md)：资源在库内，ROS/example外部依赖显式配置。
- [数采使用](tools/pico_collection/README.md)；[导出和训练检查](tools/gr00t_dataset/README.md)
- [跟踪评测](tools/tracking_eval/README.md)；[三摄像头人工检查](docs/CAMERA_MANUAL_CHECK.md)
- [旧基线与工作区导引](docs/WORKSPACE_GUIDE_20261020.md)；[毫米升级前README](docs/history/before_mm_20261022_README.md)

`mod.yaml`的 `gripper_backend: bxi_demo` 选择BXI。BXI参数位于根目录 `bxi_gripper_mm.json`；mod.yaml的DM增益只用于DM后端。DM、模型、身体控制及完整10帧管理未改。

本库是Mod API 4插件；example宿主、basic_actions、ROS2环境由外部提供。模型、PICO资源、夹爪原生源码、数采、数据合同和跟踪工具均随库提供。客户按交付入口配置路径，不使用test-01专用SSH安装器。

旧弧度数据/统计/checkpoint不能标成mm后直接用。本次不修改历史采集数据，不执行后训练，不自动安装、启动或操纵机器人。
