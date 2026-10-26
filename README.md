# SONIC / ELF3

当前发布整理：**20261026-mm-v16-camera-tools**。集成已现场使用的SONIC输入管理、BXI毫米夹爪demo04、官方数采v16、跟踪评测，以及已现场核对的相机时钟补丁。

| 功能 | 当前行为 |
|---|---|
| SONIC输入 | 完整连续10帧；成功推理后推进；断流保持参考 |
| A+X | 切出POSE保持参考，切回恢复；官方清理后前后仍在同一episode |
| BXI夹爪 | 左扳机→CAN5、右扳机→CAN6；使能后打开70mm，闭合0mm；KP2/KD0.05 |
| 退出 | 停止发送夹爪命令；BXI故障不主动请求身体零力矩 |
| 数采 | 官方MP4/Parquet写入、保存/弃用和清理；保留v15非阻塞接收 |
| 训练合同 | 状态34维、动作66维；身体rad、夹爪开口mm；ELF3_NATIVE_1D/slot14 |
| 相机时间 | 每次取帧更新单调时钟到墙钟的换算；保留真实内核帧时间 |
| 跟踪评测 | 记录观察点、模型参考重建、指令/实际误差及报告 |
| 下载与检查 | 全部/最近会话、续传、SHA256、三路视频页面、分段GR00T读取检查 |

## 使用入口

- [客户部署与依赖配置](tools/customer/README.md)；[系统Python / venv环境](docs/PYTHON_ENVIRONMENT.md)
- [数采操作](tools/pico_collection/README.md)；[数据导出与训练检查](tools/gr00t_dataset/README.md)
- [批量下载、人工检查、GR00T读取](tools/collection_review/README.md)
- [跟踪精度评测](tools/tracking_eval/README.md)
- [毫米协议与参数](docs/GRIPPER_MM_PROTOCOL_20261022.md)；[相机时钟补丁](tools/camera_clock/README.md)
- [本次发布、验证边界及待处理问题](docs/RELEASE_20261026.md)
- [维护与Git发布流程](docs/MAINTENANCE.md)

## 验证状态

10月22日毫米版现场完成两次保存、两次弃用，保存/弃用各一次暂停续采；后续还验证了VLA手柄进入SONIC后的采集。10月23日双腕相机报告确认新库已加载，旧约565ms时钟偏差消失，内核帧时间到接收仍约129ms。5条历史毫米测试数据已清理导出并通过GR00T首/中/尾40步窗口读取及归一化检查；没有运行模型后训练。

**尚未解决：** 数采相机/记录器退出会结束监听器；现场服务配置不自动重启。10月23日晚无新录制。当前发布没有修改服务恢复策略，不能把过去保存成功当作无人看守可靠性验收。开始正式任务前确认服务存活、RECORDING及帧数增长；完成后确认saved结果及文件。

本库是Mod API 4插件，example宿主、basic_actions、ROS2及Python依赖为显式外部依赖。库内包含模型、PICO资源、夹爪源码/运行库、数采、合同和评测工具。客户入口按库位置和站点配置找文件，不依赖开发机目录。

旧弧度数据/统计/checkpoint不能重新标成mm后混用。冻结历史包不改写；当前完整发布仍须客户设备环境验收。
