> v14已进一步复用官方状态处理、清理/合并和统计缓存；v15仅修复接收/编码性能，下面是历史实现说明。当前流程见[在线说明](../tools/gr00t_dataset/ONLINE_COLLECTION.md)。

# 官方采集代码复用说明

当前本地方案：**保留ELF3取数适配，训练文件由NVIDIA SONIC官方写入器生成。** 不再维护另一套Parquet、MP4、LeRobot元数据写入实现。尚未安装到机器人。

## 官方有哪些现成代码

来源更正（2026-10-21）：此前 e3018c79 是本地 ELF3 集成提交，不是 NVIDIA 官方提交。现已按真正官方提交 `45c03facbf848e0c743091b7d9c17952096ff908` 逐项核对，源文件和哈希见 `tools/gr00t_dataset/vendor/sonic_data/UPSTREAM.json`。

| 部分 | 官方实现 | 本库处理 |
|---|---|---|
| 收取机器人数据 | `scripts/run_data_exporter.py`收G1的ZMQ数据，依赖G1关节/手模型及FK | 保留ELF3 ROS2适配，不能直接把G1入口当ELF3入口 |
| 图像与数值节拍 | 定时采集，使用最近收到的图像缓存 | 沿用此原则；相机30Hz，本项目实际推理行50Hz；保留源时间和有效性校验 |
| 训练文件 | `data/exporter.py`的`Gr00tDataExporter`调用LeRobot，管理Parquet、任务、episode及统计元数据 | 直接复用；仅将一个导入路径改为库内相对路径 |
| 视频 | `data/video_writer.py`异步PyAV编码MP4 | 复用并修正结束等待、异常传递和空下一条视频问题 |
| 归一化统计 | Isaac-GR00T `gr00t/data/stats.py` | 直接提取未改动的`calculate_dataset_statistics`函数，不引入训练框架其他依赖 |
| A+X同条暂停续采 | 官方`EpisodeState`没有对应暂停状态 | 保留已实现的ELF3 session协议；暂停期间不记任务行，恢复仍为同一原始任务 |
| G1额外字段 | G1手、EEF、VR/SMPL/planner等字段 | 不强行添加到ELF3训练输入，继续用既有native_1d配置 |

官方循环会使用缓存数据；ELF3适配仍拒绝用陈旧token补出动作行。正常静止动作可以有相同数值，但必须来自新收到的有效推理结果。

## 实际数据流

```text
ELF3 ROS2 + PICO + 三相机
  → 已有源时间对齐 / A+X暂停续采
  → 原始任务：frames.jsonl + JPEG + meta.json
  → ELF3字段适配和有效片段检查
  → 官方Gr00tDataExporter / VideoWriter
  → LeRobot v2.1：Parquet + 三路MP4 + meta
  → 官方GR00T统计 / 本地既有ELF3 N1.7加载与训练
```

本版复用的是官方写入核心，**没有将机器人入口换成官方G1采集程序，也没有改成现场边录边写MP4**。先保留原始记录，再在工作站生成训练包，可以避免给正在控制机器人的进程加入Torch/视频编码依赖。原始保存完成不等于训练包已生成；仍需执行一次导出。

A+X前后属于同一原始任务。供训练读取时按官方清理后的方式合并成同一episode，重建连续帧号和时间；40步窗口允许跨接缝。`meta/source_map.json`保留原始时间和每行capture_segments，便于追溯。弃用/失败的原始条不进入训练导出。

训练输入保持：身体29＋夹爪2；`observation.state`31维，重力3维，token64维，`action.wbc`31维。双爪是实测/最终目标电机rad，未改成开度。既有modality、embodiment和训练配置未变。

## 官方源码及补丁位置

- `tools/gr00t_dataset/vendor/sonic_data/exporter.py`：官方写入器。
- 同目录`video_writer.py`：官方视频写入器及小补丁。
- 同目录`UPSTREAM.json`、`LOCAL_CHANGES.patch`：提交、文件SHA256、精确差异。
- `tools/gr00t_dataset/vendor/gr00t_stats.py`及`GR00T_STATS_UPSTREAM.json`：官方统计函数和来源。
- `tools/gr00t_dataset/official_writer.py`：ELF3字段、依赖检查和视频读回。
- `tools/gr00t_dataset/export_dataset.py`：源数据检查和调用入口；原自写Parquet/MP4/LeRobot元数据生成代码已删除。

官方视频原实现以“队列为空”判定可关闭，不能保证已被工作线程取出的最后一帧编码结束。补丁采用结束标记并等待线程，错误回传给调用方；新条尚无帧时不创建空文件或线程。失败输出留在`.inprogress`目录，不冒充成功训练包。

使用方法与依赖见[导出工具说明](../tools/gr00t_dataset/README.md)。离线验证记录见同目录`VALIDATION.json`；真实相机、帧率和现场任务仍需验收。
