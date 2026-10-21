# 哪些本地改动未部署到test-01

比较基准：2026-10-20取回的真机完整归档，不是Git提交号。这里“未部署”只表示真机没有这份内容；**不表示每份本地文件都更新或更好**，部分只是被现场版本取代的旧实现。

| 类别 | 原本地位置（相对对应仓库） | 与现场差别、当前处理 |
|---|---|---|
| ZeroLab另一种输入链路 | SONIC `zerolab/`、`reference_gate.py`、`deploy/*zerolab*` | UDP输入、格式转换、重采样、显式arm/hold/rearm及网络部署。test-01使用PICO链路，不依赖这些；已移出当前运行树留存 |
| 人体模型额外预检查 | SONIC `pico/manager_launcher.py`、`pico/gear_sonic/trl/utils/numpy_smpl.py` | 本地多了启动前模型检查及有限数值/骨架父节点检查；现场有基础检查和库内模型文件，但没有这套额外检查；额外实现留存未恢复 |
| Python虚拟环境路径修正 | SONIC `pico/python_runtime.py`；example同名模块及`test/test_sonic_python_runtime.py` | 本地保留venv的bin/python符号链接，现场resolve会指向系统解释器。**本次客户交付整理恢复了这项路径修正到独立库，并离线验证；尚未部署真机** |
| 框架检查接口 | 主example `framework/inspection.py`、`framework/__init__.py`及相关测试 | 本地额外检查/导出接口，现场没有；留存，不纳入当前运行宿主 |
| 框架按状态推理频率 | 主example `framework/runtime/controller.py`、`framework/runtime/state_builder.py` | 本地有按状态请求频率的调度改动；现场运行实现不同，保留现场版本，包括现场退出准备逻辑；原改动留存 |
| 辅助工具/旧测试 | `pico/offline_reference_player.py`、`pico/gear_sonic/trl/utils/elf3_wrist.py`、ZeroLab/状态机/旧playout测试等 | 真机未部署的辅助实现与版本专用测试；不是现场启动缺件 |
| 旧控制/配置差异 | SONIC `policy.py/state.py/plugin.py/gripper.py/mod.yaml`及旧文档 | 原独立版与实际demo03/PICO链路不同，已被真机版本替换。这类不能统称“未部署的新功能” |

完整备份：`/data/ros2_ws/robot_software_snapshots/local_before_sync_20261020/`。
逐文件变更前内容：`/data/ros2_ws/robot_software_snapshots/local_pending_20261020/files/com.bxi.sonic-main/`及同级`bxi_rl_controller_ros2_example-pr/`。
操作清单：`local_pending_20261020/plan.json`、`APPLIED.json`。原快照及备份不在客户运行依赖内。

本次新增的客户路径入口、USB相机运行库、数据字典和README也属于**本地交付整理，尚未部署真机**。它们不替代现场验收；身体输入/消费/暂停策略、夹爪算法和参数、数采同步/暂停保存算法未在本次重写。

## 既有Native 1DoF接口复用（尚未现场部署）

- tools/gr00t_dataset撤回包含头部的36/68配置；采用已存在的ELF3_NATIVE_1D、状态34／动作66、50Hz。原modality和配置逐字节收入库内；ABI的夹爪语义已按后续用户要求改为真实电机rad。
- 导出器只使用身体29和双爪2，允许缺失头部；复用native66.py验证坐标映射。旧30Hz记录不能仅修改fps转为50Hz；夹爪直接采用真实电机rad，不需要开度映射。
- 客户入口保留任务文字，撤回默认自动导出；原始记录、r7同步和同条暂停恢复继续保留。verify_saved支持按源记录端口元数据离线读回。
- 控制源码、模型、PICO帧管理、夹爪增益／映射、真机现场旧r7均未修改。
- 细节见LOCAL_VLA_REUSE_AUDIT_20261020.md。客户v2中新增的头部训练方案已撤回，勿作为当前交付入口。

## 客户v5：官方相机缓存方式（本地新增，未部署）

- 数采新增`sonic50_latest_rgb`：实际50Hz live推理作锚点，30Hz相机取最近缓存，保留图像源时间、序号和复用标记。无新推理不补造动作；图像最长100ms。
- 改动采集同步、写入元数据、读回和导出来源记录；A+X同条暂停恢复协议、SONIC输入与推理、夹爪控制没有修改。历史r7归档不覆盖。
- 任务文字由`training_task`提供，启动打印确认；具体见TASK_LABELING.md。
- 合成120行、72张不同源画面、48次复用经真实写入器和GR00T数据加载验证。并非120行真机测试结果；训练未启动、机器人未更新。

## 客户v6：上机前复核（本地新增，未部署）

- 复用同一源图像时复用JPEG编码，每行仍写独立图片文件；缓存按相机、源序号、源时间及暂停片段区分，每次录制重置。未扩大队列或复制缺失动作。
- 监听状态打印实际采集代码版本09，同时保留r7目录名信息。
- 修正导引：真机同步主宿主是example-pr，example-main为旧一体化版本。
- 上机流程见FIELD_TEST_20261020.md，离线证据见PREFLIGHT_20261020.json。
