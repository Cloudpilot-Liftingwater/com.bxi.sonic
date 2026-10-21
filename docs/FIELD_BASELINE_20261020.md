# test-01真机基线与本地同步（2026-10-20）

以操作员确认可用后的真机文件为基线。输入帧管理、夹爪、控制框架和模型从真机归档同步；本地评测、采集工具保留。此次没有远端写入、服务启停、机器人启动或CAN动作，也没有更新GitHub。

## 实际使用的软件

| 部分 | 真机来源 | 本地保存/维护位置 |
|---|---|---|
| SONIC + BXI demo03 + DM | bxi_vla_ws/bxi_rl_controller_ros2_example/src/bxi_example_py_elf3/mods/com.bxi.sonic | /data/ros2_ws/com.bxi.sonic-main |
| 主example + basic_actions + 消息/遥控器/相机源码 | /home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example | /data/ros2_ws/bxi_rl_controller_ros2_example-pr |
| 下层example环境 | /home/bxi/bxi_ws/bxi_rl_controller_ros2_example | /data/ros2_ws/bxi_rl_controller_ros2_example-test01-underlay |
| r7任务采集 | /home/bxi/sonic_pico_collection_20261019_r7 | tools/pico_collection/sonic_pico_collection_20261019_r7 |
| r7实际服务单元 | /etc/systemd/system/elf3_pico_collection.service | tools/pico_collection/field_test01_20261020 |
| 原采集基础库与本地ROS相机支持 | /home/bxi/bxi_vla_ws/real_robot_collection | /data/ros2_ws/real_robot_collection-test01 |
| 跟踪记录/观察点 | 已安装policy/tracking_tap + v4上传包 | 根policy/tracking_tap及tools/tracking_eval/recording |
| 参考重建/误差报告 | 工作站软件，机器人未部署 | tools/tracking_eval/evaluation |
| 同事C控制核及ROS桥接 | 机器人demo03上传包 | tools/gripper_native |
| 硬件/communication二进制、XR服务、网关 | /opt/bxi/bxi_ros2_pkg、/opt/apps/roboticsservice、/opt/bxi/bxi_rc_ros2 | 下述快照rootfs的相同路径 |
| 仍被使用的环境脚本 | /home/bxi/sonic_gripper_fastfollow_20260917/environment.sh | 快照rootfs对应路径；不表示整包旧控制逻辑在用 |

真机快照根：`/data/ros2_ws/robot_software_snapshots/robot_20261020_Elb7su`。
`rootfs`按原路径展开；离线绝对符号链接已改为镜像内相对链接，原始链接保留在tar。不要直接source镜像里的ROS环境或把它当成工作站安装目录。

## 为什么选择这份版本

- 归档时hardware进程106329来自 `/opt/bxi/bxi_ros2_pkg`；控制器106331来自主工作区install。AMENT/PYTHON/动态库优先级是主bxi_vla_ws → 供应商环境 → 下层bxi_ws → ROS Humble。
- 主工作区Git HEAD为 `c62e07716f491fa02c3a18a3ec2e1487e2ef688e`，下层为 `eafe471b893548d399bdbdf22faf759c6e3a6b36`；两者都有未提交/未跟踪现场文件，故以文件哈希而非仅Git提交同步。
- SONIC源目录与install目录122个文件内容一致；宿主包72个Python文件src/build一致。
- 当前控制器日志明确打印demo03、KP25/KD0.25、左CAN5/右CAN6、速度1400°/s；左右都有first_feedback、open_ready及control记录。更早进程里反向映射的日志是历史，不覆盖新证据。
- r7源码77文件与本地原包完全相同；实际安装服务的bus参数是5/6，原包模板仍为6/5。归档时监听器inactive，本次未启停。

## 同步内容与本地差异

初始差异：独立库12个同名文件不同、缺6个真机文件；主example的src有23个不同、缺7个。主要补入demo03、tracking_tap、collection_telemetry、token模型，并统一plugin/state/policy和现场PICO逻辑。主框架同时包含现场退出准备钩子、图导出失败不阻断启动等实现，调度逻辑保持现场版。

`pico/streamed_smpl_ref.py`原本已相同，继续采用完整10帧、成功推理后推进、重复不刷新新鲜度、断流保持参考。A+X切出POSE后保持人体参考，SONIC继续用当前本体反馈计算控制，并非冻结电机命令。r7暂停写任务帧，恢复后保持同一session，用capture_segment分隔两段。

本地独有ZeroLab接口、reference_gate、额外wrist模块、相关部署脚本/测试，以及人体模型启动预检、venv路径修正、宿主inspector/独立推理频率改动，没有被当作真机已验证功能混入。被覆盖的旧文件和独有代码完整保存在：

- 全库同步前归档：`/data/ros2_ws/robot_software_snapshots/local_before_sync_20261020`。
- 本次逐文件原件与操作清单：`/data/ros2_ws/robot_software_snapshots/local_pending_20261020/files`、`plan.json`。

人体模型数据本身在真机库内，并未删除。上述更严格预检与venv修正仍是有价值的待合并改动，未来单独审查，不应宣称同步后包含所有本地最新特性。

旧monolithic `/data/ros2_ws/bxi_rl_controller_ros2_example-main` 和原同事GUI库 `/data/ros2_ws/bxi_gripper_demo-main` 保留为历史/来源，不用下层Mod框架覆盖。主example旧build/install未重建；本次同步的是维护源码，部署资产完整保留在原始快照。

## 参数和快捷键

当前BXI选择 `gripper_backend: bxi_demo`，闭合0°/打开300°为电机绝对角度。入口各侧使能→收到反馈→打开300°→扳机控制；退出只停止发送。BXI KP/KD为25/0.25，速度1400°/s、加速度28000°/s²。`mod.yaml`里40/3等增益和寻端点参数属于保留的DM路径，不能据此判断BXI增益。

A+X切换POSE；ABXY是PICO人体标定/启停；左握把+A开始/保存任务条，左握把+B弃用；本地2026-10-21版已删除Y+A/Y+B采集别名；旧归档保留历史代码。保持和恢复依赖实际POSE有效帧。

## 验证与边界

- 原始归档17678条目、读取期间变化0，整包及文件校验已通过。
- 从真机demo03包中取回的C桥接和同事原始backend可离线复建；复建 `.so` 与实际运行二进制SHA256相同：`7123406190d91131a279265440c0630432a1158475305f3399610a81af5665fc`。
- 本轮离线行为检查及最终逐文件结果另见 `OFFLINE_VALIDATION_20261020.json`。不将离线测试等同于新增真机验收。
- 跟踪软件此前15秒记录724条trace，120秒6000条trace、99499条可评分反馈；它们不是99499次独立推理。参考是同token学习模型重建，不是独立动捕真值。缺少世界位姿的全局项目仍不可测。
- 任务采集此前同session暂停/恢复有记录，但有效行数因输入缺项为0，不能宣称已通过完整有效任务条的现场验收。

列出的5个链接缺口均为系统项：两条Ubuntu模块配置链接，以及三条ROS Humble ament/path/library环境hook。系统ROS/Python采用版本清单留存，不复制完整OS。厂商只交付的二进制不代表有其源码。额外PanelOS/离线TTS等服务只留存了发现到的单元及进程信息，未导出其全部软件树，不能称整机磁盘备份。

归档器原摘要声称未包含录制数据不够准确：此快照实际含旧real_robot_collection/data下8814个历史文件（约1.66GiB）。原始快照保持不动；此次维护源码同步排除了该data目录，没有把旧任务数据混入代码库，也没有将其视为新r7测试结果。
