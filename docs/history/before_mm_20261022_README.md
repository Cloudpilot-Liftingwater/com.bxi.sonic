# SONIC / ELF3 test-01

本库包含 **2026-10-20 test-01真机基线** 的SONIC运行源码、模型、夹爪程序，以及后续完善的采集和跟踪评测工具。客户从[交付入口](tools/customer/README.md)配置外部ROS/example环境；运行资源随库提供，不依赖开发工作站路径。

**验证范围：** 身体输入管理、A+X保持/恢复、BXI夹爪及跟踪记录链路已有真机验证；数采v14已在真机完成2次保存、3次弃用，但发现写入积压和时间错位。当前 `20261021-collection-15-nonblocking-ingress` 完成本地修复和ROS回放，尚未真机验收。上传代码不代表机器人已安装该版本。

## 当前功能与版本

| 项目 | 当前基线 |
|---|---|
| 身体输入 | playout-01：完整连续10帧窗口；推理成功后才消费；重复帧不刷新输入新鲜度；断流保留参考 |
| A+X | 切出POSE保留最后人体参考，SONIC继续闭环；切回有效POSE恢复跟随 |
| BXI夹爪 | demo03；左扳机→CAN5，右扳机→CAN6；每侧独立使能/等待反馈 |
| 夹爪位置 | 电机绝对角度：闭合0°、打开300°；入口打开后接扳机；退出停止发送 |
| 夹爪参数 | KP25、KD0.25、最大速度1400°/s、加速度28000°/s²；不重新寻零，不由BXI故障请求全身零力矩 |
| 模型/采集输出 | `assets/sonic_action_token.onnx`，29身体关节输出及64维motion_token；夹爪遥测接口已包含 |
| 任务采集 | v15：在线官方MP4/Parquet写入；A+X暂停恢复同一episode，清理后训练窗口可跨接缝 |
| 跟踪评测 | v4观察点已包含；记录器+v2参考重建/误差报告保留在tools/tracking_eval |

DM控制代码仍保留。`mod.yaml`里 `gripper_backend: bxi_demo` 选择当前BXI链路；其中大量旧 `gripper_kp` / 校准参数属于DM后端，**不控制BXI demo的25/0.25增益**。BXI原生源码及构建方法见 [gripper_native](tools/gripper_native/README.md)。

## 常用入口

- [真机基线、外部依赖、同步证据与限制](docs/FIELD_BASELINE_20261020.md)
- [任务采集使用说明](tools/pico_collection/README.md)
- [跟踪评测使用说明](tools/tracking_eval/README.md)
- [历史现场采集服务配置：左5/右6](tools/pico_collection/field_test01_20261020/README.md)

客户使用[本机记录和离线评估入口](tools/customer/README.md#跟踪记录及离线评估)。test-01维护者也可在工作站使用历史SSH便捷入口，并显式指定机器人当前地址：

```bash
cd /实际路径/com.bxi.sonic
SONIC_EVAL_HOST=bxi@机器人当前IP bash tools/tracking_eval/tracking_eval.sh --record 120
```

这只记录已运行控制器的数据，120秒后自动结束并生成HTML；不需要相机。它不启动机器人或任务采集监听器。原始数据、模型重建参考、控制目标、最终指令和实测角度分层保存；没有全局真值的项目仍标为不可测。

## 配套工作区

- 外部example宿主提供Mod API 4、basic_actions、ROS通信消息及机器人I/O。
- 数采与评测的项目源码、原生运行资源、模型和合同均在本库；部署位置由站点配置指定。
- 工作站原始快照、历史库名和同步证据见[工作区导引](docs/WORKSPACE_GUIDE_20261020.md)，它们不是客户运行依赖。

这是Mod API 4插件，宿主、basic_actions、ROS2/Python环境由外部提供。复制源码不是安装或启动；工作站example原有build/install未在本轮重建，不能据此称工作站已经可直接启动真机。

## 留存与验证范围

[本次同步说明](docs/FIELD_BASELINE_20261020.md)列出实机证据、离线检查及未部署改动。原README和硬件文档保存于 `docs/history/`，仅供历史追溯，其旧夹爪/ZeroLab流程不代表当前test-01配置。本地未部署ZeroLab、额外检查代码和旧测试已从当前运行树移到同步前备份，未丢弃。

跟踪工具已有15秒和120秒真机链路验证；身体暂停/恢复、夹爪使用及左右映射由操作员确认。旧采集版本已有有效数据和同session暂停/恢复记录，也出现过JPEG队列中断；v14现场原始文件读回通过，仍有秒级错位；不能据此认定训练质量合格。

## 客户交付与数据说明（2026-10-20 路径整理）

客户使用 [交付入口](tools/customer/README.md)，不使用test-01专用远程安装器。已将采集相机配置、时间戳修正USB驱动和记录入口补为库内相对定位；外部仅显式配置ROS/example环境。此路径整理阶段未改r7采集核心和身体/夹爪控制算法；后续采集频率更新见下文，Python运行器恢复了保留venv路径的修正。此本地路径版尚未部署真机。

- [三摄像头人工检查](docs/CAMERA_MANUAL_CHECK.md)
- [逐文件、逐字段数据说明及仿真格式对照](docs/COLLECTION_DATA_FORMAT.md)
- [原本地未部署改动清单](docs/LOCAL_UNDEPLOYED_CHANGES_20261020.md)
- [交付范围与检查结果](docs/CUSTOMER_DELIVERY_20261020.md)

## 2026-10-20：复用既有ELF3 Native 1DoF训练接口

采用已有Isaac-GR00T的native_1d：身体29＋双爪2＝31自由度，不含头部关节；状态34维（含重力）、动作66维（token64＋双爪），ELF3_NATIVE_1D/slot14，50Hz。原modality和训练配置逐字节收入库内；ABI已按用户要求改为真实电机rad，没有新建embodiment。50Hz是本项目选择，非GR00T统一要求。

- [本地六个资产目录的核查与选择](docs/LOCAL_VLA_REUSE_AUDIT_20261020.md)
- [既有格式的逐项数据字典](docs/GR00T_N17_DATA_FORMAT.md)
- [离线导出和已有训练入口](tools/gr00t_dataset/README.md)

现场已测试v12（50Hz实际推理行），但四条均因旧JPEG队列满而中断。v14将官方写入器接入在线采集，复用30Hz最新图像；现场出现35行/秒写入积压，v15修复接收阻塞和编码吞吐，尚待现场复测。夹爪直接保存真实电机角，不再要求开度映射。因此training_export默认关闭；之前加入的头部训练方案已经撤回。原始采集和A+X同条暂停续采保留，身体和夹爪控制源码未改。本地离线验证见tools/gr00t_dataset/VALIDATION.json，不能当成真机训练数据验收。

- [工作站各库目录导引](docs/WORKSPACE_GUIDE_20261020.md)

- [任务文字标注、实测/目标区别与官方相机处理](docs/TASK_LABELING.md)

当前待上机版本：`20261021-collection-15-nonblocking-ingress`。保存和弃用直接调用官方Gr00tDataExporter接口，在线不再启动旧JPEG队列。[在线采集与依赖安装说明](tools/gr00t_dataset/ONLINE_COLLECTION.md)。旧现场测试文档仅供历史追溯。


数据保存优先复用官方：v14在线调用NVIDIA SONIC/LeRobot写入器；离线调用官方process_dataset清理和合并，训练统计使用GR00T官方函数；原自写Parquet/MP4/元数据写入逻辑已删除。ELF3 ROS取数、A+X同条暂停续采和既有训练配置保留。详细边界、官方代码来源及小补丁见[复用说明](docs/OFFICIAL_COLLECTION_REUSE_20261020.md)。工作站独立导出环境为`/data/ros2_ws/.venvs/sonic-data-export`；机器人已确认安装v14，尚未安装v15。

本次v14还移除了每条重新启动采集进程、最低100行保存门槛、30ms匹配丢行、20ms时间格删帧。已统一GR00T默认训练合同为夹爪电机弧度，沿用官方开环评测入口；旧仿真资产不改单位。原生MP4不在导出时统一缩放，图像处理由模型Processor负责。详见[当前采集说明](tools/gr00t_dataset/ONLINE_COLLECTION.md)。

## 2026-10-21：数采写入积压修复 v15

保留官方状态机、数据写入和清理算法。ROS接收不再等写盘；1kHz状态/IMU只缓存消息，按真实推理行转换。1080p编码使用4线程、腕部各1线程。新增逐行token接收延迟/写入排队时长及编码统计；过载明确保留为中断，避免旧token混配新状态。模型、身体和夹爪控制不变。

本地真实画面写入约60.5→84.4行/秒；120秒隔离ROS回放6000个token写6000行，写入排队P99约4.4ms。不是机器人实测吞吐，现场35行/秒及机器人动作延迟仍需分别核查。相机接收有独立丢帧/延迟边界，详见[性能验证](tools/gr00t_dataset/PERFORMANCE_V15.md)。
