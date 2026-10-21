# ELF3 / SONIC / GR00T 本地工作站目录导引

更新：2026-10-20。此文组织各库的职责和入口，不搬动代码/模型/数据。以下`/data`、`/home`地址是当前工作站资产位置，不是客户SONIC运行时的隐式依赖。客户按SONIC库内相对入口与显式外部环境配置部署。

## 日常应打开哪个目录

|用途|当前工作站目录|主要入口／备注|
|---|---|---|
|真机SONIC独立库、夹爪、采集与评测交付|[com.bxi.sonic-main](/data/ros2_ws/com.bxi.sonic-main)|README.md；policy.py；bxi_demo.py；tools/customer/README.md|
|GR00T N1.7后训练主库|[Isaac-GR00T](/data/ros2_ws/Isaac-GR00T)|examples/ELF3/native_1d；gr00t/model/gr00t_n1d7；examples/finetune.sh|
|完整SONIC/WBC、模型、仿真资产集合|[GR00T-WholeBodyControl](/data/ros2_ws/GR00T-WholeBodyControl)|gear_sonic、gear_sonic_deploy、checkpoints、external_dependencies；资料最齐全，不等于所有子目录都是最新真机版本|
|ELF3推理和仿真对照|[ELF_Gr00t-wc_reasoning](/data/ros2_ws/ELF_Gr00t-wc_reasoning-6a8ce1469481ecb531119fbe37a59a5bc291c969)|native_vla_sonic.py、bxi_mapping.py、config、sonic_runtime；仍有旧四夹爪坐标路径，不能视为本次电机rad VLA部署完成|
|旧训练模型元数据|[ELF3_VLA_Models_metadata](/data/ros2_ws/ELF3_VLA_Models_metadata)|carry_boxes_mition_token_ckpt10000；旧36状态/68动作统计，不当作本次34/66模型|
|数据复制/核验报告|[ELF3_VLA_Models_data](/data/ros2_ws/ELF3_VLA_Models_data)|3份JSON报告；这里不是完整视频/Parquet数据集|
|历史需求和接口材料|[VLA补充材料](/data/ros2_ws/VLA补充材料)|5份Markdown；历史规定与当前实码冲突时按已审计代码/用户新要求|
|真机example Mod宿主源码|[bxi_rl_controller_ros2_example-pr](/data/ros2_ws/bxi_rl_controller_ros2_example-pr)|SONIC外部宿主；Mod API 4、basic_actions、ROS消息依赖；源码同步不等于build/install已重建|
|旧一体化example（历史对照）|[bxi_rl_controller_ros2_example-main](/data/ros2_ws/bxi_rl_controller_ros2_example-main)|不是此次真机同步的Mod宿主，勿与example-pr混用|
|真机下层example源码|[test01-underlay](/data/ros2_ws/bxi_rl_controller_ros2_example-test01-underlay)|与宿主分开留存，防止覆盖现场不同overlay/underlay|
|基础采集库镜像|[real_robot_collection-test01](/data/ros2_ws/real_robot_collection-test01)|历史/基础对照；当前监听器用SONIC tools/pico_collection中r7内附collector|
|全量真机原始快照与同步备份|[robot_software_snapshots](/data/ros2_ws/robot_software_snapshots)|robot_20261020_Elb7su、同步前本地备份、客户交付包；不在快照上直接开发|

总资产历史报告：[GR00T_东华VLA本地资产汇总报告_20260821.md](/home/huangchenwei/GR00T_东华VLA本地资产汇总报告_20260821.md)。用于查模型/大文件来源；10月代码与坐标选择以当前说明为准。

## SONIC独立库内部

```text
com.bxi.sonic-main/
├── policy.py + pico/                 输入窗口、PICO处理、暂停参考保持
├── state.py + plugin.py + mod.yaml   Mod接入及配置
├── bxi_demo.py + bxi_demo_native.so  已验证的BXI控制
├── tools/gripper_native/            同事控制程序的原生构建源码
├── gripper.py                       保留的DM控制
├── collection_telemetry.py          真实夹爪反馈与最终发送指令
├── tracking_tap.py                  只读跟踪观察点
├── tools/
│   ├── tracking_eval/
│   │   ├── recording/              记录、对时、基础分析
│   │   └── evaluation/             参考动作重建与精度报告
│   ├── pico_collection/
│   │   └── sonic_pico_collection_20261019_r7/
│   │       ├── button_supervisor.py PICO开始/保存/弃用
│   │       ├── session_node.py      A+X同条暂停续采
│   │       └── collector/          三RGB、状态、token、夹爪原始采集
│   ├── gr00t_dataset/              原始数据→训练Parquet/MP4；电机rad
│   └── customer/                   客户配置、路径检查、离线打包入口
└── docs/                           数据、相机、部署、历史范围说明
```

测跟踪精度走tracking_eval；录VLA任务走pico_collection；生成训练数据走gr00t_dataset；训练模型去Isaac-GR00T。这四者不是一个命令。

## 训练侧复用哪些文件

|路径，相对Isaac-GR00T|职责|
|---|---|
|examples/ELF3/native_1d/elf3_native_1d_config.py|模态顺序、当前状态/视频、未来40步动作、tag注册；本SONIC库副本保持相同|
|examples/ELF3/native_1d/modality.json|数据列切片；本SONIC库副本保持相同|
|examples/ELF3/native_1d/elf3_native_1d_abi.json|旧仿真native约定；本SONIC副本已按用户要求改为绝对电机rad，不能说两者语义相同|
|gr00t/data/embodiment_tags.py|ELF3_NATIVE_1D注册|
|gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py|该tag对应slot14|
|examples/ELF3/validate_dataset.py|数据验证；真实数据显式传SONIC内的电机rad ABI|
|examples/ELF3/native_1d/train_smoke.sh、train_small.sh、train_formal.sh|现有训练入口；单位约定、统计和checkpoint元数据需与真实数据一致，不另造训练配置|

维度/顺序不变，不代表旧仿真开度与真实电机角可混用。本轮未覆盖六个训练/资产源目录的历史文件。

## 数据与归档分开放

- [旧仿真训练数据](/data/dataset/GR00Tdata/boxestakeaway_sim)：旧33关节存储、68维模型动作体系。
- [native1d仿真数据](/data/dataset/GR00Tdata/boxestakeaway_sim_native1d_v1)：31维关节存储、66维模型动作；爪仍是仿真开度，不能当作电机rad。
- [用户指定的仿真资产](/data/dataset/东华VLA资产/BXI_whole_body_vla_dataset_isaac/)：场景和仿真数据来源。
- 真机原始数据位置由站点配置`data_root`决定；监听器会输出每条session/episode路径。不要按工作站示例路径猜机器人存储位置。
- [客户包目录](/data/ros2_ws/robot_software_snapshots/customer_delivery_20261020/)：以CURRENT_DELIVERY.json指向为准；旧v2/v3是历史版本，当前v6保留真实电机rad与相机缓存，并修复复用图像重复JPEG压缩造成的写盘积压。

字段与数据文件逐项解释见SONIC的`docs/GR00T_N17_DATA_FORMAT.md`。工作站入口：[数据字典](/data/ros2_ws/com.bxi.sonic-main/docs/GR00T_N17_DATA_FORMAT.md)、[导出工具](/data/ros2_ws/com.bxi.sonic-main/tools/gr00t_dataset/README.md)、[客户部署说明](/data/ros2_ws/com.bxi.sonic-main/tools/customer/README.md)。

任务文字与最新采集模式：[TASK_LABELING.md](/data/ros2_ws/com.bxi.sonic-main/docs/TASK_LABELING.md)。
