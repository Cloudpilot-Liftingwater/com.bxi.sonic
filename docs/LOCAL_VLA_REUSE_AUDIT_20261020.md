# 本地VLA资产核查与复用决定

核查日期2026-10-20。依据用户指定的《GR00T_东华VLA本地资产汇总报告_20260821.md》和六个目录的实际文件。8月报告用来找入口，结论以当前代码、配置和数据字段为准。没有修改六个源资产目录，没有运行训练或机器人。

## 采用哪个库

**训练接入以`/data/ros2_ws/Isaac-GR00T/examples/ELF3/native_1d/`为主。整体模型、仿真和部署资产以`/data/ros2_ws/GR00T-WholeBodyControl/`为资料主库。** 不把某个目录所有文件一股脑覆盖另一个目录。

| 用户提供的目录 | 当前实际内容 | 采用方式 |
|---|---|---|
|Isaac-GR00T|N1.7训练实现、ELF3 legacy/native两套配置、tag注册、processor、数据验证、训练入口、接口测试|当前31自由度／66维动作的训练基线|
|GR00T-WholeBodyControl|SONIC、完整模型资产、训练／推理子库、仿真工具、文档|整体资产最齐全；按用途引用，不用旧68维checkpoint替代native66|
|ELF_Gr00t-wc_reasoning-6a8ce…|推理、SONIC桥接、IsaacLab仿真及工具|用于推理／仿真对照，不替代完整训练库|
|ELF3_VLA_Models_metadata|carry_boxes_mition_token_ckpt10000的config、processor、statistics、embodiment_id、权重索引和latest|旧模型的元数据证据；该目录没有完整safetensors权重|
|ELF3_VLA_Models_data|三个复制／验证JSON报告|不是原始训练数据；真实数据位置需按报告指向查找|
|VLA补充材料|5份需求、ABI、Schema、任务协议、交接Markdown|设计与历史上下文；不能覆盖已实现代码的字段、tag和单位|

本机已有实际数据：

- `/data/dataset/GR00Tdata/boxestakeaway_sim`：旧33状态／68动作体系。
- `/data/dataset/GR00Tdata/boxestakeaway_sim_native1d_v1`：身体29＋双爪2，31状态列、31 WBC列、64 token、50Hz；形状可复用，但爪坐标仍是仿真开度，不能与真机电机角混用。

两个推理目录中的native_vla_sonic.py、bxi_mapping.py、runtime_env.sh、run_server.sh内容相同。该native_vla_sonic.py内部仍有gripper_target4路径；文件名里的“native”不能证明它就是新的一爪一自由度接口。未把整个推理目录认定为逐文件完全相同。

Git基线：Isaac-GR00T为b9955401d50c92a29258732e3ad6ccd579f1bdc0（15个已跟踪文件有本地更改，另有ELF3新增文件）；WBC为e3018c79f1a0c92b254470db312c25d08794806d（9个已跟踪文件有本地更改）。单看commit不足以复现ELF3适配，应保留本地文件及指纹。

## 直接复用的文件

以下路径相对训练库根：

| 文件 | 用途 |
|---|---|
|examples/ELF3/native_1d/elf3_native_1d_config.py|视频／状态／动作／语言顺序和动作表示；40步窗口|
|examples/ELF3/native_1d/modality.json|数据列到各模态的切片|
|examples/ELF3/native_1d/elf3_native_1d_abi.json|31自由度、34状态输入、66动作、50Hz、slot14及模型约定|
|gr00t/data/embodiment_tags.py|ELF3_NATIVE_1D枚举|
|gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py|elf3_native_1d对应projector槽14|
|gr00t/utils/elf3_embodiment.py|checkpoint身份、统计维度、旧槽保护等检查|
|examples/ELF3/native_1d/build_processor.py|按native统计构建processor，保留独立身份|
|examples/ELF3/validate_dataset.py|已有数据校验器|
|examples/ELF3/native_1d/train_smoke.sh、train_small.sh、train_formal.sh|已有训练入口与参数，不另建一套|

modality和Python配置逐字节收进SONIC；ABI元数据随后按用户要求修正为绝对电机rad，并记录原指纹和修改，详见REUSED_ASSETS.json；训练源码继续使用用户已有训练库。复制ABI中的绝对lineage路径只是来源说明，不作为SONIC运行依赖。

## 旧模型为什么不能直接替代

ELF3_VLA_Models_metadata中的statistics实际显示new_embodiment：身体29＋左爪2＋右爪2＋重力3＝36状态；token64＋两爪各2＝68动作，embodiment_id指向槽10。它属于旧仿真联动关节布局，既不是用户要求的31自由度，也不是“身体29＋头部2”的布局。

现有native ABI明确标记SIM_DATA_DERIVED_HARDWARE_CALIBRATION_PROVISIONAL_NO_TRAINED_CHECKPOINT。应理解为：**接口与训练工具已实现；不能据此宣称已取得匹配的真机native66训练模型。** 旧step10000模型禁止直接改tag或改维度后当native模型使用，沿用原库既有检查。

## 真机接入还差什么

1. 现场冻结r7按30Hz相机取行；本地客户v5已实现sonic50_latest_rgb，数值使用真实推理源时刻、图像允许有审计的复用，离线验证通过，尚未部署真机。
2. 按用户随后确认，真实训练以电机rad为标准，不向仿真opening_coordinate转换。本库导出器与ABI已经修正；旧仿真数据/统计不能直接混用，正式训练与推理仍需统一单位元数据。
3. 三相机和A+X前后同条有效数据仍待现场验收；文件完整和离线测试不能替代现场采集。

本轮撤回先前新建的头部训练字段和NEW_EMBODIMENT配置，改为既有native31/66，停止默认自动导出不兼容的30Hz数据。原始r7记录和现场控制保持可用；控制程序、夹爪参数、模型、输入帧管理和A+X控制逻辑未改。

## v3历史验证（当前v4结果见VALIDATION.json）

- native接口及embodiment迁移的已有测试：15通过。
- 导出器测试：9通过，含头部缺失允许、身体缺失拒绝、30Hz拒绝、错误映射／硬件身份拒绝。
- 本地ELF3适配GR00T加载120条合成记录、2个连续子序列、360张解码图像，取得42个合法40步窗口；状态[1,34]、动作[40,66]，槽位14。
- 抽查既有native仿真数据第1条1013帧，原库validator无FAIL；5个WARN为4组float64需转float32、缺任务成功标签。未重解码整套仿真视频。
- 合成夹爪映射不是实机标定；未训练模型，未在机器人安装，未操作设备。

详细证据见tools/gr00t_dataset/VALIDATION.json。客户v2包中包含头部的训练接口已撤回，当前使用motor_rad v4包；此前合成开度映射不再用于真实数据。
