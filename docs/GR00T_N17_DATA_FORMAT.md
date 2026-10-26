> 2026-10-22当前版本：BXI demo04 / 数采v16，夹爪开口0～70mm；旧角度参数和v15验证记录仅供历史追溯。使用[毫米协议与合同说明](GRIPPER_MM_PROTOCOL_20261022.md)。新版本尚未真机验证。

# ELF3真机训练数据：频率、字段与文件说明

核对日期：2026-10-20。以当前真机电机坐标和已有ELF3 Native 1DoF模型接口为准。身体29＋双爪2＝31自由度；头部两个关节不进入本接口。**头部RGB画面仍要采集。**

本文件说明目标训练数据与当前工具，不能当成50Hz真机采集已验收。现场冻结r7仍按30Hz相机取原始行；本地客户版新增sonic50_latest_rgb，已离线验证50Hz推理行与30Hz画面复用。未升级机器人采集服务。

## 1. 频率到底是多少

**GR00T N1.7没有统一强制50Hz。** 官方示例数据的`info.json`就是30Hz；SONIC全身VLA教程采用50Hz低层动作时间轴。它们不是矛盾的标准：

|名称|官方证据／本项目选择|含义|
|---|---|---|
|GR00T数据集fps|数据集声明；官方示例30|每秒多少行训练时刻，不代表所有传感器原生都拍这么快|
|SONIC低层解码／控制|教程50Hz|每20ms消费一个token并计算身体动作|
|VLA重规划|SONIC教程默认2.5Hz|每0.4s预测一次动作块，不是每秒仅有2.5个动作|
|动作窗口|沿用本地40步|50Hz时覆盖约0.8s；默认执行前20步后重规划|
|官方相机程序|默认30fps|新画面的产生速度|
|本项目训练目标|沿用50Hz，40步|这是现有ELF3配置选择，不是GR00T全球硬性规定|

来源：[官方30Hz数据示例](https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/demo_data/cube_to_bowl_5/meta/info.json)、[SONIC VLA教程](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)、[相机默认配置](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/camera/composed_camera.py)。

可以训练30Hz策略，但同样40步会变成约1.33s，执行20步约0.67s；动作调度、SONIC接口、数据检查和部署配置必须一起处理。不能只改`info.json.fps`。本轮不更换已有训练配置。

### 50Hz动作如何配30Hz相机

当前GR00T加载器按**行号取视频帧**，不能把每秒50行直接配每秒30帧的视频而不处理对应关系。适合沿用现有加载器的方案：

1. 按真实50Hz推理／动作时刻保存数值数据。
2. 每个时刻选此前最近的有效相机画面；相机没更新时允许沿用上帧，保留相机源序号、拍摄时间和图像年龄。断流不能无限复用。
3. 训练MP4写为50fps、每行对应一帧。其中只有约30帧/秒是新的画面；不做插帧生成新内容。

示例：动作时刻0、20、40、60、80ms，对应相机时间0、0、33、33、67ms。动作仍来自各自真实推理；只复用图像。不能把30Hz token／身体记录复制成50Hz。本地实现限制图像年龄不超过100ms，不用未来画面给早先动作配图。

这是根据[GR00T按索引读取视频的实现](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/dataset/lerobot_episode_loader.py)提出的接入方式。官方SONIC的[导出循环](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/scripts/run_data_exporter.py)也保留最近收到的相机消息供循环使用。**本地客户版sonic50_latest_rgb已实现；现场冻结r7仍为head30，尚需部署和真机验收。** 新模式的数值按实际live推理时刻取样，未复制缺失token；图像缓存处理遵循官方方式。

## 2. 夹爪坐标

训练反馈、训练标签和推理反归一化后的输出均使用**真实夹爪电机绝对角度**，不转成仿真开合坐标，不用0～1开度替代角度：

- 完全闭合：0°＝0 rad。
- 使用最大打开位置：300°＝5.235987756 rad。
- 实测反馈来自`position_rad`，动作标签来自实际发送的`command.position`，均为rad；二者不要求相等。
- 当前demo的`demo_step(target_deg, ...)`接口收“度”，CAN编码与反馈使用“弧度”。若VLA接这个接口，只做`rad × 180/π`单位转换，不是换坐标；也不能把5.236当成5.236°发送。
- 左右按明确的side身份排列，当前真机左CAN5／右CAN6。不能按收到数据的先后顺序拼接。

源代码依据：库根`bxi_demo.py`、`collection_telemetry.py`、`tools/gripper_native/bxi_demo_bridge.c`。导出器不再接受`--gripper-mapping`。模型训练内部仍会归一化，部署时必须用同一数据统计反归一化回电机rad；这不同于改变物理坐标。

旧仿真native31/66数据虽然维度相同，夹爪数值仍是仿真开合坐标。不能与本真机数据直接混用，不能把旧统计或旧checkpoint标成电机角模型。原仿真资产保留原样；本库ABI已明确`absolute_motor_radians`。沿用原modality配置和ELF3_NATIVE_1D/slot14，无需另造embodiment。

## 3. 每份数据包含什么

|数据|训练类型／尺寸|保存位置|目标时间轴|
|---|---|---|---|
|头部RGB|解码后uint8[480,640,3]|ego_view MP4|50行/秒对应50视频帧；相机新画面约30fps|
|左腕RGB|同上|left_wrist MP4|同上|
|右腕RGB|同上|right_wrist MP4|同上|
|身体实测29＋左右爪实测2|float32[31]，rad|Parquet `observation.state`|50Hz|
|SONIC根坐标系投影重力|float32[3]，单位向量|`observation.projected_gravity`|与状态同一行|
|SONIC在线post-FSQ token|float32[64]|`action.motion_token`|与实际推理对应的50Hz行|
|身体最终目标29＋双爪最终目标2|float32[31]，rad|`action.wbc`|与token配对；模型只取末两项|
|真实任务指令|UTF-8字符串|`meta/tasks.jsonl`；每行以task_index引用|任务改变时更新，不是50Hz新文本|
|时间、行号、条目号|float32秒／int64|Parquet五个标量列|每行|
|来源、分段、指纹、物理身份|JSON元数据|`source_map.json`、`elf3_contract.json`|每份导出|

模型实际看到：**三幅RGB＋34维状态＋任务文字**；监督目标是**66维动作**。34＝29身体＋2实测爪角＋3重力；66＝64 token＋2目标爪角。3维重力不是3个关节，64维token也不是64个电机。

`action.wbc[:29]`沿用既有数据格式，保存身体下发目标供核对；它不再拼进VLA动作，不会变成95维。PICO原包、SMPL、IMU原始四元数、速度、力矩和温度不属于当前模型的必需输入；需要的诊断信息留在原始记录。深度图、全局定位和头部关节也不是这套配置的训练项。

当前相机**配置值**：头部1920×1080@30、左右腕各640×480@30；实际支持与实际帧率须现场看图／记录确认。训练导出统一640×480，保留比例、补黑边；头部16:9内容缩为640×360，上下各60黑边。MP4采用H.264/yuv420p，无音频，加载后是RGB，画面有压缩损失。当前r7原始图片保存为JPEG；训练包保存MP4，不需要把JPEG再塞进Parquet。

## 4. 文件夹如何组织

```text
dataset/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   └── episode_000001.parquet
├── videos/chunk-000/
│   ├── observation.images.ego_view/episode_000000.mp4
│   ├── observation.images.left_wrist/episode_000000.mp4
│   └── observation.images.right_wrist/episode_000000.mp4
├── meta/
│   ├── info.json
│   ├── modality.json
│   ├── stats.json
│   ├── episodes_stats.jsonl
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   ├── source_map.json
│   ├── elf3_contract.json
│   └── native_1d_derivation.json
└── EXPORT_RESULT.json
```

每个训练episode一个Parquet、三份同长度视频；上面只列了episode0的视频，episode1同样有三份。chunk-000是分组目录，本库每1000条episode换一个chunk；不是1000帧。

教程中的`data/train-00000.parquet`是概览示意，不是本库实际文件名。当前使用LeRobot v2.1常见的按episode分文件布局，并在info.json明确路径模板。不能只重命名文件而不改路径模板。对应[GR00T数据准备说明](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)与[LeRobot v2.1路径定义](https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/utils.py)。

## 5. Parquet里面是什么

Parquet是**有类型、压缩的列式二进制表格**。不是装着其他文件的压缩包，也不是一行一个JSON文件。可以理解成支持“某格内存一个固定长度向量”的表格。

|完整列名|Arrow类型|每行的含义|
|---|---|---|
|observation.state|fixed_size_list<float32>[31]|这一时刻31个实测角|
|observation.projected_gravity|fixed_size_list<float32>[3]|这一时刻重力向量|
|action.motion_token|fixed_size_list<float32>[64]|这一时刻token|
|action.wbc|fixed_size_list<float32>[31]|这一时刻29身体＋2爪最终目标|
|timestamp|float32|清理后同一episode相对秒数，frame_index/50|
|frame_index|int64|本episode内行号，从0开始|
|episode_index|int64|当前训练episode编号|
|index|int64|整个数据集中的全局行号|
|task_index|int64|tasks.jsonl中的任务编号|

这是9列，不是把66个动作元素各写一列。**每行只存一个时刻；训练加载器再读取连续40行组成[40,66]动作块。** 一条10秒、50Hz的连续数据约500行、每个视频500帧。

人类可读示意（省略号只用于解释，真实向量有完整数值）：

```text
frame_index  timestamp  observation.state       action.motion_token  action.wbc           task_index
0            0.00       [身体29项, 2.5, 2.4]    [64项]               [身体29项, 1.0, 1.2] 0
1            0.02       [身体29项, 2.4, 2.3]    [64项]               [身体29项, 0.9, 1.1] 0
```

这里2.5是实测电机rad，1.0是目标电机rad；不是0.025开度，也不是2.5°。图片没有嵌入表中；episode_index定位三份视频，frame_index定位视频中的同序号帧。

### 31维角度逐项索引

`observation.state`与`action.wbc`使用相同顺序：

|索引|名称|单位／含义|
|---:|---|---|
|0|waist_y_joint|rad|
|1|waist_x_joint|rad|
|2|waist_z_joint|rad|
|3|l_hip_y_joint|rad|
|4|l_hip_x_joint|rad|
|5|l_hip_z_joint|rad|
|6|l_knee_y_joint|rad|
|7|l_ankle_y_joint|rad|
|8|l_ankle_x_joint|rad|
|9|r_hip_y_joint|rad|
|10|r_hip_x_joint|rad|
|11|r_hip_z_joint|rad|
|12|r_knee_y_joint|rad|
|13|r_ankle_y_joint|rad|
|14|r_ankle_x_joint|rad|
|15|l_shoulder_y_joint|rad|
|16|l_shoulder_x_joint|rad|
|17|l_shoulder_z_joint|rad|
|18|l_elbow_y_joint|rad|
|19|l_wrist_x_joint|rad|
|20|l_wrist_y_joint|rad|
|21|l_wrist_z_joint|rad|
|22|r_shoulder_y_joint|rad|
|23|r_shoulder_x_joint|rad|
|24|r_shoulder_z_joint|rad|
|25|r_elbow_y_joint|rad|
|26|r_wrist_x_joint|rad|
|27|r_wrist_y_joint|rad|
|28|r_wrist_z_joint|rad|
|29|l_gripper_motor_joint|rad，绝对电机位置|
|30|r_gripper_motor_joint|rad，绝对电机位置|


旧r7原始JSONL的身体向量仍有31个历史槽，最后两个是可选头部槽。导出只取前29个身体值，再从独立夹爪遥测补两个真实电机角；**不能直接照搬原始31维向量**。头部反馈缺失不拒绝导出，身体反馈缺失才拒绝。

## 6. 每个meta文件做什么

JSON是一个完整对象；JSONL是UTF-8文本，**每行一个独立JSON对象**，行间不写逗号、外面不套数组。`.jsonl`可用文本编辑器、`head`直接查看；Parquet要用读取工具。

|文件|内容与用途|
|---|---|
|info.json|版本、robot_type、fps、总条数/帧数/任务数/视频数、chunk大小、train分割、Parquet/MP4路径模板；features逐项声明dtype、shape、维度名称及视频编码/分辨率/fps|
|modality.json|GR00T对列的解释：哪个列的哪个切片是左臂/右爪；视频键映射；task_index如何对应语言|
|stats.json|各数值维度的mean/std/min/max/q01/q99；供GR00T按本数据归一化，不能套旧开度坐标统计|
|episodes_stats.jsonl|v2.1逐episode统计；每行episode_index＋stats，含计数和各列统计、RGB通道统计|
|episodes.jsonl|每行一个训练episode的编号、任务列表、长度|
|tasks.jsonl|每行一个task_index与真实任务文字的对照|
|source_map.json|训练episode对应哪个原始录制、源行号/时间、推理序号、暂停段、丢弃行原因、文件SHA256、夹爪身份和相机信息|
|native_1d_derivation.json|已有native build_processor入口要求的来源说明；标记本次来自真机motor rad，指向source_map和contract，并非仿真转换|
|elf3_contract.json|本项目约定：joint_names、34/66维、40步、频率、token模型指纹、绝对电机rad语义、图像处理、暂停去除与同条合并策略|

`stats.json`是当前GR00T读取的整集统计；`episodes_stats.jsonl`补齐LeRobot v2.1逐条统计结构。后者本轮已补写；这不等于已经跑过整个LeRobot SDK的所有功能。本轮实际跑的是已有ELF3适配GR00T的数据加载、视频解码、训练取样和归一化。

info的关键片段（完整文件另含所有features、计数和拆分字段）：

```json
{
  "codebase_version": "v2.1",
  "robot_type": "elf3_sonic_native_1d",
  "fps": 50,
  "chunks_size": 1000,
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
}
```

episodes.jsonl示例：

```jsonl
{"episode_index":0,"tasks":["把桌上的盒子放入收纳箱"],"length":500}
{"episode_index":1,"tasks":["把桌上的盒子放入收纳箱"],"length":400}
```

tasks.jsonl示例：

```jsonl
{"task_index":0,"task":"把桌上的盒子放入收纳箱"}
```

多个episode可以引用同一task_index。不要拿“test”或“diagnostic”替代实际训练指令。

### modality.json和modality configuration不是同一个文件

- **数据集内`meta/modality.json`**回答“数据在哪里”：例如`action.left_gripper`取`action.wbc[29:30]`；`state.left_arm`取`observation.state[15:22]`。
- **训练侧`elf3_native_1d_config.py`**回答“模型怎么取”：选三路视频/哪些状态、拼接顺序、当前帧索引[0]、动作索引[0…39]、ABSOLUTE表示和embodiment注册。它通常不放在每份数据集里，而随训练代码交付。

两份文件都已在本库`tools/gr00t_dataset/`，沿用本地原配置；`modality.json`由导出器复制到每个数据集。ABI JSON另记单位/维度/频率约定，它也不是modality configuration。参见[官方配置说明](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md)。

## 7. 暂停与当前尚未解决的事项

A+X切出/进入POSE仍是同一条原始任务，不自动保存结束。恢复后继续采；保存时才结束。导出时按有效连续段生成训练episode，用source_map保留它们属于同一任务的关系，防止40步窗口横跨几秒暂停。训练/验证分组也应按原始任务分，避免同条两段落到两边。

当前还有四个具体事项：

1. **新采集模式尚未真机验收。** 本地sonic50_latest_rgb已支持真实推理时刻与30Hz画面关联，离线通过；现场冻结r7仍30Hz相机取行。部署后应核对实际有效频率和图像年龄，不能把旧30Hz记录当作已含全部50Hz动作。
2. **三路画面和有效行仍待本轮现场验收。** 要确认左右视角、实际尺寸、连续帧率与对时；设备出现在/dev/video里不等于已采到有效数据。
3. **真机电机角的数据/模型要单独标识。** 原开度数据不能混用；重新计算统计，训练所得checkpoint携带相同单位约定。新VLA推理接入真实双爪的单位边界仍需验证。
4. **任务与有效连续段要正确。** 保存真实任务文字，只使用live且token/身体目标匹配的数据，暂停/断流不补造动作；需要有实际有效任务可供训练。

本轮已解决：离线导出不再要求电机→开度映射；状态/标签保存原始motor rad；元数据明确单位，旧仿真语义会被本工具拒绝；补齐逐条统计文件。**不要求再次标定夹爪开度曲线。**

## 8. 定位任意一项数据

以下仅离线读文件，使用已有Isaac-GR00T环境中的pyarrow：

```python
from pathlib import Path
import json
import pyarrow.parquet as pq

root = Path("/实际训练数据目录")
info = json.loads((root / "meta/info.json").read_text())
episode, frame_index = 0, 10
table = pq.read_table(root / info["data_path"].format(
    episode_chunk=episode // info["chunks_size"], episode_index=episode))
print(table.schema)
row = table.slice(frame_index, 1).to_pylist()[0]
print("时刻(秒):", row["timestamp"])
print("左肩Y实测(rad):", row["observation.state"][15])
print("双爪实测(rad):", row["observation.state"][29:31])
print("双爪目标(rad):", row["action.wbc"][29:31])
print("token64:", row["action.motion_token"])
tasks = {x["task_index"]: x["task"] for x in
         map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
print("任务:", tasks[row["task_index"]])
print("头部视频第", frame_index, "帧:", root / info["video_path"].format(
    episode_chunk=episode // info["chunks_size"], episode_index=episode,
    video_key="observation.images.ego_view"))
```

库的用途和文件入口见[工作站目录导引](WORKSPACE_GUIDE_20261020.md)；当前原始JSONL字段见[原始采集说明](COLLECTION_DATA_FORMAT.md)；导出命令与验证范围见[工具README](../tools/gr00t_dataset/README.md)。

任务文字的填写位置、例子及保存过程见[任务标注说明](TASK_LABELING.md)。


## 2026-10-20 官方写入器接入

训练字段和既有modality不变；Parquet、MP4、info/tasks/episodes/episodes_stats现由官方SONIC的Gr00tDataExporter及LeRobot生成，stats.json由GR00T官方函数计算。官方SONIC的episodes_stats只含数值字段，不含图像均值方差。原始JPEG/JSONL仍用于ELF3数据取数与审计，保存后在工作站导出。见[复用范围及补丁](OFFICIAL_COLLECTION_REUSE_20261020.md)。
