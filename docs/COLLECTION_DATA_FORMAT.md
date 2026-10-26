> 2026-10-22当前版本：BXI demo04 / 数采v16，夹爪开口0～70mm；旧角度参数和v15验证记录仅供历史追溯。使用[毫米协议与合同说明](GRIPPER_MM_PROTOCOL_20261022.md)。新版本尚未真机验证。

> 此文说明原始JSONL/JPEG。新增训练成品格式见[GR00T N1.7数据字典](GR00T_N17_DATA_FORMAT.md)；原始的training_export_allowed=false仍保留，不原地篡改旧合同。新工具经过另行验证后生成独立训练目录。

# 真机任务数据文件说明（2026-10-20）

## 当前客户版与现场冻结r7的区别

下文原始字段清单仍适用；其中头图锚点、30Hz、图像不复用等描述专指历史`head30`模式。当前客户v5新增`sonic50_latest_rgb`，未部署到机器人：

- 数值锚点改为实际live推理时间，每次推理最多一行，名义50Hz；相机仍30Hz。
- 三路画面各取锚点之前最近的有效图像，最多100ms，可复用。`auxiliary.camera_reused`及`camera_source_age_s`分别记录复用标记和图像年龄，原始序号、时间保留。
- `auxiliary.anchor_stamp_unix`对应推理源时间；`meta.timebase.row_clock=sonic_inference_source_stamp`，`row_fps=50`。暂停后不能沿用暂停前画面。
- 原始`observation.state`仍是接口原始数组，训练导出器选身体29维，再接左右真实夹爪电机rad组成31维。重力另占3维，模型状态共34维；不采头部训练自由度。
- 原始诊断合同不因配置50Hz自动变成训练验收通过。独立导出器验证实际时间与各项源数据后生成训练目录。
- 任务文字的设置和保存见[任务标注说明](TASK_LABELING.md)，新训练列和文件见[数据字典](GR00T_N17_DATA_FORMAT.md)。

## 现场冻结r7 / head30原始字段与规则

本文对应库内 `tools/pico_collection/sonic_pico_collection_20261019_r7` 的实际启动配置与写入代码。以下为现场冻结r7 / head30的格式；客户v5差异见上节。**当前保存的是三路 JPEG＋逐帧 JSONL 原始数据，不是已经转换好的 LeRobot 训练包。** schema 为 `elf3_three_rgb_gripper_source/v2`，`diagnostic_only=true`、`training_export_allowed=false`。头部深度不采集；三路彩色、身体状态/指令、在线 motion_token、PICO/SMPL、两爪遥测必须满足当前同步规则，才形成一行。

训练格式另见[既有Native 1DoF字典](GR00T_N17_DATA_FORMAT.md)：身体29＋双爪2，共31自由度，头部关节不参与。下文31关节原始数组中的末两项只是旧诊断接口可选槽位，不能直接作为训练31向量。现行r7的30Hz和电机角记录尚不等于50Hz native训练数据。

## 1. 文件位置及生命周期

现场默认路径如下。客户版可在 `tools/customer/site.example.json` 的副本中指定 `runs_root`、`data_root`；留空则使用当前用户家目录下同名目录。

```text
~/pico_collection_runs/
  status.json                         # 监听器状态、当前session、上一次结果
  session_XXXXXX/
    intent.json                       # 操作序号、开始/保存/弃用/暂停/恢复、暂停边界
    trial.yaml                        # 本条实际使用的全部配置
    session.log                       # 本条采集器输出
    camera_left.log / camera_right.log
    session_status.json               # 当前有效行数、缺失项、暂停状态等
    session_result.json               # 某些中断/弃用路径写入的结果，非必有
    launcher_status.json              # 启动/结束结果、saved_data_checks读回检查
~/pico_collection_data/
  session_XXXXXX/three_rgb_gripper_source/
    episode_0000/
      meta.json                       # 整条数据的合同、配置指纹和结束结果
      frames.jsonl                    # 一行JSON = 一组按时间对齐的数据
      head_rgb/rgb/000000.jpg
      left_wrist_rgb/rgb/000000.jpg
      right_wrist_rgb/rgb/000000.jpg
      ...                             # 文件名与行idx一一对应
```

录制中的目录以 `.episode_0000.inprogress` 命名；JPEG由写入线程处理，`frames.jsonl` 在结束时落盘。**看到目录不等于保存完成**，应等保存、读回检查结束。弃用保留在 `_discarded/`，中断保留在 `_interrupted/`，并标记排除训练。当前检查结果在 `launcher_status.json.saved_data_checks`，不是旧说明中的独立 `audit.json`。

A+X切出POSE后不写行；切回有效POSE继续原session、原episode。`idx`连续，`t_episode`保留暂停造成的真实时间间隔；`auxiliary.capture_segment`增加。训练导出按官方清理后的方式将有效行相接，重建连续时间，暂停前后仍为同一训练episode；原始时间仅保留供追溯。

## 2. 如何精确取一个数据项

`frames.jsonl` 中带点的名字是**完整的顶层键名**，不是嵌套路径。例如 `row['observation.state'][15]` 是左肩Y角，`row['gripper']['state'][1]` 是右夹爪电机绝对角。图片位置是 `episode目录 / row['observation.images']['head_rgb']`。

```python
from pathlib import Path
import json, math
p = Path('/替换为实际数据目录/episode_0000')
with (p / 'frames.jsonl').open() as f:
    row = json.loads(next(f))
print('行号、时间、片段:', row['idx'], row['t_episode'], row['auxiliary']['capture_segment'])
print('左肩Y角度:', math.degrees(row['observation.state'][15]))
print('左/右夹爪角度:', [math.degrees(x) for x in row['gripper']['state']])
print('头部图片:', p / row['observation.images']['head_rgb'])
```

## 3. frames.jsonl 顶层字段（完整清单）

数字向量在JSON中是数组，没有 NumPy dtype 包装。尺寸及顺序由下表、`meta.joint_contract`确定。

| 键 | 类型/尺寸 | 内容、单位 |
|---|---|---|
| `diagnostic_only` | bool | 当前固定true，尚非训练坐标合同 |
| `idx` | int | 本episode保存行号，从0开始 |
| `t_episode` | float | 本行头图源时间减首行头图源时间，秒，包含暂停间隔 |
| `t_wall_episode` | float | 写入时刻相对录制开始的墙钟时间，秒 |
| `t_unix` | float | 组装本行的Unix时间，秒；不是相机曝光时刻 |
| `anchor_sequence` | int | 头部图像接收序号，可能不连续 |
| `observation.state` | float[31] | 实测关节位置，rad，身体29＋头部2 |
| `observation.velocity` | float[31] | 实测关节速度，rad/s |
| `observation.effort` | float[31] | 硬件消息报告的关节力度；保留驱动语义，未做独立力矩标定 |
| `observation.driver_temperature` | float[31] | 驱动温度，°C；结合对应present掩码 |
| `observation.motor_temperature` | float[31] | 电机温度，°C；结合对应present掩码 |
| `observation.joint_present` | bool[31] | 本行该关节是否确实存在 |
| `observation.driver_temperature_present` | bool[31] | 驱动温度是否真实提供 |
| `observation.motor_temperature_present` | bool[31] | 电机温度是否真实提供 |
| `observation.projected_gravity` | float[3] | 按SONIC身体/IMU约定计算的重力方向，单位向量 |
| `dataset.observation.state` | null | 原始诊断合同保留null；训练导出器另行生成身体29＋真实电机双爪2 |
| `observation.imu` | object | 见下文IMU字段 |
| `action` | float[31] | 最终 ActuatorCmds 位置指令，rad；不是实测角 |
| `action.velocity` | float[31] | 最终速度指令，rad/s |
| `action.torque` | float[31] | 最终前馈力度，沿用硬件接口单位 |
| `action.kp` / `action.kd` | 各float[31] | 最终位置/速度增益，沿用控制接口定义 |
| `action.joint_present` | bool[31] | 指令中是否含该关节 |
| `dataset.action.wbc` | null | 同上，尚未填训练坐标 |
| `action.motion_token` | float[64] | **同一次在线推理**产生的post-FSQ decoder输入；不是29维关节指令 |
| `sonic.inference` | object | 模型指纹、序号、状态及推理目标，见下文 |
| `sonic.effective_continuous_action_29` | float[29] | `(最终身体指令-default_position)/action_scale`反算连续动作，非motion_token |
| `sonic.effective_continuous_action_history_10x29` | float[10][29] | 上述连续动作的10步历史；不是SMPL未来帧窗口 |
| `pico.raw` | object | 对齐到本行的PICO网络包，字段见下文 |
| `pico.smpl_ref` | object | 对齐到PICO帧号的SMPL参考网络包，字段见下文 |
| `state_machine` | object | 源状态机JSON原样保存；当前状态在`current.name`，其他字段随宿主版本变化 |
| `motion_command` | null | 当前SourceCollectionNode没有填此项 |
| `auxiliary` | object | 对齐、时钟、片段及选择证据，见下文 |
| `gripper` | object | 两爪实测/最终指令及原始遥测，见下文 |
| `observation.images` | object | `head_rgb`、`left_wrist_rgb`、`right_wrist_rgb`三个相对JPEG路径 |
| `observation.depth` | null | 当前不记录深度 |
| `source_timestamps_unix` | object | 各源Unix时间（秒），见下文 |
| `source_receive_offsets_s` | object | 各源接收时间相对头图锚点的差值（秒） |

缺失的可选头部值可能填0，但掩码为false，不能把0当测得的角度。身体29关节缺失会拒绝此行。关节轴是URDF关节坐标，不能把人体SMPL轴角逐列当作机器人关节角。

### 31维逐索引表

| 索引 | 名称 | 索引 | 名称 |
|---:|---|---:|---|
|0|waist_y_joint|1|waist_x_joint|
|2|waist_z_joint|3|l_hip_y_joint|
|4|l_hip_x_joint|5|l_hip_z_joint|
|6|l_knee_y_joint|7|l_ankle_y_joint|
|8|l_ankle_x_joint|9|r_hip_y_joint|
|10|r_hip_x_joint|11|r_hip_z_joint|
|12|r_knee_y_joint|13|r_ankle_y_joint|
|14|r_ankle_x_joint|15|l_shoulder_y_joint|
|16|l_shoulder_x_joint|17|l_shoulder_z_joint|
|18|l_elbow_y_joint|19|l_wrist_x_joint|
|20|l_wrist_y_joint|21|l_wrist_z_joint|
|22|r_shoulder_y_joint|23|r_shoulder_x_joint|
|24|r_shoulder_z_joint|25|r_elbow_y_joint|
|26|r_wrist_x_joint|27|r_wrist_y_joint|
|28|r_wrist_z_joint|29|head_z_joint|
|30|head_y_joint|—|—|

### IMU及推理结果

- `observation.imu.orientation_wxyz`：4元数，顺序WXYZ；`angular_velocity`：3维rad/s；`linear_acceleration`：3维m/s²；`frame_id`：原ROS坐标标识。这不是独立外部动捕真值。
- `sonic.inference.sequence`：整数推理序号；`model_id`、`model_sha256`：模型标识和完整SHA256；`policy_status`：有效行要求`live_reference`；`target_position`：31维rad（按同一标准顺序）；`source`：`same_run_online_post_fsq_decoder_input`。
- 当前token为64维32级FSQ网格数据；不要重命名连续29维动作来冒充它。默认姿态、action_scale和token校验见 `collector/real_robot_collection/contracts.py`。

### 夹爪字段

`gripper.state[0/1]`及`gripper.command[0/1]`分别是左/右电机的实测绝对角与最终线上位置指令，单位rad；换算为度乘180/π。用户使用的闭合0°、打开300°对应0和约5.235988rad，**没有另改零点**。

| `gripper`子键 | 内容 |
|---|---|
| `coordinate` | `motor_rad_diagnostic_only` |
| `mapping.verified` | false：电机角还未转换为训练模型中的夹爪关节坐标 |
| `trigger_intent` | 左/右扳机深度[2]或null；0松开、1按下 |
| `state` / `state_is_measured` | 左右实测角[2] / true |
| `command` / `command_is_final` | 左右最终线上目标角[2] / true |
| `native_raw_sample` | 原始双爪遥测对象，结构如下 |

`native_raw_sample`顶层包含：`schema`(1)、`monotonic`(发布单调时钟秒)、`control_monotonic`(控制循环单调时间)、`stamp.sec/nanosec`(ROS发布时间)、`training_eligible`、`ineligible_reasons`(字符串数组)、`coordinate`、`samples`(左、右两个对象)。此处`training_eligible`只表示遥测当时满足采集前提，**不覆盖整条数据的training_export_allowed=false**。

每个`samples[i]`含：

- 身份：`side`、`bus`、`motor_id`、`protocol`、`profile`、`position_unit`、`effort_unit`。当前左CAN5、右CAN6，ID均1。
- 反馈：`position_rad`、`velocity_rad_s`、`torque_nm`、`feedback_monotonic`。力度单位名是协议标签，物理力矩未独立标定；本遥测不保存夹爪温度。
- 范围：`open_rad`、`closed_rad`、`closure_fraction`、`command_closure_fraction`。闭合比例按`(位置-open)/(closed-open)`计算，未强制截断，也不是训练几何关节坐标。
- 操作者：`trigger`、`trigger_monotonic`。
- `control.active`：本侧当前是否跟随新输入；`control.mode`：当前BXI为`tracking`或`input_hold`，DM可能有其他模式。
- `command.position`、`velocity`、`kp`、`kd`、`feedforward`：从最终CAN数据解回的五个指令参数；`can_data`：8个0–255整数；`sent_monotonic`：软件发送时间。增益可能因量化略不同于配置25/0.25。

### PICO及SMPL字段

以下是当前POSE包字段；N通常为10，直接保存网络包数组，后续发送端扩展键也会保留。

| `pico.raw`子键 | 形状 | 含义 |
|---|---|---|
| `smpl_pose` | N×21×3 | 人体局部姿态轴角，rad，不是机器人关节角 |
| `smpl_joints` | N×24×3 | SMPL局部关节位置，m |
| `body_quat_w` | N×4 | 人体根旋转，WXYZ，已转换到机器人参考约定 |
| `joint_pos` | N×29 | PICO处理链路生成的身体关节参考，rad；不等于策略最终指令 |
| `head_joint_pos` | N×2 | 头部参考角，rad |
| `joint_vel` | N×29 | **当前发送端填零的占位量，不是实测速度** |
| `vr_position` | 9 | 左腕、右腕、颈部位置依次展开，机器人参考系，m |
| `vr_orientation` | 12 | 左腕、右腕、颈部四元数依次展开，各WXYZ |
| `frame_index` | N | 发送端连续参考帧号 |
| `left_trigger/right_trigger` | 各1 | 扳机深度 |
| `left_grip/right_grip` | 各1 | 握把输入 |
| `pico_dt/pico_fps` | 各1 | 输入间隔秒/频率Hz |
| `timestamp_realtime/timestamp_monotonic` | 各1 | 源采样墙钟/单调时间，秒 |
| `left_hand_joints/right_hand_joints` | 各7 | 发送端手部构造值；非真实机器人手部反馈 |
| `start_data_collection/finish_data_collection` | 历史可选bool[1] | 仅旧原始记录可能包含；当前版本不发出也不消费这两个采集别名 |
| `heading_increment` | 1 | 航向增量，rad |
| `stream_mode` | int[1] | 当前POSE=1 |
| `calibration_ready` | bool[1] | PICO人体标定完成 |

`pico.smpl_ref`含：`source_chunk[1]`、`source_stream_epoch[1]`、`source_received_monotonic_ns[1]`、`frame_index[N]`、`term1_local[N][72]`(24关节局部XYZ展开)、`root_quat[N][4]`(WXYZ)、`wrist[N][6]`、`head_joint_pos[N][2]`、`valid_horizon[1]`、`clamp_slots[1]`。

**这里是源包，不保证等于某次策略实际消费的1770维encoder输入。** 要严格核对消费帧、重建参考和跟踪误差，仍使用`tools/tracking_eval`的`trace`记录；不要从任务JSONL推断未保存的消费细节。

### 对齐和时钟证据

`auxiliary`逐项：

| 键 | 含义 |
|---|---|
| `anchor_stamp_unix` | 本行头图源header时间，Unix秒 |
| `control_tick_monotonic` | 锚点减clock_offset_s的派生单调时间；不是直接采到的控制周期时间 |
| `capture_segment` | 暂停恢复分段号，从0递增 |
| `selection_monotonic` | 选择这一组数据时的本机单调时间 |
| `clock_offset_s` | Unix墙钟－单调时钟偏移 |
| `gripper_ingest_method` | `source_history_v2` |
| `gripper_publication_to_callback_s` | 双爪遥测发布到采集回调延迟 |
| `source_delta_to_anchor_s` | 每个源时间减锚点时间，键同source_timestamps_unix |
| `received_monotonic` | 各源本机接收时间，键见下文 |
| `selected_camera_sequences` | 三路图像各自接收序号 |
| `pico_smpl_frame_index` | 这组PICO/SMPL配对的末帧号 |
| `smpl_timestamp_basis` | `derived_from_exact_pico_frame_index` |
| `gripper_feedback_time_basis` | `CAN_receive_monotonic_bridged_by_same_sample_publish_clocks` |
| `state_machine_time_basis` | `collector_receive_monotonic` |

`source_timestamps_unix`的完整键：`actuator_state`、`actuator_command`、`imu`、`sonic_inference`、`gripper_state`、`gripper_command`、`pico`、`smpl_ref`、`head_rgb`、`left_wrist_rgb`、`right_wrist_rgb`、`left_gripper_feedback`、`left_gripper_command`、`right_gripper_feedback`、`right_gripper_command`。

`auxiliary.received_monotonic`的键：`actuator_state`、`actuator_command`、`imu`、`sonic_inference`、`gripper`、`pico`、`smpl_ref`、`head_rgb`、`left_wrist_rgb`、`right_wrist_rgb`。`source_receive_offsets_s`使用同样键加`_minus_anchor`后缀，值为接收单调时间减派生锚点单调时间。

SMPL源时间借助严格匹配的PICO帧号派生；夹爪反馈是CAN接收时间换算；并非每个时间都是设备硬件曝光/采样真值。重现分析时保留全部时间字段，不要只留`t_unix`。

## 4. meta.json逐组索引

- `schema_version`、`episode_id`、`mode`、`task_name`、`task_id`、`fps`：数据格式、任务和名义帧率。
- `start_time_utc/start_time_unix`、`end_time_utc/end_time_unix`、`duration_s`、`wall_duration_s`、`num_frames`、`actual_row_rate_hz`：起止和统计；部分字段仅成功结束路径出现。
- `metadata`：公共配置、任务、模型运行信息合并字典；当前包括`robot_serial/operator_id/site`、`camera_calibration_status`、`gripper_identities`、`collector_release`、`camera_left_right_verified/physical_ports/camera_role_evidence`、`notes`及`motion_token`模型信息。额外键按任务配置保留。
- `configuration.path/sha256`：生成的trial.yaml路径与SHA256；此绝对路径是采集溯源信息，读数据不需要该路径仍存在，建议随数据一起保留trial.yaml。
- `joint_contract.canonical_names/dimension/missing_value/missing_value_is_valid_only_when_presence_mask_is_false`：上述31关节顺序与缺省规则。
- `diagnostic_action_contract.name/dimension/history/source/is_training_motion_token`：反算连续动作的29维/10历史说明。
- `camera_contract.rgb/depth/rgb_storage/depth_storage/mounts`：三路相机身份、分辨率、topic、配置来源。depth_storage是通用声明，本模式没有深度文件。
- `timebase`：`row_clock=head_rgb_source_header`、`timestamp_rule=head_source_stamp - first_saved_head_source_stamp`、`row_fps=30`、`row_fps_is_nominal_only=true`、`uniform_training_timebase_verified=false`、`wall_clock_is_diagnostic_only=false`、`camera_alignment=buffered_nearest_source_header_within_25ms_no_reuse`、`source_header_stamps_preserved=true`、`source_receive_offsets_preserved=true`。
- `diagnostic_only/training_export_allowed/missing_modalities/gripper_coordinate/gripper_identity_contract`：诊断格式、禁止直接训练导出、缺head_depth、电机角及左右身份合同。本模式**没有**通用录制器的`published_dataset_contract`。
- `synchronization`：同步选择统计、拒绝原因及`gripper_ingest.invalid_intervals`；`pose_pauses`记录Unix秒的暂停区间，`pause_semantics`解释处理方式。
- `termination`：success、reason及结束检查；异常/弃用可有accepted/saved行数和writer_errors。`incomplete/discarded/excluded_from_training`仅相应路径添加。

需要原样保留完整meta，不能只用`num_frames>0`判定成功。

## 5. 什么是有效行；为什么有输入却没有保存行

当前按头图时间选择一组数据：三路RGB不复用且在±25ms内；身体状态/指令/IMU/同次推理/双爪遥测按±30ms配对；各爪实际反馈与命令在±50ms内；PICO包及SMPL末帧号严格一致，PICO源数据不得落后超过200ms；在线推理必须是live_reference且身体目标与最终指令匹配。还需已标定POSE、SONIC状态及同一暂停片段。

缺相机、某侧爪没有合格遥测、暂停或同步不满足时，PICO仍可以有输入，但不会凑一条缺项的有效行。单爪能够控制不等于当前双爪采集条件满足。当前头图1920×1080@30，腕图各640×480@30；JPEG质量90；最少100有效行。30Hz是名义配置，真实有效行率可能更低。

## 6. 与提供的仿真数据对照

实读样例：`BXI_whole_body_vla_dataset_isaac/datasets/elf3_box_carry_50eps_20260731_193249_lerobot_v21.tar.zst`中的`meta/info.json`、`meta/modality.json`、`meta/tasks.jsonl`及一个parquet。该样例50条、49695帧、150视频；其他仿真归档请以各自info为准。参考元数据副本见本目录`simulation_format_reference/`。

| 项目 | 当前真机r7 | 本次仿真样例 |
|---|---|---|
| 格式 | 原始JSONL＋JPEG | LeRobot v2.1：Parquet＋H.264 MP4 |
| 表格路径 | `episode_0000/frames.jsonl` | `data/chunk-000/episode_000000.parquet` |
| 视频/图像 | 每行三个相对JPEG路径 | `videos/chunk-000/{video_key}/episode_000000.mp4`；metadata另声明JPEG镜像 |
| 帧率/时间 | 名义30Hz，真实头图时间，暂停有缺口 | 50Hz，timestamp=frame_index/50 |
| 实测state | 身体29＋头2=31，爪另存2电机角 | `observation.state`=身体29＋左爪2＋右爪2=33，float64 |
| 指令 | `action`31，爪另存2；`dataset.action.wbc=null` | `action.wbc`33，float64 |
| motion_token | `action.motion_token`64，在线同次推理 | `action.motion_token`64，float64 |
| 重力 | `observation.projected_gravity`3 | 同名float64[3] |
| 图像名/尺寸 | head_rgb1920×1080；左右腕640×480 | ego_view/left_wrist/right_wrist，均640×480×3 |
| 行元数据 | idx、t_episode、capture_segment及全部源时钟 | timestamp(float32)、frame_index/episode_index/index/task_index(int64) |
| 任务描述 | meta.task_name/task_id | meta/tasks.jsonl，通过task_index定位 |
| 能否直接统一训练 | 不可直接混用；新真机目标采用电机rad，见新数据字典 | 已组织成该仿真数据合同 |

仿真33维切片：[0:3]腰，[3:9]左腿，[9:15]右腿，[15:22]左臂，[22:29]右臂，[29:31]左爪两个关节，[31:33]右爪两个关节。注意Python右边界不包含。该表对照旧仿真数据，不是新真机训练目标。新目标只保留每爪一个真实电机角，不向这四个仿真坐标转换，见[真实电机角训练数据字典](GR00T_N17_DATA_FORMAT.md)。

## 7. 代码出处

以下均相对仓库根：

- `tools/pico_collection/sonic_pico_collection_20261019_r7/configure.py`：实际运行配置。
- 同目录`collector/real_robot_collection/recorder.py`：meta及完整逐行写入字段。
- 同目录`collector/real_robot_collection/source_node.py`、`source_sync.py`：对齐源及具体时间含义。
- 同目录`collector/real_robot_collection/contracts.py`、`wrist_diagnostic.py`：关节、token及诊断合同。
- 同目录`session_node.py`、`session_protocol.py`、`verify_saved.py`：暂停片段、完成/弃用和读回检查。
- 根目录`collection_telemetry.py`：在线推理和原始双爪遥测；`bxi_demo.py`：当前BXI控制状态。
- `pico/gear_sonic/scripts/pico_manager_thread_server.py`、`pico/pose_to_smpl_ref_bridge.py`：PICO、SMPL包字段。

字段列表另有机器可读文件 `COLLECTION_FIELDS.json`，用于按JSON路径检索。本文不宣称已完成三路相机齐全、两段有效数据同条保存的现场验收；摄像头人工测试完成后再验证。
