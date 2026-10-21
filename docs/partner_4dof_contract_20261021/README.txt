ELF3 / SONIC / GR00T N1.7 合作方训练数据接口合同
2026-10-21，四夹爪自由度草案；请先确认关节表，再用于正式交付验收。

1. 适用范围
假设合作方身体仍采用ELF3的29个身体关节、同一SONIC的64维motion token。
用户已说明左右夹爪各2自由度，共4；它们是否独立驱动、单位/零位/方向/范围尚待确认。
本合同沿用已有ELF3四坐标数据布局，不另造存储格式；但不能照搬旧“驱动+联动”语义。
当前test-01只有左右各1个真实电机自由度；它的数据不能通过复制变成合作方四自由度实测数据。

2. 文件清单
- data_contract.json：字段、维数、频率、关节物理映射待填项和验收要求。
- modality.json：沿用已有ELF3四坐标切片布局；不是模型训练配置脚本。
- JOINT_ORDER.txt：完整33个坐标顺序。最后4个名字为槽位占位名，须绑定真实关节。
- tasks.example.jsonl / episodes.example.jsonl：格式示例，不是真实采集记录。
- provenance.json / VALIDATION.json：来源与结构核验；不代表已用四维真机数据训练验证。

3. 一行数据是什么
observation.state：float32[33]，同一时刻的实测位置，身体29+左夹爪2+右夹爪2。
observation.projected_gravity：float32[3]，机器人基座系下单位重力向量。
action.motion_token：float32[64]，该次SONIC推理真实产生、输入decoder的post-FSQ token。
action.wbc：float32[33]，实际发出的目标位置，身体29+左夹爪2+右夹爪2。
三路observation.images：ego_view、left_wrist、right_wrist，训练解码RGB uint8[480,640,3]。
timestamp：float32，训练episode内秒数；frame_index / 50。
frame_index、episode_index、index、task_index：int64，分别为条内帧号、条编号、全局行号、任务编号。

observation是“机器人实际到了哪里”，action是“当时要求它去哪里”。二者不能互相替代。
模型真正学习的action拼接为[token64,left_gripper2,right_gripper2]=68维；
action.wbc前29维用于保存/核对身体目标，不再额外拼进模型的68维动作。
模型state拼接为[body29,left_gripper2,right_gripper2,gravity3]=36维。
头部两个自由度不进入本合同的state/action。

4. 坐标定义
身体使用JOINT_ORDER中的真实关节rad。夹爪建议同样使用真实绝对关节/电机rad；
若合作方第二自由度是直线位置、开度或联动坐标，应先确认物理定义，再修改合同和训练配置。
单位转换只做deg→rad等明确物理换算，不映射到我们旧仿真开度，不取两指均值，不生成[q,q]伪实测。
每个夹爪坐标必须分别确认：真实名称、独立性、传感器来源、目标命令来源、零位、正方向、上下限。
我方0°/300°是我方夹爪电机设置，不自动成为合作方四个坐标的范围。

5. 切片：左闭右开
身体：[0:29]；左夹爪：[29:31]；右夹爪：[31:33]。
modality.json的action.motion_token取自己的[0:64]；夹爪从action.wbc取上述切片。
state.projected_gravity从observation.projected_gravity取[0:3]。
训练horizon沿用40步，50Hz下跨度配置约0.8秒；状态/图像各取当前一步。
当前ELF3_NATIVE_1D是34维state/66维action，不能直接接本合同的36/68。
旧ELF3的36/68布局可参考，但旧夹爪是驱动+联动，维数相同不代表归一化统计和模型权重可直接用。
embodiment tag/processor注册由双方训练环境确认；本包没有擅自覆盖已有tag或训练配置。

6. 频率和相机
本项目训练行50Hz；这不是GR00T的通用强制标准。
数值来自真实推理，不复制token/动作补齐缺帧。断流、暂停切分为独立连续训练episode。
物理相机可以30Hz；每条数值配对应时刻最新有效图像，训练视频按50fps与数据行一一对应，允许重复图像。
三路训练视频均640×480，RGB语义，MP4/H264/yuv420p；保持全视野等比例缩放并补黑边。
原始相机分辨率可以不同，必须保留源时间戳和相机身份，不能左右互换。
本机当前图像有效期上限100ms；更改应在合同版本和数据来源中明确记录。
原始记录保留真实时间/暂停边界；训练timestamp按每个连续片段重新从0排列，不把暂停前后硬连成连续动作。

7. LeRobot v2.1文件组织
DATASET/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.ego_view/episode_000000.mp4
  videos/chunk-000/observation.images.left_wrist/episode_000000.mp4
  videos/chunk-000/observation.images.right_wrist/episode_000000.mp4
  meta/info.json
  meta/modality.json
  meta/episodes.jsonl
  meta/tasks.jsonl
  meta/episodes_stats.jsonl
  meta/stats.json
  meta/source_map.json

Parquet是列式文件；逻辑上一行对应一个采样时刻，上述数值向量和索引各为一列。像素在MP4，不塞入数值向量。
视频通过info中的路径模板、episode编号和帧号匹配。
info记版本/fps/features/总条数/总帧数/路径模板；modality记每个模型部件从哪个列取哪段。
JSONL每一行是一个完整JSON对象；episodes每条一行记录编号、任务、length，tasks记录编号和任务文字。
任务例：“把桌上的红色盒子放入右侧托盘”。录后可按真实示范补写；不能把示例/diagnostic占位当真实标签。
stats包含真实采集数据各坐标的统计及q01/q99；不得沿用旧仿真开度的统计。
source_map把同一原始任务的多个连续训练片段关联起来，不是额外模型输入。

8. 接收验收
(1) 填完data_contract.json的gripper_mapping，确认身体/SONIC模型token定义一致。
(2) 所有数值有限、维数正确、顺序固定；实测与命令来源分明。
(3) 三视频全部解码，帧数等于对应episode的Parquet行数；没有缺视角/左右错配。
(4) 连续片段按50Hz真实数值构建；40步窗口不跨A+X暂停或断流；缺行不能仅改fps。
(5) 明确排除弃用/失败条；不足40步的片段不能独立构成完整40步训练窗口。
(6) 使用合作方实际GR00T N1.7 loader检查state[1,36]和action[40,68]、RGB三视角、任务文字。
(7) 用实际夹爪范围重算统计并做归一化往返检查，再开始训练。
目前只完成接口结构和维数核验，尚无合作方四自由度真机样本、训练loader验收或推理执行验收。

参考
官方采集：https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html
官方训练工作流：https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html
项目写入器来源与提交见provenance.json。无需向合作方提供我们工作站上的绝对路径或机器人IP。
