ELF3真机66维训练数据合同（2026-10-21 v2，同条合并）

本包直接交付我们当前使用的单夹爪一个电机自由度接口，合作方据此适配自己的机器人。
未给合作方扩展到四自由度，也不要求合作方采用我们机器人的CAN口接线。
66是模型动作维数，不是电机数；身体29+左右电机2=31个位置坐标，无头部关节。
夹爪使用真实绝对电机角度rad，不转换成仿真开度，不用0~1比例替代。

1. 模型输入和输出
输入：三路RGB + state[1,34] + 真实任务文字。
state顺序：身体29、左爪实测电机角1、右爪实测电机角1、基座系投影重力3。
输出/监督：action[40,66]，每步为token[0:64]、左爪目标角[64:65]、右爪目标角[65:66]。
SONIC在线post-FSQ token64由同一模型产生；SONIC解码输出身体29，两个夹爪目标绕过SONIC直接控制电机。
训练采样50Hz、窗口40步，当前执行默认20步；50Hz是本项目约定，不是GR00T通用强制要求。

2. Parquet精确字段（每行一个时刻，共9列）
observation.state             float32[31] 实测身体29+实测左右爪电机2
observation.projected_gravity float32[3]  基座系单位重力向量
 action.motion_token          float32[64] 本次真实在线推理的post-FSQ token
 action.wbc                   float32[31] 实际下发身体目标29+左右爪目标2
 timestamp                    float32     清理后训练episode内frame_index/50秒
 frame_index                  int64       条内帧号，从0开始
 episode_index                int64       训练条编号
 index                        int64       全数据集行号
 task_index                   int64       任务文字索引
上述action列名前无空格，缩进仅为本说明排版。
Parquet是列式二进制文件；一个向量占一列，不是把66个元素拆成66列。
action.wbc前29项保留身体下发目标供核对，不再额外拼入66维动作。
完整31维关节顺序见JOINT_ORDER.txt；切片右边界不包含在内。

3. 夹爪坐标和左右顺序
observation.state[29]是左爪实测， [30]是右爪实测。
action.wbc[29]是左爪实际发送的目标， [30]是右爪实际发送的目标。
两者来源不同，不能拿扳机深度或目标位置冒充实际反馈。
我方闭合0deg=0rad，打开300deg=5.235987755982989rad；此范围是我方电机参考。
示例：左目标150deg应保存2.617993878rad；右目标300deg应保存5.235987756rad。
若控制API接收度，反归一化后的rad乘180/pi再传入；这是单位换算，不是改成开合比例。
我方现场左CAN5、右CAN6；训练始终左后右，与到包先后无关。
训练内部按真实电机rad统计归一化，推理后用同一统计反归一化；不得使用旧仿真开度统计。

4. 三路画面与文字
键：observation.images.ego_view、observation.images.left_wrist、observation.images.right_wrist。
训练解码RGB uint8[480,640,3]；MP4/H264/yuv420p、无音频。全视野等比缩放+黑边补齐。
训练视频50fps，一行对应一帧；物理相机约30fps，重复最近有效图像，不复制token补动作缺帧。
原始头部配置1920x1080@30、两腕640x480@30；训练统一640x480。
例如任务文字“把桌上的红色盒子放入右侧托盘”，按真实示范填写；可录后补填。
任务文字在tasks.jsonl中，由每行task_index引用；无需每帧重复存整段文字。

5. 文件布局
DATASET/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.ego_view/episode_000000.mp4
  videos/chunk-000/observation.images.left_wrist/episode_000000.mp4
  videos/chunk-000/observation.images.right_wrist/episode_000000.mp4
  meta/info.json
  meta/modality.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/stats.json
  meta/episodes_stats.jsonl
  meta/source_map.json
  meta/elf3_contract.json
  meta/native_1d_derivation.json
每个episode一份Parquet、三段视频；每1000条换chunk目录。
info给features/fps/总量/路径模板；modality给字段切片；tasks给任务索引和文字；episodes给条编号/length/tasks。
JSONL是一行一个JSON对象。stats包含本数据mean/std/min/max/q01/q99；episodes_stats为逐条统计。
source_map保留原始任务、真实时间、行序号和暂停边界；其余来源JSON不是额外模型输入。
*.example.json/jsonl来自已有合成软件测试，只展示结构，不是真机任务，也不能当归一化统计使用。

6. 暂停和数据质量
A+X前后仍为同一原始任务；暂停期间不写有效训练行。
导出时删除暂停/缺失时间段，将前后有效行及三路图像按原顺序合为同一训练episode。
官方写入器重建连续frame_index和timestamp=frame_index/50；40步窗口允许跨过已删除的暂停接缝。
不足40帧的单个片段也保留，只要求整条清理后至少40帧。不同原始任务不会互相合并。
source_map保留每行原始时间和capture_segments，仅用于追溯，不再作为训练拆分边界。
不复制缺失动作、不把30Hz数值改标签成50Hz。
弃用、失败、缺必要数据的条不进入训练；任务文字在训练导出前必须补为真实指令。

7. 随包配置和验证
elf3_native_1d_abi.json、modality.json、elf3_native_1d_config.py、contract.py与我方现用代码逐字节一致，来源SHA见PROVENANCE.json。
modality.json放在数据集meta中；elf3_native_1d_config.py是训练侧配置，负责state/action顺序和40步取样。
我方训练环境使用自定义ELF3_NATIVE_1D（值elf3_native_1d，projector slot14）。合作方原版GR00T若没有该tag，需在其环境注册；不要把它当官方已有tag，也不要覆盖其现有占用槽。
validate_official.py是我方现用加载核验脚本，需要已有ELF3适配的GR00T N1.7环境、numpy和视频依赖。
在满足上述环境的情况下，运行：python validate_official.py /实际训练数据目录。
它检查三路视频、实际数据统计、state[1,34]、action[40,66]及电机rad归一化往返；不启动机器人，不训练模型。
本次用暂停前60帧、后60帧的合成记录验证：一条120帧episode、三段各120帧视频；官方GR00T加载器可取得跨接缝的40步窗口。当前现场三相机保存验收和训练效果不能由此推定。
此包只交付当前66维合同，不包含模型权重、训练统计或合作方四自由度改造。
