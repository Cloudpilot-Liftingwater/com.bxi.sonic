# SONIC 真机跟踪记录与分析 v4

记录真实运行中的参考输入、模型输出、最终发布角度与驱动反馈，生成逐关节曲线、误差与滞后报告。不需要任何相机，也不依赖三摄采集服务。

2026-10-15兼容更新：支持原demo-03及新的SONIC输入帧管理版本。
观察数据格式和误差算法不变；安装/移除观察点保留对应控制版本，不会降级policy。
已安装观察点会由输入链路升级包自动保留。统一入口名称不变。

安装器v4（2026-10-19）：修复 `pgrep -af` 把自身远端 bash 脚本中的未执行 `record.py` 分支误认为活动记录器的问题。现场 `sonic_tracking_review.7qxCfL` 和 `fMMnM7` 均只有安装器包装进程，却因此被拦截；观察点没有安装，继续录制不会得到 trace。

现在按真实父进程PID链排除本安装器自身，其他控制器、输入和记录进程仍阻止写入。同时避免仅因相机二进制位于 bxi_example_py_elf3 目录而误判为控制器；PICO manager_launcher 仍受检查。
`--record` 在开始前只读检查观察点文件完整性；缺失或部分安装直接报 `TRACE_HOOK_MISSING`，不启动空录。文件完整不代表当前进程已加载，以实际 trace 为准。
观察点、policy补丁、记录程序、分析算法与v3逐字节相同。v3旧包保留，统一入口改用v4。

## 直接使用

以下在工作站 `huangchenwei-System-Product-Name` 运行。SSH / 安装 sudo 密码在终端输入。

```bash
TRACKER=/home/huangchenwei/Documents/Codex/2026-10-13/elf3-sonic-home-huangchenwei-documents-codex/outputs/sonic_tracking_20261014.sh
bash "$TRACKER" --check
```

第一次需安装一个只记录内部张量的观察点。机器人空闲、正常结束你自己的控制和采集流程后执行；脚本发现活动控制进程会拒绝安装，不会替你停止它们：

```bash
bash "$TRACKER" --install
```

看到 `TRACKING_HOOK_INSTALLED_NOT_HARDWARE_VALIDATED` 后，手动启动原有硬件/键盘/PICO流程、进入 SONIC。模型、运动计算、身体增益和夹爪参数不变，原稳定存档不变。需要这一次手动重启，是为了让运行进程加载新增观察点。

开始一次 60 秒记录：

```bash
bash /home/huangchenwei/Documents/Codex/2026-10-13/elf3-sonic-home-huangchenwei-documents-codex/outputs/sonic_tracking_20261014.sh --record 60
```

在已有 SONIC POSE 遥操中按原方式运动即可，不需要按采集快捷键。终端会显示 `state`、`command`、`trace`、`pose` 数量。`trace=0` 表示内部观察点未加载或当前没有 SONIC 推理，不能声称已记录完整链路。

时间到后只结束记录器，机器人继续原流程；随后自动拉回数据并分析。打开最后显示的 `HTML_REPORT=.../analysis/tracking_reference.html`。若提前 Ctrl+C 或 SSH 断开，原始文件保留；可能需要手动取回打印的 `REMOTE_RUN_DIR`。不会自动重试运动、重新进入 SONIC 或重启服务。

可以把 `60` 改成 1–600 秒。建议首条 60 秒内包含短暂静止及左右肘、膝、腕的往返运动，以便区分偏差与动态滞后；具体运动范围由现场操作员控制。

## 实际链路：哪些值能比较

| 层 | 记录内容 | 单位/含义 |
|---|---|---|
| PICO → SMPL | `smpl_pose` 的 21×3 旋转向量、24×3 关节点位置、root、6个原生腕角、帧编号 | 旋转向量以 rad 保存，CSV 转成度；其 x/y/z 分量不是三个独立铰链关节角 |
| encoder 实际输入 | 每次实际推理使用的完整 1770 维模型输入，其中前840维为参考窗口 | 10帧×72维关节点位置 + 10帧×6维相对身体朝向 + 10帧×6维腕角；位置是位置，不能当角度 |
| encoder 结果 | 同一次模型执行返回的64维 post-FSQ token | 隐变量，不是关节角 |
| decoder 原始输出 | 29维 `raw_action` | 无量纲 action，不直接当角度 |
| decoder 转换后目标 | `default_q + clip(raw_action,−20,20) × action_scale` | 机器人29关节角，rad |
| 过渡处理后目标 | SONIC 自身源切换混合后的 `policy_target` | 机器人关节角，rad |
| 最终发布指令 | `/hardware/actuators_cmds` 的按名称对齐的 `pos/kp/kd/vel/torque` | 经过状态机和 override 后的 ROS 指令；不等于电机端接收 ACK |
| 实际执行反馈 | `/hardware/actuator_states` 的 `position/velocity/effort` | 驱动发布的关节坐标；不是额外外部测量的物理真值 |

**此 SONIC 路径没有“decoder 输出后再跑一次完整人体→机器人 IK 重定向”的独立29维阶段。** 人体姿态转换和六腕角处理发生在前端；模型产生机器人动作，后面是缩放/偏置、过渡混合、关节名称编排和最终指令处理。报告按真实层次命名，不虚构一个“重定向后”数据源。

本工具按已归档 demo-03 和 playout-01 两种 policy 精确校验安装；机器人 policy 若有其他修改，会拒绝覆盖并保留现场文件。

## 怎样看跟踪精度

1. 先看数据质量：是否确实有 live 样本，名称/时钟/输入是否有效，是否丢包，观察点开销多少。缺数据不填零，不能把“0有效样本”说成“0误差”。
2. 逐关节主指标：实际角度−同源时间最近已发布的目标角度；给出 MAE、RMSE、偏差、绝对误差 P95、最大绝对误差。正误差表示实际角度更大。
3. decoder→过渡后目标→最终指令曲线：用于看输出裁剪、混合、其他指令处理对最终目标的影响；最终指令与实际反馈偏离则定位到执行链。
4. 估计滞后：扫描0–250ms，步长5ms；正值表示实际运动滞后于指令。保留原始误差，同时给出补偿后的 RMSE。静止、幅度不足5°或连续数据不足不报滞后。到达搜索边界不算已确定延迟。
5. 人体动作层：六个腕角是输入中已有的机器人原生腕角，可直接画参考—输出；肘/膝以两段骨骼夹角定义“弯曲角”，比较人体参考、decoder、指令和反馈。全身 SMPL 旋转向量与29个电机关节不能逐项直接相减。

弯曲角的机器人一侧使用随包 URDF 做正运动学；该 URDF 来源和 SHA 在 `contract.json`，尚未重新核对机器人当前模型文件。因此这些几何指标是形状跟踪诊断，不是经过独立动捕验收的全身重定向准确率。

人体旋转角与 encoder 输入采用位置+root+wrist 的逐帧内容精确匹配。相同位置可能对应不同扭转姿态，遇到歧义或未收到对应源帧会明确留空。原始人体角度流始终单独保留，不由关节点位置硬反推人体扭转角。

## 时间和统计边界

- 同机记录 ROS 源时间戳、接收 ROS 时刻、接收 monotonic 时刻；转换到同一 monotonic 时间轴。模型内部观察点直接带 monotonic 时刻。
- 执行主误差按反馈的源时间匹配已经发布的指令，使用零阶保持；只接受年龄≤50ms的指令，不用未来指令。trace→指令也只做≤50ms的因果时间匹配，尚无贯穿硬件的统一序号，报告保留匹配间隔，不能视为物理电机执行的逐包ACK。
- ROS 时间跳变>20ms会分段；不跨时钟段计算。原始错误包、缺失、序号间隙与排队丢失有计数。
- 主统计只含 SONIC live、未混合、状态机信息新鲜、KP>0且有限的具名关节数据；过渡/空闲数据保留在曲线和原始包中。报告分母是本记录器收到的反馈帧，不声称测到了所有硬件帧。
- 反馈最大时间戳年龄1秒，未来时间容差10ms；设备时钟与ROS时钟若不兼容会拒绝评分，而不是用接收时间冒充准确源时刻。
- 保留模型SHA、源码SHA、会话、推理序号。不同模型混在一段记录中时不合并给出精度结论。
- 当前没有用户定义的精度合格阈值，所以软件报告实测数值，不自动判“达标”。

## 输出文件

| 文件 | 用途 |
|---|---|
| `events.jsonl` | 全部原始 ROS 观察值和带原始字节的 PICO/SMPL/模型观察包 |
| `metadata.json` / `capture_result.json` | 环境、源码指纹、计数、丢失与退出结果 |
| `analysis/report.html` | 离线交互报告；选择29关节或肘膝角、查看各层曲线、执行误差及时间区间 |
| `analysis/report.json` | 完整误差、匹配质量、时间/观察开销与限制 |
| `analysis/joint_metrics.csv` | 各关节精度表 |
| `analysis/aligned_joint_angles.csv` | 对齐后的各层机器人角度、误差、有效标记、匹配年龄 |
| `analysis/body_shape_angles.csv` | 人体与机器人肘膝弯曲角 |
| `analysis/human_smpl_angles.csv` | 原始人体21关节旋转向量的度数分量，带源帧编号；不是机器人角度表 |
| `analysis/inference_arrays.npz` | 完整1770维实际输入、64维token、原始/裁剪action、目标、实际被推理使用的反馈，以及10帧对应SMPL旋转向量；歧义留NaN |

`npz` 以 `allow_pickle=False` 读取；原始角度数组为 rad，图表/CSV 为度。图像最多显示5000个反馈点，指标及CSV使用全量有效数据。

已有记录重新分析：

```bash
bash /home/huangchenwei/Documents/Codex/2026-10-13/elf3-sonic-home-huangchenwei-documents-codex/outputs/sonic_tracking_20261014.sh --analyze /data/tmp/sonic_tracking_review.XXXXXX/run.XXXXXX
```

移除观察点（先正常结束自己的控制流程，原始记录保留）：

```bash
bash /home/huangchenwei/Documents/Codex/2026-10-13/elf3-sonic-home-huangchenwei-documents-codex/outputs/sonic_tracking_20261014.sh --remove
```

## 实现与验证范围

仅在 `policy.py` 同一次推理结束处增加容错观察调用，并新增 `tracking_tap.py`。控制算法、模型调用次数、目标数组与夹爪实现均保持原样。观察采用本机非阻塞 Unix 数据报；无记录器时不序列化，有背压时丢观察记录，不阻塞等写盘。只记录器进行磁盘写入。

56 项离线检查通过（含安装器误判回归），覆盖真实 policy 方法的数值等价、套接字丢失、时序/缺失处理、已知滞后合成数据、安装/回退计划和报告读回。离线性能数据不替代机器人现场实时性验证；首条记录需同时查看已有控制周期告警与报告中的观察开销。

尚未部署或取得真机跟踪成绩；示例报告一律标注“合成测试数据”。

工作站预热1,000次后，5,000 次本机 Unix 数据报实测：每帧约10KB，观察调用中位数19.85µs、P95 21.12µs、最大41.30µs；没有记录器且处于连接重试冷却期时中位数0.37µs。此数据仅代表当前工作站，不能替代机器人实时性测试。

报告数据与 JavaScript 语法已离线检查。内置浏览器策略阻止本地文件预览，因此没有完成浏览器视觉/交互验收；HTML可在工作站普通浏览器离线打开。
