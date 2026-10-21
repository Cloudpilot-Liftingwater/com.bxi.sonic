> 客户可移植入口见 [customer](../customer/README.md)。`recording/record.py`现可用`--mod-root`指定实际装载Mod的指纹路径；记录/评测算法未变。下文SSH一键入口只适用于test-01，不是客户部署前提。

# SONIC 真机跟踪评测工具

这是同一套工具的两部分：`recording/` 记录运行中的数据，`evaluation/` 重建参考动作、计算误差和生成报告。统一入口为 `tracking_eval.sh`。

2026-10-19 收入独立库。记录工具来自 `sonic_tracking_20261019_v4`，参考评估来自 `sonic_eval_20261015_v2`，均已用于 ELF3 test-01 的15秒、120秒真机记录。此次只追加工具目录和根README入口，没有修改库里的SONIC、PICO、夹爪、模型、配置或依赖文件，也没有给本地policy自动加观察点。

## 当前与现场版本关系（2026-10-20同步后）

库根policy/state/plugin、token模型和tracking_tap已同步test-01真机源码。输入帧管理、A+X保持/恢复、motion_token与夹爪遥测接口均在当前根代码中。外部宿主使用同期主example基线。同期r7采集程序见 `../pico_collection/`。

10月19日只添加工具、未改根源码是当次整合的历史说明；当前根源码已包含观察点和真机基线。详见 [现场基线说明](../../docs/FIELD_BASELINE_20261020.md)。

r7有同session暂停/恢复记录；当次因相机/夹爪缺项，有效行数为0，不能据此称完整任务数据验收通过。跟踪评测不依赖这些任务采集输入。

## 记录和计算什么

```text
人体参考 → SONIC encoder → 实际使用的token
                         ├→ 实时控制目标 → 最终下发指令 → 实际关节反馈
                         └→ 工作站离线g1_kin → 重建参考 ───┘比较
```

- 记录实际使用的1770维模型输入、64维token、29关节模型目标/最终指令/反馈，以及原始PICO、SMPL、状态机信息。
- 观察点是SONIC内部的可选数据出口；通过本机非阻塞Unix数据报给记录器，无记录器时不序列化数据。
- 离线重建只在工作站分析时运行，不向机器人发送重建角度。
- 主报告显示参考、控制目标、最终指令和实际角度，分别计算参考→实际与指令→实际的MAE/RMSE/P95、偏差与轨迹滞后。
- 同一token的学习模型参考不是独立动捕真值；没有全局位姿时不计算完整全局指标。缺项不会填成零误差。

## 直接使用（工作站）

```bash
cd /data/ros2_ws/com.bxi.sonic-main
bash tools/tracking_eval/tracking_eval.sh --help
bash tools/tracking_eval/tracking_eval.sh --record 120
```

看到“记录已开始”后操作。时间到只结束记录器，随后下载数据并自动分析。等输出 `HTML_REPORT=.../analysis/tracking_reference.html`、返回提示符再结束该终端。上传、下载或分析时不要为结束机器人操作而中断此脚本。

120可改为1–600秒。首次使用先录15秒确认trace增长。正常跟随时state、command、trace增长；A+X切出POSE时继续记录保持行为，切回POSE继续记录恢复。跟踪记录与任务采集器是独立链路，跟踪记录不会被A+X暂停，也不会替你开始/保存任务数据。

本工具不需要相机，不以双夹爪同时就绪为前提，不自动停服务、杀进程、启动机器人、发CAN或使能。

### 只分析已有记录

把末尾路径换成实际含有 `events.jsonl`、`metadata.json`、`capture_result.json` 的run目录：

```bash
bash tools/tracking_eval/tracking_eval.sh --analyze /实际记录目录/run.XXXXXX
```

纯离线分析不连接机器人。原始记录保留，`analysis/` 报告会重新生成。要比较不同分析版本时，先复制整条记录，再对副本分析。

## 当前机器人适配与观察点

随包启动配置针对已验证的 **ELF3 test-01**：

| 项目 | 当前值 |
|---|---|
| SSH | `bxi@192.168.88.149`；使用已信任的 `elf3-test-01.local` SSH主机指纹 |
| 机器人工作区 | `/home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example` |
| 机器人ROS环境 | `/home/bxi/sonic_gripper_fastfollow_20260917/environment.sh` |
| 观察点基线 | 已归档demo-03 / playout-01两种policy，精确校验源码 |
| 参考解码器 | 同源step28800，匹配模型SHA与关节顺序 |

当前库根已经包含与test-01归档一致的v4观察点；复制目录不会自动安装到另一台机器人。安装器仍只接受已知policy，遇到未知源码会拒绝覆盖。更换策略/模型后需要重新核对观察点和关节契约。

test-01的v4观察点已安装并取得现场trace，平时可直接录制。其他时候先检查：

```bash
bash tools/tracking_eval/tracking_eval.sh --check
```

只有确实需要安装/移除时，正常结束自己的控制、PICO和记录流程，再手动运行以下对应命令：

```bash
bash tools/tracking_eval/tracking_eval.sh --install
# 移除观察点：
bash tools/tracking_eval/tracking_eval.sh --remove
```

上述两个动作会修改机器人已验证基线上的观察点文件；**复制工具到本库不会执行这些动作**。安装成功后手动启动原控制流程，确认trace增长。机器人多人共用时不要干扰其他操作者。脚本不代停进程。

`--record` 会只读检查远端观察点文件；缺失时输出 `TRACE_HOOK_MISSING`，不空录。文件完整仍需运行中的控制器实际加载观察点。

## 依赖与结果位置

工作站默认使用 `/usr/bin/python3`，依赖NumPy、ONNX Runtime、Matplotlib；已验证版本列在 [requirements.txt](requirements.txt)。按需在自己的环境安装，不会自动改根项目依赖：

```bash
/path/to/python -m pip install -r tools/tracking_eval/requirements.txt
SONIC_EVAL_PYTHON=/path/to/python bash tools/tracking_eval/tracking_eval.sh --record 120
```

远端仍使用机器人现有ROS2环境、communication消息和pyzmq，不启动额外控制服务。本目录已包含离线解码模型、URDF、测试样本及上传包；日常运行不依赖原Codex outputs路径。

工作站结果：`/data/tmp/sonic_tracking_review.XXXXXX/run.XXXXXX/`。
机器人原始副本：`/home/bxi/sonic_tracking_runs/run.XXXXXX/`。
请按终端打印的实际路径打开报告；长期保留应归档完整run目录，不把临时目录当唯一备份。数据未收入本代码库。

| 文件 | 用途 |
|---|---|
| `analysis/tracking_reference.html` | 四条曲线与跟踪误差主报告 |
| `analysis/report.json` | 捕获质量、配对情况、执行误差 |
| `analysis/eval_report.json` | 参考重建、参考误差、延迟与指标边界 |
| `events.jsonl` | 原始记录，后续重算需要 |

`PARTIAL_OR_NO_VALID_SAMPLES` 是较宽泛的质量标记，不等于零有效数据：需同时看scored数量和quality缺项。已验证120秒记录中，长暂停使用旧参考超出人体旋转回查时间范围，造成部分旋转未匹配；trace完整且仍有有效参考/指令跟踪成绩。A+X主动暂停产生stale_hold是预期行为。

## 软件组织与检查

```text
tracking_eval.sh       一次完成记录、下载和分析
recording/             记录器、观察点、安装器、基础分析、已知policy基线（仅安装校验用）
evaluation/            g1_kin模型、参考解码、误差计算、HTML报告
packages/              原版v4远端上传包与SHA256校验
PROVENANCE.json        来源、整合改动及验证范围
SHA256SUMS             本目录文件校验
```

源码沿用已验证算法；整合只调整工作站入口和离线模块的相对路径。没有复制旧v1分析库，因为所需基础模块与v4字节相同。

```bash
cd tools/tracking_eval
sha256sum --quiet -c SHA256SUMS
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m pytest -q recording/tests evaluation/tests -p no:cacheprovider
```

pytest为开发测试依赖。测试使用合成数据和本机Unix套接字，不连接机器人。一个旧版本逐字节对照测试依赖原开发目录，离开该目录时会跳过。
`evaluation/export_kinematic.py` 保留模型导出源代码，仅供维护者使用：它另需历史训练归档、原模型和PyTorch/ONNX/OmegaConf等依赖，并带有当时归档路径。日常记录、解码、评估不需要执行导出。

软件检查不等于独立外部真值校准；没有指定精度阈值时，报告实测数值，不自动宣称达标。

## 2026-10-20 连接地址及验证记录复核

两部分评测软件均已收入本目录，根README也有使用入口。已有15秒真机记录取得724条trace，120秒记录取得6000条trace、99499条可评分actuator反馈；完成记录、离线参考重建、指标和HTML报告全链路。整合时84项离线测试通过、1项历史开发基线检查跳过，并复算15秒实测得到一致报告。此为软件链路验证，不代表已用外部真值完成计量校准或达到预定精度指标。

本次复核发现旧启动入口仍写192.168.88.45，现已改为192.168.88.149，并保留StrictHostKeyChecking：连接固定IP，但按已信任的elf3-test-01.local核验同一台机器的主机密钥，避免依赖mDNS解析。记录器、观察点、模型和精度算法未修改。地址变化时可以设置`SONIC_EVAL_HOST=bxi@新IP`；只有确认是同一台机器人时才沿用该主机密钥别名。

此更新只发生在工作站工具，不会安装观察点或启动机器人。本次完整归档已取回，库根运行代码已同步；见现场基线说明。
