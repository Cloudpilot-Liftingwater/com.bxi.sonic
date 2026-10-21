# test-01 数采现场测试准备

此文只准备操作步骤。机器人尚未安装本版；同事使用期间不要执行上传后的启动、相机占用或服务切换操作。

## 本轮范围

- 继续用机器人现有SONIC、夹爪和example控制流程。
- 新采集代码：`20261020-collection-09-jpeg-cache`，客户包v6。
- 数值按真实50Hz live推理取样，图像30Hz取最近缓存；复用画面只编码一次JPEG，各行仍保存独立图片文件。
- 左扳机/CAN5、右扳机/CAN6；身体29＋双爪2，头部关节不作为训练项。
- A+X暂停写数据、恢复同一条；导出训练时按官方方式合为同一episode，删除暂停空段后允许动作窗口跨接缝。
- 主宿主本地镜像是`bxi_rl_controller_ros2_example-pr`；`example-main`是旧一体化版本。真机现有安装路径仍为`/home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example`。

## 1. 空闲后解压到独立测试目录

将客户v7包及同名`.sha256`文件传到机器人的个人测试目录。主机地址使用操作者当时确认的地址；此前为`192.168.88.149`。不要把整个包覆盖到机器人现有Mod目录。

在存有包的机器人终端：

```bash
sha256sum -c sonic_customer_20261020_official_writer_v7.tar.gz.sha256
mkdir sonic_collection_trial_v7
tar -xzf sonic_customer_20261020_official_writer_v7.tar.gz -C sonic_collection_trial_v7
cd sonic_collection_trial_v7/com.bxi.sonic
python3 -B tools/customer/check_delivery.py
```

目录已存在时先确认它的来源，不反复覆盖。必须看到`PASS`。

## 2. 确认三相机与站点配置

先按[人工相机检查](CAMERA_MANUAL_CHECK.md)分别看图、遮镜头确认左右；关闭自己的预览后再录制。头部需要现有驱动提供1920×1080 RGB，两腕640×480 RGB，均30Hz。只有设备枚举不算画面通过。

```bash
cp tools/customer/site.example.json tools/customer/site.local.json
nano tools/customer/site.local.json
```

在完整配置中修改以下项，保留其他字段：

| 字段 | test-01设置 |
|---|---|
| `controller_setup` | `/home/bxi/bxi_vla_ws/bxi_rl_controller_ros2_example/install/setup.bash`；使用已有构建环境 |
| `capture_profile` | `sonic50_latest_rgb` |
| `training_export` | `false`；保存后在工作站离线导出 |
| `training_task` | 本次实际示范指令，例如“把桌上的红色盒子放到右侧托盘上” |
| `camera_roles_confirmed` | 实际看图确认后才填`true` |
| `camera_role_evidence` | 写本次确认日期、左/右USB口及遮挡结果 |
| `wrist_usb_ports` | 历史左7、右1；以本次实际确认结果为准 |
| `robot_hostname` / `ros_domain_id` | `elf3-test-01` / `31`；若现场控制环境已变，先核对 |

使用运行现有控制程序的同一用户`bxi`：

```bash
export SONIC_SITE_CONFIG="$PWD/tools/customer/site.local.json"
source tools/customer/environment.sh
/usr/bin/python3 -B -c 'import rclpy, cv2, yaml; from communication.msg import ActuatorCmds, ActuatorStates; print("IMPORTS_OK")'
```

这一步只配置当前终端及检查导入，不启动控制器或相机。缺依赖时保留错误，不改动同事的运行环境。

## 3. 仅保留一个任务采集监听器

由操作者在空闲时检查：

```bash
systemctl is-active elf3_pico_collection.service
```

如果旧监听器仍active，先由负责操作者正常结束旧采集并停止旧监听器，再开本次前台测试。不要同时运行两个监听器，不删除互斥锁。本次没有安装或修改systemd单元。

机器人按原流程启动SONIC并进入有效POSE后，在上述新测试包终端执行：

```bash
bash tools/pico_collection/sonic_pico_collection_20261019_r7/listen.sh \
  --left bxi --right bxi --left-id 1 --right-id 1 --left-bus 5 --right-bus 6
```

监听就绪只表示可以接收按键，不表示已保存有效数据。第一条开始时检查`session.log`：

- `COLLECTOR_RELEASE=20261020-collection-09-jpeg-cache`
- `CAPTURE_PROFILE=sonic50_latest_rgb; numerical_fps=50; camera_fps=30`
- `TASK_INSTRUCTION=`是实际任务文字
- `EXPECTED_GRIPPERS`左5/右6

## 4. 先测一条保存，再测一条弃用

| 操作 | 要看到的结果 |
|---|---|
| 松开组合键；左握把+A开始，正常示范15秒 | `RECORDING`，有效行和写盘行持续增长；没有持续队列积压 |
| A+X切出POSE，等5秒 | `PAUSED`；停止新增任务行，仍是原session |
| A+X重进POSE，正常示范15秒 | 恢复增长；仍是同一session/episode |
| 左握把+A保存 | 等`SAVING/AUDITING`结束；拿到数据路径及读回结果 |
| 另起一条录10秒，左握把+B | 保留在弃用目录并排除训练；上一条保存结果不受影响 |

按键均应先释放再按。遇到缺输入、错误或队列持续积压，保存日志与结果，不靠连续重试掩盖问题。任务保存和文件读回尚未结束时不要Ctrl+C。

结束前台测试监听器可在其终端Ctrl+C；它只清理自己启动的采集/腕相机子进程。旧监听器的恢复由操作者按原现场流程处理。

## 5. 回传整条原始episode，工作站读回并导出

原始保存仍是`meta.json`、`frames.jsonl`及三路JPEG。也保留对应session的`trial.yaml`、`session.log`、`launcher_status.json`。不要只传视频或终端截图。

从实际报告取完整episode路径，再执行；下面的占位路径需替换：

```bash
python3 -B /data/ros2_ws/com.bxi.sonic-main/tools/pico_collection/sonic_pico_collection_20261019_r7/verify_saved.py /实际episode目录
/data/ros2_ws/.venvs/sonic-data-export/bin/python /data/ros2_ws/com.bxi.sonic-main/tools/gr00t_dataset/export_dataset.py /实际episode目录 --output /新的训练数据目录
/data/ros2_ws/Isaac-GR00T/.venv/bin/python /data/ros2_ws/com.bxi.sonic-main/tools/gr00t_dataset/validate_official.py /新的训练数据目录
```

最终检查：三路画面角色正确；源时间支持50Hz；暂停两段属于同一原始条；没有假补动作；模型状态34维、动作66维；夹爪是实际电机rad；任务文字正确；GR00T加载通过。

离线测试只证明软件链路与合成负载通过，现场帧率、设备时钟、磁盘和真实图像仍需本轮验收。跟踪评测、SONIC输入帧管理和夹爪控制本轮未改，不要求为这个采集修复重新安装控制器。

本版仅将训练文件生成改为官方SONIC写入器、统计改为官方GR00T函数；原始ROS采集前端及控制链路保持v6内容。客户不使用上面的工作站固定路径，按[导出说明](../tools/gr00t_dataset/README.md)创建自己的独立环境。
