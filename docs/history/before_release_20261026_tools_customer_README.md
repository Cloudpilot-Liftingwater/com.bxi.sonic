> 2026-10-22当前版本：BXI demo04 / 数采v16，夹爪开口0～70mm；旧角度参数和v15验证记录仅供历史追溯。使用[毫米协议与合同说明](../../docs/GRIPPER_MM_PROTOCOL_20261022.md)。新版本尚未真机验证。

# 客户交付入口（2026-10-21，数采v14）

本目录解决“只有开发机/旧机器人目录存在才能运行”的问题。当前支持的预编译组合为Ubuntu22.04、x86_64、Python3.10、ROS2 Humble。虽然主库还有其他平台资源，当前BXI和采集相机运行库不能据此宣称已验证ARM。

## 包含及外部依赖

库内包含：SONIC模型、参考动作、人体骨架数据、PICO处理程序、XRoboToolkit服务/绑定、夹爪demo03原生库及构建源码、DM代码、跟踪记录/参考重建/评测模型及URDF、r7采集源码/相机配置、现场修正时间戳的usb_cam运行库。运行资源都通过当前库位置定位。

外部需客户提供：匹配的example Mod API 4宿主、basic_actions、communication ROS消息、ROS2 Humble及系统共享库、已构建的宿主install/setup.bash、Python依赖和真实设备权限。头部相机驱动属于example的bxi_depth_camera组件，启动/序列号由客户现有相机配置提供；本工具不偷偷启动该驱动。

库内`requirements-pico.txt`、`tools/tracking_eval/requirements.txt`、r7的`collector/requirements.txt`分别列出Python依赖；本次另附`requirements-field.txt`列出真机已安装的相关新版本（包括SciPy1.15.3），避免只装系统旧SciPy而缺少scalar_first接口。此文件不是整套ROS依赖锁定。不要把源码复制当作宿主已构建安装，也不要将`build/install`中的旧路径直接复制给客户。宿主消息可包含身体29＋头部2；本次训练只取身体29，并另取双爪2，共31自由度。头部关节不作为训练或采集准入条件。训练工具使用已有ELF3适配的Isaac-GR00T环境。

## 设置站点

在独立库根目录执行（只复制配置，不启动设备）：

```bash
cp tools/customer/site.example.json tools/customer/site.local.json
```

编辑`site.local.json`：

- `controller_setup`必须填客户实际构建的example `install/setup.bash`绝对路径。这是**显式外部框架依赖**，不是遗漏的项目文件。
- `robot_hostname`和`ros_domain_id`填实际机器人及其控制域；默认test-01/31是现场资料，不是所有客户通用值。
- DDS默认沿用现场CycloneDDS回环网络配置；若客户主程序使用其他DDS配置，应一起匹配，不能只改记录器。
- `wrist_usb_ports`默认左7/右1，PCI设备拓扑沿用当前主板。更换主板PCI位置需相应修改configure.py的设备路径，不能宣称任意硬件自动适配。
- 遮镜头确认左右后再将`camera_roles_confirmed`置true并填写`camera_role_evidence`。`runs_root/data_root`留空采用当前用户家目录；也可显式指定数据盘路径。
- `capture_profile`填`sonic50_latest_rgb`，相机仍30Hz，数值按真实live推理记录；未填此项的历史站点保持head30。夹爪直接保存电机rad。`training_export`保持false，保存后执行离线导出。
- `training_task`可填写实际任务，如“把桌上的盒子放到右侧托盘上”，也可先留空、录制后在导出时标注；未标注的数据不能作为训练就绪数据。详见[任务标注](../../docs/TASK_LABELING.md)。
- 站点配置含具体设备身份，不建议当作通用模板提交。禁止直接执行未填写的示例配置来启动采集。

```bash
export SONIC_SITE_CONFIG="$PWD/tools/customer/site.local.json"
source tools/customer/environment.sh
```

此步仅设置环境，并选择库内XR服务和相机运行库；不会启动机器人、ROS节点、相机或监听服务。SONIC本体仍按客户example的Mod安装/启动流程装载根目录`mod.yaml`；启动宿主的同一终端也应使用这份环境以选中库内XR服务。

## 相机确认后再开始采集

先看[人工三相机检查](../../docs/CAMERA_MANUAL_CHECK.md)，并安装隔离写入依赖（只下载和检查，不启动设备）：

```bash
bash tools/gr00t_dataset/prepare_online_runtime.sh "$PWD/.runtime/collection-writer-v13"
```

环境目录名v13保留兼容，与当前采集发行号v14无关。下载代理处理及版本见[在线采集说明](../gr00t_dataset/ONLINE_COLLECTION.md)。通过后才由操作员在机器人执行前台监听器：

```bash
bash tools/pico_collection/sonic_pico_collection_20261019_r7/listen.sh \
  --left bxi --right bxi --left-id 1 --right-id 1 --left-bus 5 --right-bus 6
```

同一机器人已有监听器时不要重复开；互斥检查仍保留。这个入口本身不安装/启动systemd服务。左握持键+A开始/保存，左握持键+B弃用；A+X暂停/恢复同一条数据。v14启动常驻观察进程和两路腕部相机，沿用已经运行的头部图像；按开始后才写入episode，每条不再重复启动进程。它不发电机命令。

格式和每个字段见[数据说明](../../docs/COLLECTION_DATA_FORMAT.md)。当前保存原始记录。本地已实现50Hz推理取样与30Hz图像缓存，尚未真机验收；保存原始记录后离线导出为训练包。详见[训练工具](../gr00t_dataset/README.md)及[训练字段字典](../../docs/GR00T_N17_DATA_FORMAT.md)。

## 跟踪记录及离线评估

在机器人、已有SONIC控制运行的情况下，由操作员执行：

```bash
bash tools/customer/record_tracking.sh --output "$HOME/sonic_tracking_runs/customer_trial_01" --seconds 120 --mod-root "$PWD"
```

`--mod-root`必须指向**正在运行的**SONIC根目录；它只用于记录源码指纹，不安装观察点。若宿主加载的是另一份复制到install的Mod，应填写那份路径。当前根policy已含tracking_tap；实际trace增长才算运行链路完整。记录器不要求相机，不发控制命令。

把整条记录目录传到工作站后，在库根执行：

```bash
python3 tools/tracking_eval/recording/analyze.py /实际记录目录
python3 tools/tracking_eval/evaluation/evaluate.py /实际记录目录
```

不需要重新导出模型；重建模型和URDF已在库内。`tools/tracking_eval/tracking_eval.sh`是test-01专用远程便捷入口，客户应使用上述本地记录/分析命令。

## 完整性与打包

```bash
python3 tools/customer/check_delivery.py
python3 tools/customer/build_delivery.py /保存客户包的目录
```

检查文件指纹、资源存在、内部符号链接及关键运行路径；不启动设备。打包排除历史现场安装器、冻结旧包、历史模型导出辅助脚本及历史文档；不会将旧IP/旧安装路径当客户入口。完整仓库中这些历史资料仍保留。交付tar含SHA256清单，解压后可再运行check_delivery.py。

通过文件与离线测试不等于客户机器驱动、权限和真实采集已验收。三路完整输入和两段有效数据同条保存仍需本轮现场测试完成。真机现有程序未因本次本地整理而升级。

## 本次上机准备

当前采集版本为`20261021-collection-14-official-session`；代码目录名r7、依赖环境目录名v13为兼容路径。复用官方录制状态及MP4/Parquet写入，详情见[当前说明](../gr00t_dataset/ONLINE_COLLECTION.md)。`FIELD_TEST_20261020.md`等带日期的旧文档仅供历史追溯。身体和夹爪基线已有真机验证，v14数采仍需现场验收；不要在其他操作者使用机器人时覆盖其运行代码。


本版训练文件保存和统计改为[官方SONIC/GR00T实现](../../docs/OFFICIAL_COLLECTION_REUSE_20261020.md)。原始任务回传工作站后，用[独立导出环境](../gr00t_dataset/README.md)生成LeRobot训练包；不在机器人控制环境混装LeRobot/Torch。官方核心及其来源/补丁已随库提供。
