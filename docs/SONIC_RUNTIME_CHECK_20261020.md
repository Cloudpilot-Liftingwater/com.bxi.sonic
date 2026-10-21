# SONIC 启动、按键、关节数与人体数据核对

2026-10-20。核对对象：本地 `bxi_rl_controller_ros2_example-main`（一体化旧版）及 `com.bxi.sonic-main`（独立Mod）。独立Mod的宿主映射另参考本地 `bxi_rl_controller_ros2_example-pr`；不能据此假定同事运行目录或机器人配置相同。

## 按键：先确认是哪一版

| 动作 | 旧example：键盘 / 普通手柄 | 独立Mod配套新版宿主示例：键盘 / 普通手柄 |
|---|---|---|
| Normal | `1` / `RB+X` | `1` / `RB+X` |
| PD制动 | `!` / `RB+B` | `!` / `RB+B` |
| SONIC（从Normal进入） | `6` / `RT+X` → `btn_10=7` | `@` / `LB+RB+X` → `btn_10=9` |
| 后空翻 | `0` / `LT+X` | `6` / `LT+X` |
| 深度行走 | 本次旧配置没有对应项 | `0` / `RT+X` → `btn_10=7` |

新版键盘6不是SONIC，不能沿用旧版快捷键。实际以宿主加载的 `remote_controller/config/xbox_default.yaml` 为准。新版组合还要求释放其他肩键/扳机，避免组合冲突。

独立Mod只声明接收事件值，不拥有宿主键盘/手柄配置；本轮没有把9强改为7，否则会撞上宿主深度行走。键盘启动入口 `remote_controller_keyboard.launch.py` 和手柄入口 `remote_controller.launch.py` 都读同一个配置，前者通过 `--keyboard` 选择输入驱动。键盘空格清除速度输入，不代表启动/停止整套控制器；手柄Start/Stop另走配置里的system命令，和进入SONIC不是一回事。

PICO的ABXY/A+X是第二层控制：ABXY在OFF时请求人体标定及输入启动，再按会关闭输入；A+X切换POSE/PLANNER。它们不是上述普通手柄的进入SONIC快捷键。

## “进入SONIC不会启动遥操服务”具体指什么

- 旧example：`robot_states.py`进入SONIC只重置/预热策略，没有创建PICO进程；硬件launch也没有包含manager/bridge。该版原本要求另行运行仓库根目录的 `script/run_sonic_pico_sources.sh`。这不等于PICO模型出故障。本轮没有给旧状态机新增自动进程管理。
- 独立Mod：`mod.yaml`已声明状态生命周期节点。宿主prepare SONIC → `pico/manager_launcher.py` → manager → `RoboticsServiceProcess`，并启动 `smpl_bridge`。无需手工启动第二个PICO服务。如果SONIC已进入而人体流没有出现，需看manager的配置错误、服务退出或 `xrt.init()` 日志；仅看“没跟随”不能断言服务没启动。
- 本轮没有启动或停止以上任何进程。缺少同事实际运行路径及首条报错时，不把本地复现原因当作现场已确认根因。

## 29/31关节

旧example的回调曾把整个反馈数组直接赋给长度29的缓存。已用实际回调复现：`could not broadcast input array from shape (31,) into shape (29,)`。

已改为按 `ActuatorStates.name` / `JointState.name` 找到模型的29个身体关节。额外具名关节可以存在，反馈顺序也可不同；不直接取前29项，不改变模型维度或下发的具名29关节命令。保留旧的无名称29项反馈兼容；无名称31项无法可靠识别布局，会报明确错误。缺失身体关节、重复名称和非有限身体反馈仍拒绝。

独立Mod已用 `PolicyJointContract` 和宿主 `JointInputBinding` 按名称选取29个模型关节。本轮测试其真实step方法配合本地新版宿主映射，通过29项和31项乱序输入；没有改模型或独立库policy。这个结果不代替检查未知宿主版本的接收适配。

## 人体标定文件

两份本地文件均存在，且55关节静态坐标与父节点逐项相同：

- 旧example：`gear_sonic/data/human/human_joints_info.pkl`。
- 独立Mod：`pico/gear_sonic/data/human/human_joints_info.npz`。

这条控制链使用上述骨架数据，不需要另找完整人体网格模型。旧example曾按终端当前目录查找相对路径；从其他目录启动可复现FileNotFoundError。本轮改为基于 `torch_transform.py` 所在位置定位，并加载到CPU后按原逻辑使用。

独立库本来已使用文件绝对定位，本轮让manager在启动XR服务前加载并校验骨架；缺失、形状错误、非法父节点或NaN会提前报出，不再拖到ABXY操作时才暴露。模型数据未重建、未替换。

如果同事的部署副本确实缺失，只从已核验的同版本库补入对应文件；不要把pkl改名成npz，或反过来。本地文件存在不能证明机器人包内存在。

## 本轮交付边界

本地源码修复，未构建部署包、未安装到机器人、未运行ROS/PICO/电机流程。输入帧管理、A+X语义、夹爪算法和评测工具未改。按键映射只核对和记录，未改宿主全局按键。

新增回归：example 13项；独立库8项。覆盖真实回调31关节映射、乱序、异常名称、跨目录人体FK、独立策略31关节绑定与启动前骨架检查。均为离线测试，不代表真机验证。
