# Python环境与依赖来源

本库的PICO启动器、数采和评测工具采用系统Python或由其创建的venv；也可用uv管理非Conda来源的Python/venv。旧PICO启动器中的Anaconda/Miniconda路径搜索及`conda env list`已移除。显式指定Conda Python或从它创建的venv会报错，提示改用以下安装入口。

## PICO依赖

在库根执行：

```bash
bash tools/customer/prepare_pico_runtime.sh
```

安装器使用 `/usr/bin/python3`，默认创建 `.runtime/<平台>/pico`；从随库wheel启动pip，从PyPI安装 `requirements-pico.txt`，并使用 `tools/customer/requirements-field.txt` 约束现场版本。现有目录若不是venv则拒绝覆盖。只导入依赖检查接口，不调用XR初始化，不启动服务、ROS或硬件。

当前随包 `xrobotoolkit_sdk` 是CPython3.10扩展，推荐Ubuntu22.04系统Python3.10。其他ABI需要重建绑定，不能仅换解释器路径。

默认解释器候选包括库内专用venv、库根`.venv`、当前非Conda解释器、已激活的非Conda venv和系统Python。自定义环境通过 `SONIC_PICO_PYTHON=/绝对路径/bin/python` 指定；保持venv启动路径，不解析掉使venv失效的末端符号链接。子进程清除继承的Conda环境变量及其PATH/动态库路径，保留ROS与XR SDK路径。

## 数采、训练和评测

- 数采继续使用 `tools/gr00t_dataset/prepare_online_runtime.sh` 创建的系统Python venv；LeRobot从随包固定源码wheel安装，其余依赖从PyPI及官方PyTorch wheel源安装。
- ROS节点使用与ROS2安装匹配的系统Python；不把Conda的ROS包注入节点。
- GR00T训练和数据验证使用对应GR00T库自己的uv/venv。训练配置、权重、统计和毫米合同未因本次环境迁移而改变。
- 不复制旧Conda的site-packages到新环境。环境目录不提交Git，也不进入客户交付包；在目标设备重新安装并验证。

## 授权说明与检查范围

Conda包管理器本身与Anaconda商业仓库服务不是同一个概念。[Anaconda官方说明](https://www.anaconda.com/legal)区分了Conda、Miniconda和仓库使用；是否需要商业订阅取决于来源、组织规模及适用例外。切换uv/venv并不免除各第三方包自身的许可证要求，也不能改变历史下载行为。

本库不分发Anaconda/Miniconda安装器、Python环境或其软件仓库镜像。2026-10-26核对的55个随库ELF文件，其NEEDED/RPATH/RUNPATH未出现Anaconda/Conda路径。该检查不等于对所有供应商二进制和系统环境的完整法律审计；原有第三方LICENSE/NOTICE保留。

`docs/history/`中保留旧版Conda使用记录，仅用于追溯，交付清单排除该目录。机器上其他项目的共享Conda环境不由本库安装器卸载；它们应先单独迁移用途、依赖和测试，再移除。本次不修改机器人服务、训练进程或用户全局shell配置。
