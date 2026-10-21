# PICO采集工具

当前本地源码 `20261021-collection-12-latest-received`；本版尚待人工部署和真机验证。现场此前确认的是v11。

详见[当前采集说明](sonic_pico_collection_20261019_r7/README.md)和[客户部署入口](../customer/README.md)。目录名保留兼容已有服务。

本版按官方最近值语义保存状态、IMU和图像，取消30ms配对丢行；ROS输入接收与组装分开。机器人保存后只收尾，完整JPEG读回检查在工作站进行。按键仍为左握持键+A开始/保存、左握持键+B弃用，A+X暂停后续采同一条。

[训练导出](../gr00t_dataset/README.md)继续复用官方SONIC写入器；暂停前后合成同一训练episode。客户依赖、modality、66维动作合同均随独立库提供。
