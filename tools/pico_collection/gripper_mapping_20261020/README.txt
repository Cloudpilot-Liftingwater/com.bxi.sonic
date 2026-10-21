2026-10-20 test-01 夹爪左右映射更正

用户已确认：更换零件造成的右爪零位错误已重新校准解决。
现在采用左扳机→CAN5→左侧采集数据；右扳机→CAN6→右侧采集数据。
根据真机EIEV91取回的demo03及r7配置制作；只改映射，不换控制程序。

修改范围：
1. 机器人bxi_vla_ws内SONIC src/install的mod.yaml各两个bus值。
2. /etc/systemd/system/elf3_pico_collection.service的两个bus启动参数。
   r7包目录、其源码和SHA256SUMS不变；现有历史数据不改。
3. systemctl daemon-reload使新配置可在下次启动时读取；不启停任何服务。

KP25/KD0.25、速度1400度/秒、加速度28000度/秒²、闭合0/打开300度、
SONIC输入帧管理、A+X同条采集暂停续录、跟踪观察点全部保持原样。
脚本校验已知demo03及r7服务配置，保留文件所有者/权限/符号链接。
写入前要求控制/输入/记录/测试进程和采集监听器已经正常结束。
修改前生成BACKUP；更新失败回滚。不启动ROS、不发CAN、不改零位。

操作（工作站）：
先正常保存/结束采集，等保存结束；正常结束自己的硬件和遥操终端。
机器人多人共用时，等当前操作员结束；不要停止别人的流程。

ssh -t bxi@elf3-test-01.local 'sudo systemctl stop elf3_pico_collection.service'
bash /home/huangchenwei/Documents/Codex/2026-10-13/elf3-sonic-home-huangchenwei-documents-codex/outputs/install_gripper_mapping_20261020.sh --apply

仅看到 MAPPING_INSTALLED: left=CAN5 right=CAN6 后再手动执行：
ssh -t bxi@elf3-test-01.local 'sudo systemctl start elf3_pico_collection.service'

再按原流程启动控制、进入SONIC。左右扳机分别测试一次：
- 左扳机控制物理左爪/CAN5；右扳机控制物理右爪/CAN6。
- SONIC夹爪DEMO entry应显示 left_bus=5、right_bus=6。
- 采集监听器EXPECTED_GRIPPERS应为left bus5、right bus6。
  查看：ssh bxi@elf3-test-01.local 'journalctl -u elf3_pico_collection.service --since "5 minutes ago" --no-pager'
软件版本名仍是demo03，映射改变不会改控制算法版本。

默认使用已信任的elf3-test-01.local，不关闭SSH主机指纹检查。
仅检查可执行同一脚本 --check；无需停当前运行进程。
如需指定地址，使用ROBOT_MAPPING_HOST=bxi@192.168.88.149；须先完成该地址的SSH主机身份确认。

旧demo03/r7发布包保留历史左6右5，不再代表当前要求的接线。
不要为本次修正重装旧整包；未来重装旧包须重新检查并应用本映射。
旧r7安装器会拒绝被修改后的服务配置；不要绕过保护或用旧模板覆盖。
目前只完成本地准备和离线验证，需操作员安装及验证。
