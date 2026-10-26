> 当前综合发布：20261026；夹爪毫米合同，现场保存/弃用与官方加载已有记录。批量下载、弃用检查导出和有限窗口验证见[collection_review](../collection_review/README.md)。当前已知故障见[发布说明](../../docs/RELEASE_20261026.md)。

# ELF3采集与GR00T训练数据工具

当前为v16（毫米协议），直接复用NVIDIA SONIC录制状态、写入器和清理/合并程序，GR00T官方统计和加载器。[在线使用说明](ONLINE_COLLECTION.md)。来源固定到 SONIC `45c03facbf848e0c743091b7d9c17952096ff908`、GR00T `dac54b313160c201c450f97df0fd74ebd379c892`；来源指纹和本地补丁见 `vendor/`。

当前真机合同：身体29＋左右各1个夹爪开口（mm）；不含头部。状态34维＝身体29＋实测开口2＋重力3；动作66维＝motion token64＋最终下发开口2。`elf3_native_1d_abi.json`、`modality.json`、`elf3_native_1d_config.py` 都在库内。训练库同名ABI已同步。0mm闭合，70mm打开，合同ID `elf3_native66_bxi_opening_mm_v2`。详见[新协议说明](../../docs/GRIPPER_MM_PROTOCOL_20261022.md)。

## 原始数据与训练数据

在线原始数值为官方LeRobot v2.1 Parquet，三路原生分辨率RGB保存为MP4；相机30Hz，实际推理行目标50Hz。复用最近画面，不补造motion token。时间偏差与真实接收时间保存在 `source.jsonl`；50Hz是项目时基，不代表所有传感器硬件都是50Hz。

官方清理删除全零SMPL及相邻冻结段，删除弃用episode；暂停前后仍在同一episode，不设置训练窗口接缝隔离。没有额外20ms时间格删帧或不足100行拒绝保存。录制中的诊断列 `teleop.smpl_pose` 用于官方清理，导出时移除，不增加VLA模态。

导出调用官方 `process_single_dataset`、`write_output_dataset`；无需删帧的视频逐字节复制，不再全部解码重编码。保留每路原分辨率，由训练/推理共同使用的官方Processor做图像变换。旧模型使用不同预处理时不要直接混用。

## 工作站导出

使用隔离环境，避免把旧版LeRobot依赖装进ROS/GR00T训练环境：

```bash
python3 -m venv "$HOME/.venvs/sonic-data-export"
source "$HOME/.venvs/sonic-data-export/bin/activate"
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0' 'torchvision==0.21.0'
python -m pip install -r tools/gr00t_dataset/requirements.txt
python tools/gr00t_dataset/export_dataset.py --check-dependencies
python tools/gr00t_dataset/export_dataset.py /原始/session_x/episode_0000 \
  --output /新的训练数据目录 --task '把红色方块放进蓝色盒子'
```

多个原始目录可连续传入；任务文字必须对应实际操作。未标注时加 `--inspection-only` 仅供查看，不标记训练就绪。原始数据和已有输出都不会覆盖。本版只导出毫米合同。旧rad记录使用冻结v15工具，不能改元数据伪装为mm。

导出包含官方 `data/`、`videos/`、`meta/info.json`、`episodes.jsonl`、`tasks.jsonl`、`modality.json`，另由官方GR00T `generate_stats`生成带指纹缓存的 `stats.json`。`elf3_contract.json` 标明单位；`source_map.json` 与 `provenance/` 保留来源。每个保存任务对应一个训练episode；短任务仍保留，是否有足够40步训练样本由读取器/采样器检查。

## 检查与训练

```bash
# 原始保存/弃用检查，工作站执行
python tools/gr00t_dataset/verify_online.py /原始/session_x/episode_0000 --session /运行记录/session_x
# 切回已有ELF3适配GR00T环境
python /实际SONIC库/tools/gr00t_dataset/validate_official.py /训练数据目录
```

沿用 Isaac-GR00T `examples/ELF3/native_1d` 的训练入口和ELF3_NATIVE_1D/slot14。默认训练、推理、恢复训练都会检查开口mm合同身份；旧仿真开度数据、统计和checkpoint不能只改标签来使用。

离线测试证明官方文件写入/读回、暂停清理、保存弃用、40步采样和毫米归一化往返成立；不等于真机采集或模型后训练已完成。最新结果见 `VALIDATION.json`。
