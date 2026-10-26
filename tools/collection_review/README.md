# 数据下载、导出与人工检查

复用现场的plan/review/export_all/validate_exports工具，路径改为相对本库。只读机器人现有数据，不启动ROS/相机/录制，不改服务或电机。全部下载包含保存、弃用和中断记录。

## 1. 下载到工作站

要求工作站和机器人有python3、SSH及rsync。先按正常SSH方式核实并信任机器人主机密钥；工具始终使用StrictHostKeyChecking=yes，不自动接受未知密钥。

在库根目录，替换主机和输出盘：

```bash
python3 tools/collection_review/download.py --host bxi@elf3-test-01.local --all --output /你的数据盘/collection_review
```

IP连接但已信任的是主机名时，使用 --host bxi@当前IP --host-key-alias elf3-test-01.local。省略--all只下载最近四次。断线后：

```bash
python3 tools/collection_review/download.py --host bxi@elf3-test-01.local --resume /你的数据盘/collection_review
```

--remote-base默认为远端用户家目录，应包含pico_collection_runs和pico_collection_data。非默认存储请显式指定共同父目录；本入口不扫描全磁盘、不跟随目录符号链接。正在录制/保存时拒绝下载所选会话。断点续传依照原manifest，并逐文件核对SHA256。

## 2. 检查原始数据、清理导出

使用已装好库内导出依赖的Python。示例假定已按客户入口创建.runtime环境；也可用自己的完整环境：

```bash
.runtime/collection-writer-v13/bin/python tools/collection_review/review.py /你的数据盘/collection_review
.runtime/collection-writer-v13/bin/python tools/collection_review/export_all.py /你的数据盘/collection_review
```

review逐条读回数值、图像/视频，统计时间差和操作结果。export_all调用库内官方清理/导出器，生成exports/saved、exports/discarded等目录及index.html。浏览器打开index.html即可看三路视频、任务文字和数值预览。

默认不修改原件、不覆盖已有不同来源导出、不伪造任务文字。旧rad/JPEG数据保留原件，不隐式转换为mm。弃用导出仅供检查，保留discarded标记；所有批量检查包training_ready=false。

## 3. 用已有ELF3 GR00T环境检查

```bash
/你的GR00T环境/bin/python tools/collection_review/validate_exports.py /你的数据盘/collection_review
```

读取每条episode首、中、尾40步窗口，验证三路视频、34维状态、66维动作和夹爪归一化往返。每次只解码有限窗口，避免整条1080p视频常驻内存；这是采样检查，不是全部窗口验收或训练。loader_results.json逐条记录结果；某条不通过不删除其数据。

正式训练仍需确认示范质量并标注真实任务，使用gr00t_dataset/export_dataset.py对保存的有效数据正式导出。查看源文件、video和Parquet的对应关系，请参照docs/GR00T_N17_DATA_FORMAT.md。
