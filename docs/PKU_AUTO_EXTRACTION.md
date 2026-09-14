# DSEC 迁移后的 PKU 自动续解压

2026-09-13 更新：已完成 3 个完整 PKU Test 序列的 FP32 端到端试运行，并修复 Lightning checkpoint 类方法调用问题。设置、指标与范围见 [试运行结果](PKU_SMOKE_TEST_RESULTS.md)。

2026-09-13 已启动独立后台任务，脚本为 [wait_and_extract_pku.py](../scripts/wait_and_extract_pku.py)。启动 PID 为 430641；实际状态以 `logs/pku_auto_extract/status.json` 为准。无需保持 SSH 或 tmux 连接，但服务器重启后需重新启动任务。

## 触发条件与行为

- 等待 `/srv/datasets/DSEC_DET` 成为指向 `/data/lab_dataset/RGB_DVS_DET/DSEC_DET` 的有效软链接。这对应现有迁移脚本完成复制、校验、删除源目录和建立链接。
- 开始前要求 SSD 可用空间至少为归档解压总量约 51.2 GiB + 20 GiB 预留空间，并检查没有另一个已知的 PKU Python 解压进程。
- 压缩包保留在 `/data/lab_dataset/RGB_DVS_DET/FAOD_PKU_DAVIS_SOD.zip`。
- 解压到 `/home/zhaowenyao24/Conda_prj/lab_dataset/FAOD_PKU_DAVIS_SOD/freq_1_1`。
- 已有文件经长度与 CRC 校验后跳过；不完整文件写入临时文件，校验后原子替换。每个文件写入前再次检查 20 GiB 预留空间。
- 不移动 PEOD，不删除源压缩包，不启动 GPU 验证或训练。

这是进程级后台任务，不是系统开机服务。若迁移失败、链接没有建立，任务保持等待；若遇到解压错误，会记为 failed 并退出。空间检查不能阻止其他用户同时写满磁盘，运行期间仍需关注共享分区使用量。

## 查看状态

从 FAOD 工程根目录执行：

```bash
cat logs/pku_auto_extract/status.json
tail -f logs/pku_auto_extract/worker.log
ps -p "$(cat logs/pku_auto_extract/worker.pid)" -o pid,ppid,stat,etime,args
```

状态包括 `waiting_for_dsec_migration`、`waiting_for_space_or_extractor`、`extracting`、`waiting_for_space`、`complete`、`failed`。`complete` 表示所有归档文件已解压或对已有文件完成 CRC 校验。

仅当原任务已经退出时重新启动：

```bash
mkdir -p logs/pku_auto_extract
nohup /opt/miniconda3/envs/pytorch/bin/python -u -B \
  scripts/wait_and_extract_pku.py \
  >> logs/pku_auto_extract/worker.log 2>&1 < /dev/null &
```

脚本有单实例文件锁，重复启动会退出。不要在本任务运行时另外向同一目标目录启动解压命令。

## 提前做一次测试流程检查

当前目录只有部分 `test/` 序列；26 个序列通过归档文件清单与大小检查，尚未逐一进行 CRC 校验。

其中 `test/001_test_motion_blur` 的全部 7 个归档文件已经逐一 CRC 校验通过，HDF5 可读取：事件为 `[1199,20,260,346]`，RGB 为 `[1200,260,346,3]`，均为 uint8。该长度差来自原始归档，不应自行裁掉一帧来凑齐。

已用当前 DataModule 在 CPU 上成功读取首批：L=11，事件 `[1,20,260,346]`，RGB `[1,3,260,346]`。

已建立独立子集入口 `logs/pku_early_eval/test/001_test_motion_blur`，软链接到这个完整序列；校验记录为 `logs/pku_early_eval/verification.json`。将其他不完整序列排除在 loader 扫描范围之外。自动续解压会跳过这个 CRC 正确的序列，不会改写它。

在确认 GPU 0 空闲后，从工程根目录运行：

```bash
WANDB_MODE=offline /opt/miniconda3/envs/pytorch/bin/python validation.py \
  dataset=pku_fusion \
  dataset.path="$PWD/logs/pku_early_eval" \
  +experiment/pku_fusion=base.yaml \
  checkpoint=checkpoints/pku_fusion.ckpt \
  use_test_set=true \
  hardware.gpus=0 \
  hardware.num_workers.eval=0 \
  batch_size.eval=1 \
  training.precision=32-true \
  logging.validation.high_dim.enable=false \
  wandb.name=pku-single-sequence-pipeline-check \
  > logs/pku_early_eval/evaluation.log 2>&1
```

这会评估单个完整测试序列，而非单个 batch。可用于检查真实数据加载、模型推理、NMS 和指标管线；结果是局部 `test/AP`，不代表正式 Val、完整 Test 或论文精度。不要据此选择模型或调整超参数。正式模型选择等待真正的 `val/` 分区完整解压，并用 `use_test_set=false`。

本任务只准备了上述命令，没有自动启动 GPU 评估。

## 2026-09-14：迁移在 sudo 认证过期后恢复

原迁移脚本逐步调用 sudo，长时间复制后下一步可能再次要求密码。新建 [resume_dsec_migration.sh](../scripts/resume_dsec_migration.sh)，使用一次 `sudo bash` 启动整个脚本，内部不再次调用 sudo。它只处理 DSEC_DET：续传 → 全量 checksum 校验 → 校验无差异后删除源目录并建立链接。PEOD 保持原位。

进入 tmux 后，若停在 `bash <<'BASH'` 输入中，先按 Ctrl+C 返回 shell 提示符，再执行：

```bash
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_FAOD
sudo bash scripts/resume_dsec_migration.sh
```

输入一次密码后，Ctrl+B 然后 D 分离会话。查看日志：

```bash
tail -f logs/dsec_migration/resume.log
```

`Stage 2/3` 校验会读取两端所有文件，可能长时间没有新的进度输出。`xfr#0` 仅表示复制阶段没有传输文件，不代表 checksum 校验完成。校验失败会保留源目录；不要并行运行旧迁移脚本或修改 DSEC 文件。

脚本建立预期链接后，已运行的 PKU 自动解压任务会在下一次检查时继续，无须重新启动解压任务。
