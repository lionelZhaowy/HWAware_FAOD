# FAOD 工程上下文与进度交接

最后整理：**2026-09-14，约 09:23（Asia/Shanghai）**。这是交接快照；新对话首先重新查询后台任务、文件和 GPU 状态，不直接复用这里的 PID 或空间余量。

## 1. 一分钟接手摘要

用户正在 GPU 服务器上，以原 FAOD 为起点，准备整合已在 Dremi 上验证的 MIT EfficientViT，并最终面向 Gray–DVS 多速率检测等任务。**当前仍在原 FAOD 环境、数据和基线验证阶段，尚未实施新模型移植。**

已完成共用 Conda 环境适配、三个官方 checkpoint 下载与加载检查、全面工程教程、PKU 三个完整 Test 序列的 FP32 试运行。三个官方数据压缩包现已全部下载并通过 CRC；PKU 只解压了一部分。当前用户在 tmux 中迁移旧 DSEC 数据以腾出 SSD 空间，之后后台任务会自动续解压 PKU。

本轮新请求是保存上下文并完善根 `.gitignore`。没有启动新训练、迁移或解压，也没有提交 Git。

## 2. 用户决定与长期目标

- 共用 `/opt/miniconda3/envs/pytorch`，尽量与 EfficientViT 保持版本一致；已决定修改 FAOD 适配新版库，而非另建旧环境。
- 先让原 FAOD 跑通完整验证与小幅训练，再做正式算法修改。局部测试通过不等于该阶段全部完成。
- 后续使用用户的 **`HW_Aware_efficientvit`** 版本；v1.0/v1.1 是备份。需现场定位实际目录与提交，不凭名称认定与芯片部署逐层一致。
- PEOD 当前 RENet 预处理只是初步实验，可以为 FAOD 重新处理；原说明位于 `/home/zhaowenyao24/Conda_prj/Detection_DVS/RENet_PEOD/docs/`，重点为 `PEOD_RENET_HANDOFF.md`、`PEOD_240HZ_USAGE.md`。
- 最新存储决定：**只把 `/srv/datasets/DSEC_DET` 迁至 HDD，PEOD 保持原位**。用户消息有时写 DESC/DVS_DET，服务器真实目录是 DSEC_DET。
- 长期算法需求全文已归档：[ORIGINAL_ALGORITHM_BRIEF.md](ORIGINAL_ALGORITHM_BRIEF.md)。其中包含 Gray 1024×720/30 Hz、DVS 512×360/240 Hz、无除法 LiteMLA、4/8 帧有限 FIFO、CenterNet 式头、SOT/分割扩展等约束。这些是目标/设计需求，**不是已完成结果**。
- 实验室时间表面的极性通道、衰减和 tau 尚待确定；不能把 FAOD 的 20 通道计数表征或 PEOD 红绿可视化图称为最终实验室输入。

## 3. 仓库与环境

| 项目 | 当前约定 |
| --- | --- |
| 工程目录 | `/home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_FAOD` |
| Git HEAD | `0e6cf34322c04a0a0d0bf498e57f55d21cb1a701`，另有用户和本次会话的未提交修改 |
| 原上游参考 | `e8666ca536850807173502e6764423135194cf7f` |
| Python | `/opt/miniconda3/envs/pytorch/bin/python`，3.12.2 |
| PyTorch / torchvision | 2.5.0 / 0.20.0，Torch CUDA 12.1 |
| Lightning | 2.5.5，原项目使用 1.8.6 |
| 常用配置 | `dataset=pku_fusion +experiment/pku_fusion=base.yaml` |
| GPU | 多张共享 RTX 4090；运行前用 nvidia-smi 查空闲卡，不假定 GPU 0 可用 |

环境适配已补齐 45 个包，安装前记录的 169 个包版本保持一致。使用 torch 核心/本地实现替代所用 TorchData 功能，以 torchvision 兼容封装替代实际使用的 MMCV deform conv；还做了 timm、AMP、Lightning、日志相关适配。具体依赖、约束、测试与残余问题见 [ENVIRONMENT.md](ENVIRONMENT.md) 和 `requirements/`。

已有 13 项兼容性检查、FP32/FP16 合成训练和重载记录；这些不能代替完整真实数据基线。重跑相关检查使用环境文档中的命令，尤其不要遗漏流式 fixture 配置。

## 4. 已完成工作与证据

| 工作 | 状态 | 入口 / 证据 |
| --- | --- | --- |
| 新库环境配置 | 完成默认路径适配 | [ENVIRONMENT.md](ENVIRONMENT.md)、`docs/environment/` |
| checkpoint 下载 | 3 个文件完成，CRC/SHA256 及严格加载检查 | [权重清单快照](reference/checkpoints_manifest.json)、[验证快照](reference/checkpoints_verification.json) |
| 相对 checkpoint 路径 | validation/demo 支持工程根目录解析 | `utils/checkpoints.py`、`config/val.yaml` |
| 新版 Lightning 类方法调用 | validation/train/demo 改用 `type(module).load_from_checkpoint(...)` | 真实测试首轮暴露并修复；train/demo 未独立完成全流程测试 |
| 工程解析 | 16 章教程 + 全源码配置索引 | [FAOD_PROJECT_GUIDE.md](FAOD_PROJECT_GUIDE.md)、[FAOD_SOURCE_INDEX.md](FAOD_SOURCE_INDEX.md) |
| 模型结构检查 | PKU/DSEC CPU 合成前向成功 | `scripts/inspect_faod_model.py`、`docs/reference/*_model_trace.json` |
| 真实数据 FP32 试运行 | 3 个完整 PKU Test 序列通过 | [PKU_SMOKE_TEST_RESULTS.md](PKU_SMOKE_TEST_RESULTS.md) |
| 完整 Val / Test | **未完成** | 待相应分区完整解压 |
| 真实数据短训练 | **未执行** | [VALIDATION_AND_SMOKE_TRAINING.md](VALIDATION_AND_SMOKE_TRAINING.md) |
| EfficientViT / 新 FIFO / PEOD 模型接入 | **未实施** | 仅完成背景与原 FAOD 解析 |

### checkpoint 使用

本机权重位于 `checkpoints/`，用户要求整个目录不上传 Git：

- `pku_fusion.ckpt`：普通 PKU，3 类。
- `pku_fusion_time_shift.ckpt`：Time Shift 版本，首轮普通基线不用它替换。
- `dsec.ckpt`：DSEC，8 类。

每个文件约 244 MB；校验值和来源已复制到 docs/reference，克隆仓库后不能假定二进制文件存在。旧 Lightning checkpoint 的格式迁移在加载时完成，未改写发布权重文件。

### PKU 三序列真实测试

2026-09-13 使用物理 GPU 1、FP32、batch=1、workers=0、原模型与普通 PKU checkpoint、W&B offline 完成：

| 序列 | 事件时刻 |
| --- | ---: |
| `001_test_motion_blur` | 1199 |
| `005_test_low_light` | 899 |
| `020_test_normal` | 811 |

共 2909 个事件时刻、265 个时序 batch；21 个输入文件均逐个 CRC 校验。退出码 0，GPU 已释放。合并后的局部指标：AP=0.2278681776、AP50=0.5863040549、AP75=0.1250516252。**不是完整测试集指标，不能据此调参、选权重或宣称论文复现。**

机器可读结果：[pku_smoke_three_results.json](reference/pku_smoke_three_results.json)，输入清单：[pku_smoke_three_data_verification.json](reference/pku_smoke_three_data_verification.json)。原始日志和软链接子集位于本机 `logs/pku_smoke_three/`，Git 忽略，不随仓库分发。

## 5. 数据下载与存储布局

下载器通过匿名 OneDrive/SharePoint 共享链接，保留 Cookie 并使用 download=1，无需服务器登录个人 OneDrive。不要在新对话重新尝试租户账号登录问题。见 [DATASET_DOWNLOAD.md](DATASET_DOWNLOAD.md)。

2026-09-14 现场读取状态：**三个压缩包都 complete、crc_verified=true**，下载任务整体结束于 04:56。DSEC 状态里留有历史 ConnectionError 字段，不代表最终下载失败。

| 压缩包（均在 `/data/lab_dataset/RGB_DVS_DET/`） | 文件字节数 |
| --- | ---: |
| `FAOD_PKU_DAVIS_SOD.zip` | 54,486,709,692 |
| `FAOD_DSEC_Detection.zip` | 86,801,382,924 |
| `FAOD_EOD200.zip` | 47,797,567,175 |

PKU ZIP 顶层为 `freq_1_1`，解压约 51.2 GiB。目标数据根目录是：

```text
/home/zhaowenyao24/Conda_prj/lab_dataset/FAOD_PKU_DAVIS_SOD/freq_1_1
```

`/srv/datasets` 和 `/home` 在同一 SSD 分区 `/dev/sda2`；`/data` 为 HDD 分区。归档保留在 HDD，仅将 PKU 解压训练数据放 SSD。DSEC/EOD200 新压缩包本轮没有安排解压。

注意区分：

- 旧处理数据 `/srv/datasets/DSEC_DET`（约 73 GiB）正在迁移到 `/data/lab_dataset/RGB_DVS_DET/DSEC_DET`。
- 新 FAOD DSEC 数据是 `FAOD_DSEC_Detection.zip`；不是上述旧目录。
- HDD 已有 `DSEC_DET_orig`、`PEOD_orig` 等目录，不应覆盖、合并或删除。
- `/srv/datasets/PEOD`（约 58 GiB）**不迁移**。
- `/home/zhaowenyao24/Conda_prj/lab_dataset/DSEC_DET`、`PEOD` 是指向 `/srv/datasets/...` 的原有软链接。

## 6. 正在运行的任务：先核查，不重复启动

### A. 旧 DSEC 迁移

现场状态：tmux `faod-data-migrate` 中运行 `scripts/resume_dsec_migration.sh`，复制扫描已结束，进入 **Stage 2/3 全量 checksum 校验**。此步骤可能长时间无新输出；进程中 `rsync -aHAXnci --delete` 的 `-n` 表示 dry-run，不实际删除目标。

原脚本每步分别 sudo，因认证缓存过期要求第二次密码而中断。新版用一次 `sudo bash` 启动整个 root 脚本，后续无须再 sudo。

```bash
tmux attach -t faod-data-migrate
tail -f logs/dsec_migration/resume.log
pgrep -af rsync
ls -ld /srv/datasets/DSEC_DET
```

只有确认当前迁移已退出才考虑恢复：在 tmux 中从工程目录执行 `sudo bash scripts/resume_dsec_migration.sh`。若终端卡在未完成的 heredoc 输入，先 Ctrl+C 返回提示符。输入一次密码后用 Ctrl+B、D 分离，勿用 Ctrl+C 停止正在工作的迁移。

脚本顺序：续传 → 完整内容校验 → 无差异才删除 SSD 源目录 → 在旧位置建到 HDD 的软链接。复制期间不会逐文件释放 SSD 空间。校验差异写入 `logs/dsec_migration/verification.diff`，非空则保留源目录并停止；不要为加快迁移跳过校验。

### B. PKU 自动续解压

脚本：`scripts/wait_and_extract_pku.py`；启动 PID **430641**（须重查）。现场状态为 `waiting_for_dsec_migration`，独立后台进程仍存活。

```bash
cat logs/pku_auto_extract/status.json
tail -f logs/pku_auto_extract/worker.log
ps -p "$(cat logs/pku_auto_extract/worker.pid)" -o pid,ppid,stat,etime,args
```

每 60 秒检查：旧 DSEC 路径变为指向预期 HDD 目录的有效链接；SSD 至少有全量 PKU 解压空间 + 20 GiB；无冲突的已知 PKU 解压进程。满足后自动续解压。已有文件检查长度和 CRC，正确则不重写；不完整文件写临时文件后原子替换。

任务不依赖终端连接，但**不是开机服务**。服务器重启或 failed 退出后的恢复命令见 [PKU_AUTO_EXTRACTION.md](PKU_AUTO_EXTRACTION.md)。不要同时向同一目标运行第二个解压进程。

旧的 `python -m zipfile -e ...` 解压进程已退出；此前的 PID 407420 不能继续当作当前进程使用。

## 7. 实现中需要记住的边界

详细解释以工程指南为准，接手时至少记住：

- 配置目录名 `maxvit_yolox` 不代表默认 MaxViT；当前是 Darknet + 四阶段 ConvLSTM + Align/EF Fusion + PAFPN/YOLOX。
- 输入事件是双极性 × 10 桶的 20 通道计数直方图；RGB 不自动除以 255。输入范围与时间索引要与发布权重匹配。
- 原路径存在 AdaIN、softmax、sigmoid/tanh、DCN，不能直接称为符合未来 Dremi 部署约束的新网络。
- 默认 random/stream/mixed 状态和时间索引约定较特殊：random 也读取 `stream.unpair`；不能仅改 `random.unpair` 做时移消融。
- 原网络的部分备用开关有未完成分支，尤其关闭 align、替换 memory、改维度/深度；不要直接把配置选项当作已验证功能。
- 正式评估使用 `validation.py`，根 `test.py` 是历史 Module 文件。默认 `use_test_set=true`；真正 Val 显式 false。
- 独立 validation.py 没有接入 limit_val_batches，缩短测试通过完整序列子集实现。训练入口才接入该限制。
- 独立评估入口限定单 GPU；共享 Module 的多卡指标同步也不等于汇总全局预测后计算 AP。
- train 尚无与 validation 等价的纯本地 checkpoint 初始化入口；`wandb.artifact_local_file` 单独设置不触发加载。短训练文档当前安排的是随机初始化，不是假装已加载发布权重的微调。
- seed_everything 在 mixed/stream 训练入口中被主动禁用，不能声称仅设置 seed 即保证逐次复现。

## 8. 下一步的合理顺序

1. 核查迁移与自动解压，确认任务实际完成及磁盘余量。
2. 确认 PKU 真正的 `val/` 全分区完整，再用普通 checkpoint 做单 GPU FP32 完整 Val。
3. 同分区同权重做 `16-mixed` 对照，检查运行与指标差异。
4. 原模型真实数据 100 步短训练，验证反传、参数更新、一次验证和 checkpoint 保存，再回读评估。
5. 若用户要从发布权重微调，先补齐明确的纯本地参数初始化入口，不自动恢复原训练计数。
6. 建立可重复基线后，按最新用户任务开始 EfficientViT / PEOD / 新时序模块改造；不自动展开全部长期需求。

以上是后续路线，**不是声称这些任务已启动或已完成**。用户在新对话提出的明确优先级优先。

## 9. Git 与文档约定

用户已将 `.gitignore` 移至工程根目录，明确要求 logs/checkpoints/wandb 不上传。本轮在此基础上补充 Hydra 输出、缓存、构建产物、权重导出、数据归档、大型事件数据、临时文件和本地凭据等忽略规则。

`data/` 是代码，`config/`、`requirements/`、`tests/`、`docs/` 中的源文件/清单/小型证据应提交。不要泛化忽略所有 JSON/YAML/NPY/NPZ 或图片。文档中的本机 logs/checkpoints 链接在新克隆环境可能不存在；必要摘要已归档到 docs/reference。

`.gitignore` 对已经跟踪的文件不生效；已有缓存/日志等历史文件未自动从索引移除，也未删除磁盘文件。本轮未执行 commit/push 或历史清理。提交前用 git status / git diff 核对用户修改，并按需要单独安排跟踪产物清理。

本轮 `git check-ignore --no-index` 验证了 13 个应忽略路径和 13 个应保留路径，均符合预期。`git ls-files -ci --exclude-standard` 发现 521 个已跟踪文件受新规则覆盖，其中 510 个是 Python 字节码；其余包括 Hydra 快照、本地配置、日志等。该数字是本轮快照，不表示这些文件已经停止跟踪。
