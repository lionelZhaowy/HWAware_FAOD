# PKU 解压完成与真实数据短训练结果

**2026-09-14 已完成：PKU 全量解压 → 原 FAOD 随机初始化 100 步 FP32 训练 → 20 个验证 batch → checkpoint 保存 → 独立进程回读并验证。** 两次最终运行均退出码 0。未进行 EfficientViT/PEOD 算法移植。

## 1. 数据已就绪

旧 `/srv/datasets/DSEC_DET` 已迁至 `/data/lab_dataset/RGB_DVS_DET/DSEC_DET` 并建立软链接，PEOD 保留原位。PKU 自动续解压于 **10:47:54 +0800** 报告 complete：

- 1540 个文件完成 CRC 校验，其中已有 185 个文件校验后跳过、1355 个文件完成写入与校验。
- 额外核对所有文件的存在性和大小，均匹配 ZIP 清单。
- train=132、val=44、test=44 个完整序列。
- 数据根目录：`/home/zhaowenyao24/Conda_prj/lab_dataset/FAOD_PKU_DAVIS_SOD/freq_1_1`。

证据：[解压完成记录](../codex_artifacts/reference/pku_extraction_complete.json)。无需重启旧迁移/解压任务。

## 2. 本次训练设置

按用户最新要求，在解压完成后先做训练流程检查；官方权重的完整 Val 仍待执行。

| 项 | 本次设置 |
| --- | --- |
| 模型与初始化 | 原 FAOD PKU base，随机初始化，没有加载发布权重 |
| 环境 / GPU | 共用 pytorch 环境，物理 RTX 4090 GPU 1，进程内 GPU 0 |
| 精度 | `32-true`；TF32 设置沿用原入口默认值 |
| 数据入口 | 完整 PKU train/val 分区，实际仅运行限定的 batch 数 |
| 采样 | 默认 mixed，stream/random 各 batch 1、worker 1 |
| 序列长度 | 11 |
| batch / workers | train=2/2，eval=2/0 |
| 优化 | AdamW，LR=0.00015，weight_decay=0；关闭 OneCycle |
| 训练限制 | max_steps=100、max_epochs=1、limit_train_batches=100 |
| 验证 | 训练前默认 2 个 sanity batch；第 100 步运行 20 个 Val batch |
| 日志 | W&B offline，关闭高维可视化，每 10 步记录 |

新增 [smoke_train_with_snapshot.py](../scripts/smoke_train_with_snapshot.py) 仅在 `fetch_model_module` 创建模型后保存初始参数，然后调用原 `train.main` 主流程。它不替换 Trainer、模型、loss、优化器、采样或随机种子。初始参数用于训练后比对，避免仅凭 global_step 判断参数更新。

成功训练时间为 10:49:00–10:51:12，包含启动、数据准备和保存；进度条中的训练/验证阶段约 109 秒。并非独占设备性能基准。

## 3. 训练与保存验收

| 检查 | 实际结果 |
| --- | --- |
| 训练进程 | 退出码 0，global_step=100 |
| 参数变化 | 全部 547 个参数张量相对初始值发生变化；总差值 L2≈5.79824 |
| 参数有效性 | 最终所有参数有限，无 NaN/Inf |
| AdamW 状态 | 547 组状态均记录 step=100；一阶矩均非零，状态张量均有限 |
| loss | 10 个已记录的 step loss 均有限；首个记录点 13.75676，最后记录点 8.79429 |
| 验证 | 20 个 batch 完成，记录 val/AP 等指标 |
| checkpoint | best 与 last 均成功保存，每个约 244 MB |

这里只检查了已记录的 loss 点及最终参数/优化器状态，没有声称保存了每个 step 的全部梯度。loss 的首末变化不是收敛证明。

训练中的局部 `val/AP=0.0002781854`（约 0.0278%），`val/AP_50=0.0015525377`。随机初始化仅训练 100 步，结果很低不代表流程失败，也不能作为精度基线。`AP_L=-1` 表示该评估子集没有适用的大目标样本。

checkpoint 路径：

```text
FAOD/tvj1jsxb/checkpoints/epoch=000-step=100-val_AP=0.00.ckpt
FAOD/tvj1jsxb/checkpoints/last_epoch=000-step=100.ckpt
```

文件名的 AP 仅保留两位小数，`0.00` 不意味着实际记录值恰为 0。二进制 checkpoint 与原始日志保持本机存储，由 `.gitignore` 忽略。

## 4. checkpoint 独立回读

使用新的 `validation.py` 进程，从上述 last checkpoint 加载模型，在真实 **Val** 序列 `005_val_normal` 上运行。这个完整序列有 413 个事件时刻、414 张 RGB，按 batch=1、L=11 共执行 38 个时序 batch。

最终退出码 0，加载、时序前向、NMS、AP 汇总均完成。该序列 `val/AP=0`，仍仅用于证明模型保存后可回读运行，不用于调参或报告最终精度。本轮未测试从 checkpoint 恢复优化器并继续训练。

回读时发现并修复一处新版 Lightning 兼容问题：`VizCallbackBase.on_validation_batch_end` 的 `dataloader_idx` 需要默认值 `0`。单个验证 loader 下 Lightning 可以省略该参数；原实现把它列为必填，导致独立 Val 入口失败。改动仅为 [viz_base.py](../callbacks/viz_base.py) 的参数默认值，不改变模型计算。训练关闭了高维可视化，因此此前训练中没有实例化该回调；Test 入口也没有触发这个 Val hook。

此外，含 `=` 的 checkpoint 文件名必须保留 Hydra 的内部引号：

```bash
'checkpoint="FAOD/tvj1jsxb/checkpoints/last_epoch=000-step=100.ckpt"'
```

只给 shell 普通引号、让 Hydra 收到未加引号的多个等号，会产生解析错误。这是命令行转义问题，不是权重损坏。

## 5. 可复用命令与证据

在工程根目录、确认所选 GPU 空闲后，可按原短训练手册重跑。若还需要初始参数快照，将 `python train.py` 替换为：

```bash
FAOD_INITIAL_SNAPSHOT="$PWD/logs/新运行目录/initial_parameters.pt" \
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
/opt/miniconda3/envs/pytorch/bin/python -u -B scripts/smoke_train_with_snapshot.py \
  dataset=pku_fusion \
  dataset.path=/home/zhaowenyao24/Conda_prj/lab_dataset/FAOD_PKU_DAVIS_SOD/freq_1_1 \
  +experiment/pku_fusion=base.yaml \
  hardware.gpus=0 hardware.num_workers.train=2 hardware.num_workers.eval=0 \
  batch_size.train=2 batch_size.eval=2 training.precision=32-true \
  training.max_steps=100 training.max_epochs=1 training.limit_train_batches=100 \
  training.lr_scheduler.use=false \
  validation.val_check_interval=100 validation.check_val_every_n_epoch=null \
  validation.limit_val_batches=20 logging.train.log_every_n_steps=10 \
  logging.train.high_dim.enable=false logging.validation.high_dim.enable=false \
  wandb.name=pku-real-smoke100-fp32
```

快照路径必须不存在，包装器会拒绝覆盖已有初始参数。最初的包装器尝试因导入入口后的 Hydra 配置定位失败，在开始训练前退出；现已显式指定工程 config 目录。

- [参数、优化器、loss 和训练配置审计](../codex_artifacts/reference/pku_train_smoke100_audit.json)
- [离线日志提取的训练/验证数值](../codex_artifacts/reference/pku_train_smoke100_history.json)
- [独立回读结果](../codex_artifacts/reference/pku_train_smoke100_reload.json)
- 本机训练日志：`logs/pku_train_smoke100/attempt2/training.log`
- 本机初始参数：`logs/pku_train_smoke100/attempt2/initial_parameters.pt`
- 本机回读日志：`logs/pku_train_smoke100/reload.log`
- 本机 W&B：`wandb/offline-run-20260914_104909-tvj1jsxb`

## 6. 尚未完成的验证

目前证明了原模型 **真实数据 FP32 短训练与保存后推理链路可运行**。完整 Val/Test 精度、真实数据 AMP 训练、长期收敛、多卡训练、本地 checkpoint 续训仍未在本轮验证。下一步可先建立官方权重完整 Val 基线，再做 AMP 对照；不需要因为流程已跑通就直接开始长期训练或全量算法移植。
