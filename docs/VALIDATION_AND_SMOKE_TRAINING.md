# 正式修改前：原 FAOD 验证与短训练

2026-09-13 更新：已完成 3 个完整 PKU Test 序列的 FP32 端到端试运行，并修复 Lightning checkpoint 类方法调用问题。设置、指标与范围见 [试运行结果](PKU_SMOKE_TEST_RESULTS.md)。

建议顺序：已发布权重的完整 Val → 相同设置下的 FP16 Val → 原模型 100 步短训练 → 回读新 checkpoint。以最先完成下载的普通 PKU-DAVIS-SOD 为起点，不必等 DSEC 和 EOD200 全部下载。此流程不移植 EfficientViT，也不改 PEOD。

目前已经通过：依赖/算子回归、三个发布权重的 strict 加载与合成 CPU 推理、合成序列 FP32/FP16 训练及恢复。下一阶段要增加真实 HDF5 数据加载、时序状态、后处理、AP 评估和训练入口的组合验证。以下命令的 Hydra 配置已解析检查，但真实数据运行尚未执行。

## 1. 数据就绪与共同设置

服务器更新：PKU 压缩包已下载并校验完成；已安排在 DSEC 迁出 SSD 后自动续解压到 `/home/zhaowenyao24/Conda_prj/lab_dataset/FAOD_PKU_DAVIS_SOD/freq_1_1`。任务监控、恢复方法和提前单序列测试命令见 [PKU 自动解压](PKU_AUTO_EXTRACTION.md)。下述 `/data` 解压路径是原通用方案，当前服务器正式运行请将 `FAOD_PKU_ROOT` 设置为上述 SSD 路径。

等待 `/data/lab_dataset/RGB_DVS_DET/FAOD_PKU_DAVIS_SOD.zip` 出现；`.zip.part` 表示尚未完成下载或校验。完整包经下载程序校验后再解压到 `/data`，先用 `unzip -l` 确认归档目录结构，避免把数据解到空间不足的工程分区。

`dataset.path` 应指向包含 `train/`、`val/`、`test/` 的根目录，通常名为 `freq_1_1`，不能指向 zip、父级数据集集合目录或单个视频序列。每个序列应包含：

```text
freq_1_1/
  train/、val/、test/
    <sequence>/
      event_representations_v2/stacked_histogram_dt=50_nbins=10/
        event_representations.h5
        objframe_idx_2_repr_idx.npy
      labels_v2/
        images.h5
        labels.npz
```

下述设置在同一个终端执行；替换实际解压路径，GPU 0 按当时空闲情况更换。

```bash
conda activate /opt/miniconda3/envs/pytorch
cd /home/zhaowenyao24/Conda_prj/Detection_DVS/HWAware_FAOD
export FAOD_PKU_ROOT='/实际解压后的路径/freq_1_1'
export WANDB_MODE=offline
mkdir -p logs
set -o pipefail
```

不改事件通道数、序列长度、输入分辨率、时间对齐或后处理阈值。先使用 FP32、单卡、小 batch，并关闭高维可视化日志。

## 2. 发布权重的完整 Val

```bash
python validation.py \
  dataset=pku_fusion \
  dataset.path="$FAOD_PKU_ROOT" \
  +experiment/pku_fusion=base.yaml \
  checkpoint=checkpoints/pku_fusion.ckpt \
  use_test_set=false \
  hardware.gpus=0 \
  hardware.num_workers.eval=0 \
  batch_size.eval=2 \
  training.precision=32-true \
  logging.train.high_dim.enable=false \
  logging.validation.high_dim.enable=false \
  2>&1 | tee logs/pku_val_fp32.log
```

显式 checkpoint 便于核查；省略时也会按 dataset 选择同一个文件。
`use_test_set=false` 才调用 `Trainer.validate` 并读取 `val/`；默认 True 会读取 `test/`，调用 `Trainer.test`。脚本名称 `validation.py` 本身不决定分区。

第一轮将 eval workers 设为 0，使数据异常在主进程中直接显示；通过后可改为 2。
当前独立 `validation.py` 没有把 `validation.limit_val_batches` 传给 Trainer，也没有 test 批次数限制；因此上述命令执行完整 Val。不要通过添加该参数来假定只会跑几批。若需要先做真实数据的少量批次检查，应先接入相应 Trainer 参数或构造独立的数据子集目录，不能改动完整数据集目录。

验收：

- 正确加载普通 PKU 权重，检测头为 3 类，没有 missing/unexpected keys 或维度错误。
- Val 执行完毕，产生 `val/AP`、`val/AP_50`、`val/AP_75` 等指标；没有 HDF5、NMS、DCN、device/dtype 或时序状态异常。
- 有效总体指标正常；AP 始终为 0 时先排查标签、预测及对齐。某一尺寸类别没有样本时，该子指标可能为 -1，不能据此单独判定环境错误。
- 记录完整命令、代码版本、数据分区、权重文件与 SHA-256，以及运行日志。

然后将同一命令中的 `training.precision=32-true` 改为 `16-mixed`，日志改为 `logs/pku_val_fp16.log`。保持其余条件相同，比较 AP 和是否出现 NaN/Inf。FP16 可有数值差异；明显偏差应先定位，暂不进入算法修改。

## 3. 区分环境验证与论文指标复现

代码的 COCO AP 通常在 0–1 范围内，例如 0.305 对应 30.5%。README 标注普通 PKU 为 30.5、DSEC 为 42.5，Time Shift PKU 为 29.7；这些是作者公布值。

当前 `val/` 的结果不能直接要求等于 README。若核对作者测试结果，应在确认相同数据版本、测试分区和评估协议后，用同一条命令改成 `use_test_set=true`，查看 `test/AP`。Time Shift 权重及错位实验暂不纳入首轮基线。

完整运行成功并不单独证明新旧库在真实数据上的指标完全一致；严格判断还需相同样本、权重和配置下的参考结果。已有 DCN 数值对照和权重 strict 加载提供了算子与参数层面的证据。

## 4. 原模型 100 步短训练

这一轮使用**随机初始化的原 FAOD 模型**，检查数据 → loss → backward → AdamW → 验证 → 保存链路，不期待 100 步获得高 AP。当前 `train.py` 没有与验证入口等价的纯本地 checkpoint 参数：只给 `wandb.artifact_local_file` 不会触发加载，设置 artifact_name 又会访问 W&B artifact。不要用 `+checkpoint=...` 误以为已加载发布权重。

若下一阶段选择从发布权重微调，应先补充明确的本地初始化入口：加载模型参数，使用新的优化器、scheduler 和 global_step；不要直接恢复发布 checkpoint 已完成的训练步数及优化器状态。这是微调路径的准备工作，不是这条随机初始化命令的行为。

```bash
python train.py \
  dataset=pku_fusion \
  dataset.path="$FAOD_PKU_ROOT" \
  +experiment/pku_fusion=base.yaml \
  hardware.gpus=0 \
  hardware.num_workers.train=2 \
  hardware.num_workers.eval=0 \
  batch_size.train=2 \
  batch_size.eval=2 \
  training.precision=32-true \
  training.max_steps=100 \
  training.max_epochs=1 \
  training.limit_train_batches=100 \
  training.lr_scheduler.use=false \
  validation.val_check_interval=100 \
  validation.check_val_every_n_epoch=null \
  validation.limit_val_batches=20 \
  logging.train.log_every_n_steps=10 \
  logging.train.high_dim.enable=false \
  logging.validation.high_dim.enable=false \
  wandb.name=faod-pku-smoke100 \
  2>&1 | tee logs/pku_train_smoke100.log
```

选择理由：

- 保留默认 `mixed` 采样；它要求 train batch 和 train workers 都至少为 2。不能只把 train workers 改成 0。如果要单进程定位问题，需同时改成 `dataset.train.sampling=random`，这时检查的是 random 分支。
- 100 个训练 batch 与 100 步上限使首轮有明确边界；第 100 步触发 20 个验证 batch。训练入口确实接入了 `validation.limit_val_batches`，与独立 validation.py 不同。
- 显式关闭 OneCycleLR，避免把原来的长程 scheduler 直接压缩到 100 步。使用现有实验配置的固定学习率。
- 验证产生 `val/AP` 后，原 ModelCheckpoint 回调才有监控指标，可以保存 best/last。这里只验证部分 Val，其 AP 不能与完整 Val 比较。
- 完成位置和文件名以训练日志为准。原回调的 last 文件名类似 `last_epoch=000-step=100.ckpt`，不是固定 `last.ckpt`；W&B 离线运行也会保存本地 checkpoint。

验收不以 loss 单调下降为标准，而是确认：完成预期优化步骤、loss/梯度正常、模型参数实际改变、AdamW 状态存在、验证结束、checkpoint 保存成功。然后使用步骤 2 的 Val 命令，把 `checkpoint` 改成刚保存文件的绝对路径，确认新文件能够回读评估。

FP32 全链路通过后，再做一轮独立的 `16-mixed` 短训练；留意 GradScaler 可能跳过初始溢出更新，不能只看 global_step 就判断更新成功。已有合成训练测试专门检查过有效梯度与 optimizer state。

## 5. 进入正式修改前保留的基线

保留普通 PKU 的 FP32/FP16 完整 Val 日志、必要时的完整 Test 日志、短训练配置与输出 checkpoint，以及重新加载新 checkpoint 的验证结果。后续移植 EfficientViT 或改动事件表征时，一次只引入一类变化，以同一数据与评估配置比较。

DSEC 就绪后可将 Val 命令中的 dataset 改为 `dsec`，实验项改为 `+experiment/dsec=base.yaml`，路径改为 DSEC 根目录，权重改为 `checkpoints/dsec.ckpt`。不要只替换 checkpoint 而保留 PKU 的 3 类模型配置。
