# FAOD 与 EfficientViT 共用环境配置记录

2026-09-13 更新：已完成 3 个完整 PKU Test 序列的 FP32 端到端试运行，并修复 Lightning checkpoint 类方法调用问题。设置、指标与范围见 [试运行结果](PKU_SMOKE_TEST_RESULTS.md)。

实施日期：2026-09-13。FAOD 基线：`e8666ca536850807173502e6764423135194cf7f`。
环境：`/opt/miniconda3/envs/pytorch`。已完成安装与兼容性修改。

## 环境和依赖

保留现有 EfficientViT 计算与导出栈，补齐 FAOD 依赖。对安装前记录的 169 个包做标准化版本比较，均未改变；新增 45 个直接或间接依赖。见 [版本核对结果](environment/version-verification.json)。

| 组件 | 实际使用版本 | 处理 |
| --- | --- | --- |
| Python | 3.12.2 | 保留 |
| torch / torchvision / torchaudio | 2.5.0 / 0.20.0 / 2.5.0 | 保留 |
| Torch CUDA runtime / cuDNN | 12.1 / 9.1 | 保留；系统 nvcc 12.8 不参与本次编译 |
| NumPy / SciPy | 1.26.4 / 1.13.1 | 保留 |
| timm / OmegaConf | 1.0.15 / 2.3.0 | 保留 |
| ONNX / onnxruntime-gpu / protobuf | 1.17.0 / 1.20.2 / 4.25.3 | 保留 |
| pytorch-lightning | 2.5.5 | 新增，替代上游要求的 1.8.6 |
| torchmetrics / lightning-utilities | 1.6.1 / 0.15.2 | 新增 |
| hydra-core / einops | 1.3.2 / 0.8.1 | 新增 |
| h5py / hdf5plugin | 3.12.1 / 5.0.0 | 新增 |
| wandb | 0.19.11 | 新增，适配上游 0.14 的日志代码 |
| numba / llvmlite | 0.60.0 / 0.43.0 | 新增，与 Python 3.12 / NumPy 1.26 配合 |
| mmengine | 0.10.7 | 提供权重初始化工具，不编译 CUDA 扩展 |
| plotly / bbox-visualizer | 5.24.1 / 0.1.0 | 新增 |
| ipdb / ipython | 0.13.13 / 8.31.0 | 新增；固定 IPython 以保留已有 psutil |

[直接依赖](../requirements/faod-pytorch.txt)、[计算栈约束](../requirements/pytorch-constraints.txt)、[本次新增包精确锁定](../requirements/faod-added-lock.txt) 分开保存。最后一个文件针对当前已有环境，是增量安装清单，不是从空环境完整建环境的清单。

```bash
conda activate /opt/miniconda3/envs/pytorch
# 当前服务器已经完成；以下用于在同一基线环境重现增量安装。
python -m pip install --index-url https://pypi.org/simple \
  -c docs/environment/pytorch-before.txt \
  -r requirements/faod-added-lock.txt
```

本次安装使用官方 PyPI，因为默认镜像出现证书错误；未改全局 pip 配置。
[安装前版本约束](environment/pytorch-before.txt)、[安装前 Conda 清单](environment/conda-before.txt)、[安装后 pip freeze](environment/pytorch-after.txt)、[pip 安装报告](environment/install-report.json) 和 [安装日志](environment/install.log) 可供核查。Conda 包的 freeze 可能使用 `@ file://...` 来源表示；版本核对使用包元数据并按 PEP 440 标准化，避免把来源或版本拼写差异当成版本变更。

## 代码适配与功能契约

### Lightning 2.5 与 W&B

- 删除 Trainer 已移除的 `move_metrics_to_cpu` 参数。原先值为 False，保留指标所在设备的行为。
- 单卡 `strategy=None` 改为 `"auto"`；多卡仍沿用原 DDPStrategy 配置。
- 配置中 `32` / `16` 改成明确的 `"32-true"` / `"16-mixed"`；默认精度选择不变。
- 自定义 logger 的参数处理工具改用 Lightning Fabric 所在位置，版本比较改用 `packaging.version.Version`。
- 适配 W&B 新版 Run、公开的 entity/project/id 属性以及服务启动方式，保留指标命名、图片记录、checkpoint 别名和在线保留策略。
- 离线/禁用模式不再调用在线 artifact 枚举与轮询；离线模式仍调用本地 artifact 记录接口。

### TorchData 旧管线

FAOD 使用的 `MapDataPipe`、`IterDataPipe` 和基础组合操作迁到 PyTorch 自带模块；三个缺失操作由 [datapipes.py](../data/utils/datapipes.py) 提供：索引转迭代、无缓存重复迭代、最长 zip。保持惰性读取、顺序、填充、worker 划分以及原 concat 管线的重复行为。

无需安装 TorchData，也未把所有样本缓存到内存。迁移原因参考 [TorchData 0.10 发布记录](https://github.com/meta-pytorch/data/releases/tag/v0.10.0)。

### MMCV 的可变形卷积

FAOD 虽将封装命名为 `DCNv2Pack`，实际调用的是**无 modulation mask 的 DeformConv2d**。替代实现见 [deform_conv.py](../models/layers/compat/deform_conv.py)，使用 torchvision 已安装的 CUDA 算子。`mask=None` 的语义与 [torchvision 0.20 官方接口](https://docs.pytorch.org/vision/0.20/generated/torchvision.ops.deform_conv2d.html) 一致。

保持：

- 相同 weight 名称与形状、无 bias、ReLU Kaiming 初始化；旧 weight 可 strict 加载。
- `(dy, dx)` 交错偏移、8 个 deform groups、原 stride/padding/dilation/groups。
- 偏移分支及 GELU 不变；混合精度继续由 offset dtype 决定算子 dtype。

`im2col_step` 保留为兼容属性，实际分块由 torchvision 实现管理。没有改成 modulated DCNv2，也不需要 `mmcv`、`mmcv-full` 或 `openmim`。MMCV 初始化工具改由 MMEngine 提供。

### 其他新版接口和缓存

`timm.models.layers` 的相关导入迁到 `timm.layers`；旧 CUDA autocast 上下文迁到 `torch.amp.autocast("cuda", ...)`。移除一个无用的 turtle 导入，避免引入 GUI 依赖。

短训练恢复测试发现 YOLOX 解码网格缓存会留在旧设备。现将推理缓存设为非持久 buffer，并在空间尺寸、设备或 dtype 改变时刷新；训练缓存也检查设备与 dtype。解码公式和 state_dict 参数键不变。网格创建显式使用 `indexing="ij"`。

## 验证结果与重跑

[兼容性回归日志](environment/compatibility-tests.log)：13 项全部通过。

- 新旧 concat / sharded 数据管线在 0 和 2 个 worker 下逐项一致；另外检查样本覆盖与 rank/worker 分配。
- DCN 与独立的双精度 grid_sample 参考实现比较前向及输入、偏移、权重梯度。
- 使用现有 `/opt/miniconda3/envs/OBBDET_ZDS` 的 Torch 1.13.1 + MMCV 1.7.1 导出 5 组真实 CUDA 算子参考。该环境仅用于只读运行，未修改包。FP32 前向最大绝对差为 0；输入/偏移/权重梯度最大差分别约 `7.15e-7` / `4.77e-7` / `0`。零偏移时边界导数不唯一，仅该案例不比较 offset 梯度，仍比较输出、输入和权重梯度。
- 检查 DCN 初始化、strict 参数加载，以及 YOLOX 缓存跨设备、dtype 和尺寸变化后的解码结果。
- 检查 W&B 禁用模式的日志和 pickle 恢复；mock 检查离线 artifact 去重、别名和不访问在线 API。

```bash
# 常规回归：12 项通过；独立 MMCV fixture 未提供时跳过 1 项。
FAOD_STREAM_REFERENCE=tests/compatibility/fixtures/legacy_stream.json \
  python -B -m pytest tests/compatibility/test_upgrade.py -q

# 可选：本服务器上用已有旧 MMCV 环境生成参考，无需在 pytorch 中安装 MMCV。
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/OBBDET_ZDS/bin/python -B \
  tests/compatibility/export_mmcv_reference.py /tmp/faod_mmcv_reference.pt
FAOD_MMCV_REFERENCE=/tmp/faod_mmcv_reference.pt \
FAOD_STREAM_REFERENCE=tests/compatibility/fixtures/legacy_stream.json \
FAOD_TEST_DEVICE=cuda:0 OMP_NUM_THREADS=1 \
  python -B -m pytest tests/compatibility/test_upgrade.py -q -s

# 原 FAOD 短序列训练、推理、梯度和 checkpoint 恢复；GPU 按空闲情况调整。
OMP_NUM_THREADS=1 python -B tests/compatibility/smoke_training.py --device cuda:0
OMP_NUM_THREADS=1 python -B tests/compatibility/smoke_training.py --device cuda:0 --precision 16-mixed
```

短训练使用实际 Darknet + LSTM + 对齐 DCN + cross-CBAM + YOLOX 模型（20,265,335 参数），合成 batch=1、2 个时刻、64×96、20 通道事件与 3 通道 RGB；仅在测试配置中缩小分辨率、关闭 scheduler/训练指标，未更改实际训练配置或数据。

- [FP32](environment/training-fp32.log)：2 步训练，保存模型和 AdamW 状态，恢复至第 3 步；3 次有限梯度更新，推理通过。
- [FP16 mixed](environment/training-fp16.log)：12 步训练并恢复至第 13 步；共 2 次有限梯度更新，推理通过。这个合成测试的 GradScaler 在前 11 步跳过溢出的更新并自动降低缩放值；测试明确断言优化器已有状态、出现有限梯度，避免仅凭 global_step 判断成功。未修改默认 scaler 策略。
- 两种精度都检查 strict state_dict 加载和训练结束迁移到 CPU 后的推理，预测形状 `[1, 126, 13]`、数值有限。
- [共享环境检查](environment/shared-environment-tests.log)：train / validation / demo 入口导入、HDF5 Blosc 压缩往返、用户实际 `HW_Aware_efficientvit` 的 B1 CPU 前向/反向、mqbench_export / ONNX / ONNX Runtime 导入全部通过。

以上验证支持已修改 API 和核心训练链路的兼容性；不代表真实数据集完整训练后的精度或速度已复现。尚未进行真实数据集验证/测试循环、多 GPU DDP、在线 W&B 上传、发布 checkpoint 的优化器续训、量化导出与硬件部署。FP16 与旧 MMCV 的整网训练轨迹也未作逐步对照。

## 已有告警与可选分支

`pip check` 仍报告安装前已有的 `qonnx 1.0.0 requires onnxruntime`，因为安装的是 `onnxruntime-gpu` 发行包；ONNX Runtime 模块可正常导入。还保留了已有的 `~vitop` 无效发行目录告警。未添加另一份 CPU ONNX Runtime 或改动已有导出栈，详见 [pip check 记录](environment/pip-check.txt)。

BasicSR 的 `fused_act` / `upfirdn2d` 可选扩展会提示无法导入；当前 FAOD 主路径不调用这些算子，未编译它们。`cross_mamba` 的 Mamba 扩展和 S5 调试示例的 lovely_tensors 也不属于当前默认模型运行依赖；这些备用分支未被声明为可运行。Python 3.12 自带 StrEnum，不需要补装其兼容包。

本次未移植 EfficientViT 到 FAOD、未重新处理 PEOD；下一步可在已配置的同一环境中开展这两项工作。

## 发布权重补充验证

2026-09-13 随后已下载 README 的三个发布 checkpoint，并通过当前 Lightning 2.5.5
入口严格加载及合成 CPU 推理。原始文件保持不变，旧版元数据在加载时自动迁移。
验证记录见 [checkpoints/README.md](../checkpoints/README.md)；此项不包含发布
checkpoint 的优化器续训或真实数据集 mAP 复测。
