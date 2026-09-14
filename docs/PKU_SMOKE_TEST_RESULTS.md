# PKU 三个完整序列的 FP32 试运行

2026-09-13 已完成真实数据端到端测试，最终进程退出码 0。使用默认 FAOD 原模型和 `checkpoints/pku_fusion.ckpt`；没有训练或修改模型参数。

## 数据与运行设置

| 序列 | 场景 | 事件时刻数 | RGB 帧数 |
| --- | --- | ---: | ---: |
| 001_test_motion_blur | 运动模糊 | 1199 | 1200 |
| 005_test_low_light | 弱光 | 899 | 900 |
| 020_test_normal | 正常 | 811 | 812 |
| 合计 | 3 个完整 Test 序列 | 2909 | 2912 |

三个序列共 21 个文件均与原 ZIP 的长度及 CRC 匹配。通过独立软链接子集 `logs/pku_smoke_three/test/` 隔离其他未完整解压的序列。

使用物理 GPU 1（RTX 4090），`CUDA_VISIBLE_DEVICES=1` 后进程内 GPU 为 0；FP32、batch=1、workers=0、stream 采样、sequence_length=11、W&B offline，关闭高维可视化。共执行 265 个时序 batch；序列尾部由 loader 填充。测试进度条对应的推理与评估阶段约 78 秒，不包含全部启动时间，也不作为独占设备性能基准。进程结束后 GPU 已释放。

## 汇总结果

| 指标 | 原始数值 | 百分数 |
| --- | ---: | ---: |
| test/AP | 0.2278681776 | 22.79% |
| test/AP_50 | 0.5863040549 | 58.63% |
| test/AP_75 | 0.1250516252 | 12.51% |
| test/AP_S | 0.1717228498 | 17.17% |
| test/AP_M | 0.3713930897 | 37.14% |
| test/AP_L | 0.2838151782 | 28.38% |

这是三个序列合并后的局部 Test 指标，不是每个序列指标的算术平均，也不是正式 Val、完整 Test 或论文复现成绩。没有将它用于调参或 checkpoint 选择。

本次确认：真实 HDF5 加载、默认 checkpoint 加载、跨片段循环状态、GPU FP32 前向、DCN、NMS、COCO AP 汇总和离线日志能够完成。尚未证明 AMP 路径、真实数据反向训练和完整分区精度。

## 本次发现并修复的入口问题

首次运行在 checkpoint 加载处失败：Lightning 2.5.5 禁止通过实例调用类方法 `load_from_checkpoint`。此前合成加载检查直接使用类，未覆盖这个 CLI 调用点。

在 `validation.py`、`demo.py`、`train.py` 中统一替换为：

```python
module = type(module).load_from_checkpoint(str(ckpt_path), **{'full_config': config})
```

保留原参数、权重和返回对象使用方式，不改变模型结构或计算。修复后的 validation.py 已完成上述真实数据测试；训练与 demo 的对应调用做了相同修正和语法检查，本轮未独立执行其完整流程。

日志中 BasicSR 未编译备用算子提示和 protobuf 时间 API 的弃用警告未阻止默认路径运行；本次没有为这些非阻塞提示继续改动库版本。

## 证据与重现

- [完整运行日志](../logs/pku_smoke_three/evaluation.log)
- [机器可读结果与完整命令（可提交快照）](reference/pku_smoke_three_results.json)
- [输入文件 CRC 清单（可提交快照）](reference/pku_smoke_three_data_verification.json)
- [首次运行的类方法错误](../logs/pku_smoke_three/attempt1_classmethod_error.log)
- W&B 离线目录：`wandb/offline-run-20260913_225616-a34rcdn8`。

这些日志与软链接是当前服务器的本地产物，未打包数据到仓库；其他机器需先准备相同子集。

```bash
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
/opt/miniconda3/envs/pytorch/bin/python -u -B validation.py \
  dataset=pku_fusion \
  dataset.path="$PWD/logs/pku_smoke_three" \
  +experiment/pku_fusion=base.yaml \
  checkpoint=checkpoints/pku_fusion.ckpt \
  use_test_set=true \
  hardware.gpus=0 \
  hardware.num_workers.eval=0 \
  batch_size.eval=1 \
  training.precision=32-true \
  logging.validation.high_dim.enable=false \
  wandb.name=pku-three-sequences-fp32
```

执行前检查所选物理 GPU 是否空闲。后续完整 Val 和短训练仍按 [验证与短训练操作手册](VALIDATION_AND_SMOKE_TRAINING.md) 推进。
