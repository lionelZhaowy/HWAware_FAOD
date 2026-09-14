# FAOD 源码与配置索引

主教程：[工程入门与论文映射](FAOD_PROJECT_GUIDE.md)。

本索引在 2026-09-13 对当前工作区的 **279 个 Python 文件、43,137 行源码** 做 AST 静态扫描；语法错误 0 个。含注释、空行和新增的结构查看工具；不含 `.pyc`、模型权重及下载数据。基线 Git 提交：`0e6cf34322c04a0a0d0bf498e57f55d21cb1a701`，另有工作区修改。

表中列出顶层符号及各类前几个方法，方便定位；这不是所有备用分支均能运行的声明。方法数较多时省略后续名称，应打开文件查阅。主路径已按调用关系详细解释；BasicSR、MaxViT/Swin 工具库和其他备用实验代码按目录归类。它们可能被导入，但不一定进入默认模型计算图。

## 阅读优先级

1. `config/` → `train.py` / `validation.py` → `modules/utils/fetch.py`。
2. `data/data_module/`、`data/ev_img_dataloader/`、`data/utils/stream_*`。
3. `modules/detection_fusion.py` → `models/detection/yolox_extension/` → 默认 recurrent backbone。
4. `models/layers/align_and_fusion/`、`models/layers/rnn.py`、YOLOX head/loss。
5. 评估、日志和 callbacks；最后阅读备用 backbone、SSM 与 BasicSR 的其余内容。

## 目录职责

| 范围 | 作用与使用边界 |
| --- | --- |
| 工程根目录 | train/validation/demo 是主要入口；test.py 是历史 Module 实现，并非当前测试 CLI |
| config | Hydra 配置和动态尺寸/类别填充 |
| data | 运行时 HDF5 加载、标签、采样与增强；source_dataset_process 是原始格式转换工具 |
| frame_construction | 离线生成事件表征及频率不匹配实验数据；其中 representations 与 data 下存在副本 |
| modules | Lightning 编排；fusion 为默认，event/frame 为对照，modules/data/genx 为历史路径 |
| models/detection | 模型装配、主干工厂、PAFPN、YOLOX 检测头 |
| models/layers/align_and_fusion | 默认 EgF/AdaIN/DCN/跨 CBAM，以及备用融合方法 |
| models/layers/darknet | 默认主干用到的卷积、Focus、CSP、SPP 组件；同目录 backbone.py 不等于当前 recurrent backbone |
| models/layers/maxvit、swins、s5 | 备用空间/时间模块；部分可能有导入依赖，需按实际实例化路径判断 |
| basicsr | 嵌入的图像恢复工具库，主路径借用对齐算子等少量组件；不是 FAOD 的训练框架 |
| callbacks、loggers | checkpoint、可视化、梯度图与 W&B |
| utils | padding、checkpoint 路径、评估、计时 |
| visualization | 直接查看数据；不等于模型评估 |
| tests/compatibility | 此前库升级的回归与合成训练检查 |
| scripts | 服务器数据下载与本轮新增的结构查看工具 |



## 工程根目录

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [demo.py](../demo.py) | 235 | `sort_key`；`main` |
| [test.py](../test.py) | 900 | `remove_elements`；`Module`（`__init__`, `setup`, `forward`, `get_worker_id_from_batch`, `get_data_from_batch`…） |
| [train.py](../train.py) | 143 | `main` |
| [validation.py](../validation.py) | 107 | `main` |


## basicsr

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [test.py](../basicsr/test.py) | 58 | `main` |
| [train.py](../basicsr/train.py) | 251 | `parse_options`；`init_loggers`；`create_train_val_dataloader`；`main` |
| [version.py](../basicsr/version.py) | 5 | 无顶层类/函数；查看导入、常量或脚本主体 |


## basicsr/data

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../basicsr/data/__init__.py) | 126 | `create_dataset`；`create_dataloader`；`worker_init_fn` |
| [data_sampler.py](../basicsr/data/data_sampler.py) | 49 | `EnlargedSampler`（`__init__`, `__iter__`, `__len__`, `set_epoch`） |
| [data_util.py](../basicsr/data/data_util.py) | 331 | `read_img_seq`；`generate_frame_indices`；`paired_paths_from_lmdb`；`paired_paths_from_meta_info_file`；`paired_paths_from_folder`；`paths_from_folder`；`paths_from_lmdb`；`generate_gaussian_kernel`；`duf_downsample` |
| [ffhq_dataset.py](../basicsr/data/ffhq_dataset.py) | 65 | `FFHQDataset`（`__init__`, `__getitem__`, `__len__`） |
| [paired_image_dataset.py](../basicsr/data/paired_image_dataset.py) | 116 | `PairedImageDataset`（`__init__`, `__getitem__`, `__len__`） |
| [prefetch_dataloader.py](../basicsr/data/prefetch_dataloader.py) | 126 | `PrefetchGenerator`（`__init__`, `run`, `__next__`, `__iter__`）；`PrefetchDataLoader`（`__init__`, `__iter__`）；`CPUPrefetcher`（`__init__`, `next`, `reset`）；`CUDAPrefetcher`（`__init__`, `preload`, `next`, `reset`） |
| [reds_dataset.py](../basicsr/data/reds_dataset.py) | 237 | `REDSDataset`（`__init__`, `__getitem__`, `__len__`） |
| [single_image_dataset.py](../basicsr/data/single_image_dataset.py) | 67 | `SingleImageDataset`（`__init__`, `__getitem__`, `__len__`） |
| [transforms.py](../basicsr/data/transforms.py) | 172 | `mod_crop`；`paired_random_crop`；`augment`；`img_rotate` |
| [video_test_dataset.py](../basicsr/data/video_test_dataset.py) | 325 | `VideoTestDataset`（`__init__`, `__getitem__`, `__len__`）；`VideoTestVimeo90KDataset`（`__init__`, `__getitem__`, `__len__`）；`VideoTestDUFDataset`（`__getitem__`）；`VideoRecurrentTestDataset`（`__init__`, `__getitem__`, `__len__`） |
| [vimeo90k_dataset.py](../basicsr/data/vimeo90k_dataset.py) | 130 | `Vimeo90KDataset`（`__init__`, `__getitem__`, `__len__`） |


## basicsr/metrics

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../basicsr/metrics/__init__.py) | 4 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [fid.py](../basicsr/metrics/fid.py) | 102 | `load_patched_inception_v3`；`extract_inception_features`；`calculate_fid` |
| [metric_util.py](../basicsr/metrics/metric_util.py) | 47 | `reorder_image`；`to_y_channel` |
| [niqe.py](../basicsr/metrics/niqe.py) | 205 | `estimate_aggd_param`；`compute_feature`；`niqe`；`calculate_niqe` |
| [psnr_ssim.py](../basicsr/metrics/psnr_ssim.py) | 141 | `calculate_psnr`；`_ssim`；`calculate_ssim` |


## basicsr/models

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../basicsr/models/__init__.py) | 42 | `create_model` |
| [base_model.py](../basicsr/models/base_model.py) | 331 | `BaseModel`（`__init__`, `feed_data`, `optimize_parameters`, `get_current_visuals`, `save`…） |
| [edvr_model.py](../basicsr/models/edvr_model.py) | 71 | `EDVRModel`（`__init__`, `setup_optimizers`, `optimize_parameters`） |
| [esrgan_model.py](../basicsr/models/esrgan_model.py) | 89 | `ESRGANModel`（`optimize_parameters`） |
| [lr_scheduler.py](../basicsr/models/lr_scheduler.py) | 118 | `MultiStepRestartLR`（`__init__`, `get_lr`）；`get_position_from_periods`；`CosineAnnealingRestartLR`（`__init__`, `get_lr`） |
| [sr_model.py](../basicsr/models/sr_model.py) | 207 | `SRModel`（`__init__`, `init_training_settings`, `setup_optimizers`, `feed_data`, `optimize_parameters`…） |
| [srgan_model.py](../basicsr/models/srgan_model.py) | 142 | `SRGANModel`（`init_training_settings`, `setup_optimizers`, `optimize_parameters`, `save`） |
| [stylegan2_model.py](../basicsr/models/stylegan2_model.py) | 330 | `StyleGAN2Model`（`__init__`, `init_training_settings`, `setup_optimizers`, `model_ema`, `feed_data`…） |
| [video_base_model.py](../basicsr/models/video_base_model.py) | 172 | `VideoBaseModel`（`dist_validation`, `nondist_validation`, `_log_validation_metric_values`） |
| [video_gan_model.py](../basicsr/models/video_gan_model.py) | 15 | `VideoGANModel` |


## basicsr/models/archs

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../basicsr/models/archs/__init__.py) | 46 | `dynamic_instantiation`；`define_network` |
| [arch_util.py](../basicsr/models/archs/arch_util.py) | 302 | `default_init_weights`；`make_layer`；`ResidualBlockNoBN`（`__init__`, `forward`）；`Upsample`（`__init__`）；`flow_warp`；`resize_flow`；`pixel_unshuffle`；`DCNv2Pack`（`__init__`, `forward`） |
| [dfdnet_arch.py](../basicsr/models/archs/dfdnet_arch.py) | 186 | `SFTUpBlock`（`__init__`, `forward`）；`DFDNet`（`__init__`, `swap_feat`, `put_dict_to_device`, `forward`） |
| [dfdnet_util.py](../basicsr/models/archs/dfdnet_util.py) | 186 | `BlurFunctionBackward`（`forward`, `backward`）；`BlurFunction`（`forward`, `backward`）；`Blur`（`__init__`, `forward`）；`calc_mean_std`；`adaptive_instance_normalization`；`AttentionBlock`；`conv_block`；`MSDilationBlock`（`__init__`, `forward`）；`UpResBlock`（`__init__`, `forward`） |
| [discriminator_arch.py](../basicsr/models/archs/discriminator_arch.py) | 83 | `VGGStyleDiscriminator128`（`__init__`, `forward`） |
| [duf_arch.py](../basicsr/models/archs/duf_arch.py) | 359 | `DenseBlocksTemporalReduce`（`__init__`, `forward`）；`DenseBlocks`（`__init__`, `forward`）；`DynamicUpsamplingFilter`（`__init__`, `forward`）；`DUF`（`__init__`, `forward`） |
| [edsr_arch.py](../basicsr/models/archs/edsr_arch.py) | 65 | `EDSR`（`__init__`, `forward`） |
| [edvr_arch.py](../basicsr/models/archs/edvr_arch.py) | 420 | `PCDAlignment`（`__init__`, `forward`）；`TSAFusion`（`__init__`, `forward`）；`PredeblurModule`（`__init__`, `forward`）；`EDVR`（`__init__`, `forward`） |
| [inception.py](../basicsr/models/archs/inception.py) | 323 | `InceptionV3`（`__init__`, `forward`）；`fid_inception_v3`；`FIDInceptionA`（`__init__`, `forward`）；`FIDInceptionC`（`__init__`, `forward`）；`FIDInceptionE_1`（`__init__`, `forward`）；`FIDInceptionE_2`（`__init__`, `forward`） |
| [rcan_arch.py](../basicsr/models/archs/rcan_arch.py) | 141 | `ChannelAttention`（`__init__`, `forward`）；`RCAB`（`__init__`, `forward`）；`ResidualGroup`（`__init__`, `forward`）；`RCAN`（`__init__`, `forward`） |
| [rrdbnet_arch.py](../basicsr/models/archs/rrdbnet_arch.py) | 113 | `ResidualDenseBlock`（`__init__`, `forward`）；`RRDB`（`__init__`, `forward`）；`RRDBNet`（`__init__`, `forward`） |
| [spynet_arch.py](../basicsr/models/archs/spynet_arch.py) | 159 | `BasicModule`（`__init__`, `forward`）；`SpyNet`（`__init__`, `preprocess`, `process`, `forward`） |
| [srresnet_arch.py](../basicsr/models/archs/srresnet_arch.py) | 76 | `MSRResNet`（`__init__`, `forward`） |
| [stylegan2_arch.py](../basicsr/models/archs/stylegan2_arch.py) | 924 | `NormStyleCode`（`forward`）；`make_resample_kernel`；`UpFirDnUpsample`（`__init__`, `forward`, `__repr__`）；`UpFirDnDownsample`（`__init__`, `forward`, `__repr__`）；`UpFirDnSmooth`（`__init__`, `forward`, `__repr__`）；`EqualLinear`（`__init__`, `forward`, `__repr__`）；`ModulatedConv2d`（`__init__`, `forward`, `__repr__`）；`StyleConv`（`__init__`, `forward`）；`ToRGB`（`__init__`, `forward`）；其余定义见文件 |
| [tof_arch.py](../basicsr/models/archs/tof_arch.py) | 218 | `BasicModule`（`__init__`, `forward`）；`SPyNetTOF`（`__init__`, `forward`）；`TOFlow`（`__init__`, `normalize`, `denormalize`, `forward`） |
| [vgg_arch.py](../basicsr/models/archs/vgg_arch.py) | 171 | `insert_bn`；`VGGFeatureExtractor`（`__init__`, `forward`） |


## basicsr/models/losses

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../basicsr/models/losses/__init__.py) | 8 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [loss_util.py](../basicsr/models/losses/loss_util.py) | 95 | `reduce_loss`；`weight_reduce_loss`；`weighted_loss` |
| [losses.py](../basicsr/models/losses/losses.py) | 442 | `l1_loss`；`mse_loss`；`charbonnier_loss`；`L1Loss`（`__init__`, `forward`）；`MSELoss`（`__init__`, `forward`）；`CharbonnierLoss`（`__init__`, `forward`）；`WeightedTVLoss`（`__init__`, `forward`）；`PerceptualLoss`（`__init__`, `forward`, `_gram_mat`）；`GANLoss`（`__init__`, `_wgan_loss`, `_wgan_softplus_loss`, `get_target_label`, `forward`）；其余定义见文件 |


## basicsr/models/ops/dcn

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../basicsr/models/ops/dcn/__init__.py) | 8 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [deform_conv.py](../basicsr/models/ops/dcn/deform_conv.py) | 390 | `DeformConvFunction`（`forward`, `backward`, `_output_size`）；`ModulatedDeformConvFunction`（`forward`, `backward`, `_infer_shape`）；`DeformConv`（`__init__`, `reset_parameters`, `forward`）；`DeformConvPack`（`__init__`, `init_offset`, `forward`）；`ModulatedDeformConv`（`__init__`, `init_weights`, `forward`）；`ModulatedDeformConvPack`（`__init__`, `init_weights`, `forward`） |


## basicsr/models/ops/fused_act

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../basicsr/models/ops/fused_act/__init__.py) | 3 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [fused_act.py](../basicsr/models/ops/fused_act/fused_act.py) | 81 | `FusedLeakyReLUFunctionBackward`（`forward`, `backward`）；`FusedLeakyReLUFunction`（`forward`, `backward`）；`FusedLeakyReLU`（`__init__`, `forward`）；`fused_leaky_relu` |


## basicsr/models/ops/upfirdn2d

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../basicsr/models/ops/upfirdn2d/__init__.py) | 3 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [upfirdn2d.py](../basicsr/models/ops/upfirdn2d/upfirdn2d.py) | 189 | `UpFirDn2dBackward`（`forward`, `backward`）；`UpFirDn2d`（`forward`, `backward`）；`upfirdn2d`；`upfirdn2d_native` |


## basicsr/utils

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../basicsr/utils/__init__.py) | 31 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [dist_util.py](../basicsr/utils/dist_util.py) | 83 | `init_dist`；`_init_dist_pytorch`；`_init_dist_slurm`；`get_dist_info`；`master_only` |
| [download_util.py](../basicsr/utils/download_util.py) | 70 | `download_file_from_google_drive`；`get_confirm_token`；`save_response_content` |
| [face_util.py](../basicsr/utils/face_util.py) | 217 | `FaceRestorationHelper`（`__init__`, `init_dlib`, `free_dlib_gpu_memory`, `read_input_image`, `detect_faces`…） |
| [file_client.py](../basicsr/utils/file_client.py) | 183 | `BaseStorageBackend`（`get`, `get_text`）；`MemcachedBackend`（`__init__`, `get`, `get_text`）；`HardDiskBackend`（`get`, `get_text`）；`LmdbBackend`（`__init__`, `get`, `get_text`）；`FileClient`（`__init__`, `get`, `get_text`） |
| [flow_util.py](../basicsr/utils/flow_util.py) | 180 | `flowread`；`flowwrite`；`quantize_flow`；`dequantize_flow`；`quantize`；`dequantize` |
| [img_util.py](../basicsr/utils/img_util.py) | 165 | `img2tensor`；`tensor2img`；`imfrombytes`；`imwrite`；`crop_border` |
| [lmdb_util.py](../basicsr/utils/lmdb_util.py) | 208 | `make_lmdb_from_imgs`；`read_img_worker`；`LmdbMaker`（`__init__`, `put`, `close`） |
| [logger.py](../basicsr/utils/logger.py) | 177 | `MessageLogger`（`__init__`, `__call__`）；`init_tb_logger`；`init_wandb_logger`；`get_root_logger`；`get_env_info` |
| [matlab_functions.py](../basicsr/utils/matlab_functions.py) | 361 | `cubic`；`calculate_weights_indices`；`imresize`；`rgb2ycbcr`；`bgr2ycbcr`；`ycbcr2rgb`；`ycbcr2bgr`；`_convert_input_type_range`；`_convert_output_type_range` |
| [misc.py](../basicsr/utils/misc.py) | 139 | `set_random_seed`；`get_time_str`；`mkdir_and_rename`；`make_exp_dirs`；`scandir`；`check_resume`；`sizeof_fmt` |
| [options.py](../basicsr/utils/options.py) | 110 | `ordered_yaml`；`parse`；`dict2str` |


## callbacks

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [custom.py](../callbacks/custom.py) | 39 | `get_ckpt_callback`；`get_viz_callback` |
| [detection.py](../callbacks/detection.py) | 107 | `DetectionVizEnum`；`DetectionVizCallback`（`__init__`, `on_train_batch_end_custom`, `on_validation_batch_end_custom`, `on_validation_epoch_end_custom`） |
| [gradflow.py](../callbacks/gradflow.py) | 26 | `GradFlowLogCallback`（`__init__`, `on_before_zero_grad`） |
| [viz_base.py](../callbacks/viz_base.py) | 180 | `VizCallbackBase`（`__init__`, `_reset_buffer`, `add_to_buffer`, `get_from_buffer`, `on_train_batch_end_custom`…） |


## callbacks/utils

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [visualization.py](../callbacks/utils/visualization.py) | 23 | `get_grad_flow_figure` |


## config

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [modifier.py](../config/modifier.py) | 71 | `dynamically_modify_train_config`；`_get_modified_hw_multiple_of` |


## data/data_module

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [ev_img_data_moudle.py](../data/data_module/ev_img_data_moudle.py) | 199 | `get_dataloader_kwargs`；`DataModule`（`__init__`, `get_dataloading_hw`, `set_mixed_sampling_mode_variables_for_train`, `setup`, `train_dataloader`…） |


## data/ev_img_dataloader

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [collate.py](../data/ev_img_dataloader/collate.py) | 48 | `collate_object_labels`；`collate_sparsely_batched_object_labels`；`custom_collate`；`custom_collate_rnd`；`custom_collate_streaming` |
| [collate_from_pytorch.py](../data/ev_img_dataloader/collate_from_pytorch.py) | 149 | `collate`；`collate_numpy_array_fn`；`collate_numpy_scalar_fn`；`collate_float_fn`；`collate_int_fn`；`collate_str_fn` |
| [dataset_rnd.py](../data/ev_img_dataloader/dataset_rnd.py) | 168 | `SequenceDataset`（`__init__`, `only_load_labels`, `load_everything`, `__len__`, `__getitem__`）；`CustomConcatDataset`（`__init__`, `only_load_labels`, `load_everything`）；`build_random_access_dataset`；`get_weighted_random_sampler` |
| [dataset_streaming.py](../data/ev_img_dataloader/dataset_streaming.py) | 133 | `build_streaming_dataset`；`get_sequences`；`partialclass`；`build_streaming_train_dataset`；`build_streaming_evaluation_dataset` |
| [labels.py](../data/ev_img_dataloader/labels.py) | 468 | `ObjectLabelBase`（`__init__`, `clamp_to_frame_`, `remove_flat_labels_`, `create_empty`, `_assert_not_numpy`…）；`ObjectLabelFactory`（`__init__`, `from_structured_array`, `__len__`, `__getitem__`）；`ObjectLabels`（`__init__`, `__len__`, `rotate_`, `zoom_in_and_rescale_`, `zoom_out_and_rescale_`…）；`SparselyBatchedObjectLabels`（`__init__`, `__len__`, `__iter__`, `__getitem__`, `__add__`…） |
| [sequence_base.py](../data/ev_img_dataloader/sequence_base.py) | 151 | `get_event_representation_dir`；`get_objframe_idx_2_repr_idx`；`SequenceBase`（`__init__`, `_get_labels_from_repr_idx`, `_get_event_repr_torch`, `_get_img_torch`, `__len__`…） |
| [sequence_for_streaming.py](../data/ev_img_dataloader/sequence_for_streaming.py) | 302 | `_scalar_as_1d_array`；`_get_ev_repr_range_indices`；`SequenceForIter`（`__init__`, `get_sequences_with_guaranteed_labels`, `ev_padding_representation`, `img_padding_representation`, `get_fully_padded_sample`…）；`RandAugmentIterDataPipe`（`__init__`, `__iter__`） |
| [sequence_rnd.py](../data/ev_img_dataloader/sequence_rnd.py) | 143 | `SequenceForRandomAccess`（`__init__`, `__len__`, `__getitem__`, `getitem_with_guaranteed_labels`, `is_only_loading_labels`…） |


## data/source_dataset_process

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [from_aedat4_to_h5.py](../data/source_dataset_process/from_aedat4_to_h5.py) | 202 | `parse_argument`；`get_reader` |
| [from_json_to_npy.py](../data/source_dataset_process/from_json_to_npy.py) | 95 | `get_number_in_str` |


## data/source_dataset_process/davis_utils

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [buffer.py](../data/source_dataset_process/davis_utils/buffer.py) | 119 | `FrameBuffer`（`__init__`, `__len__`, `push`, `get_cur_frame`, `get_prev_frame`…）；`EventBuffer`（`__init__`, `__len__`, `push`, `getLowestTime`, `getHighestTime`…） |


## data/utils

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [augmentor.py](../data/utils/augmentor.py) | 449 | `ZoomOutState`；`RotationState`；`AugmentationState`；`RandomSpatialAugmentorGenX`（`__init__`, `randomize_augmentation`, `_zoom_out_and_rescale`, `_zoom_out_and_rescale_tensor`, `_zoom_out_and_rescale_recursive`…）；`get_most_recent_objframe`；`randomly_sample_zoom_window_from_objframe`；`randomly_sample_zoom_window_from_label_rectangle` |
| [datapipes.py](../data/utils/datapipes.py) | 54 | `MapToIter`（`__init__`, `__iter__`, `__len__`）；`Repeat`（`__init__`, `__iter__`, `__len__`）；`ZipperLongest`（`__init__`, `__iter__`, `__len__`） |
| [representations.py](../data/utils/representations.py) | 218 | `RepresentationBase`（`construct`, `get_shape`, `get_numpy_dtype`, `get_torch_dtype`, `dtype`…）；`StackedHistogram`（`__init__`, `get_numpy_dtype`, `get_torch_dtype`, `merge_channel_and_bins`, `get_shape`…）；`cumsum_channel`；`MixedDensityEventStack`（`__init__`, `get_numpy_dtype`, `get_torch_dtype`, `get_shape`, `construct`） |
| [spatial.py](../data/utils/spatial.py) | 26 | `get_original_hw`；`get_dataloading_hw` |
| [stream_concat_datapipe.py](../data/utils/stream_concat_datapipe.py) | 103 | `DummyIterDataPipe`（`__init__`, `__iter__`）；`ConcatStreamingDataPipe`（`__init__`, `random_torch_shuffle_list`, `_get_zipped_streams`, `_print_seed_debug_info`, `_get_zipped_streams_with_worker_id`…） |
| [stream_sharded_datapipe.py](../data/utils/stream_sharded_datapipe.py) | 96 | `ShardedStreamingDataPipe`（`__init__`, `yield_pyramid_indices`, `assign_datapipes_to_worker`, `get_zipped_stream_from_worker_datapipes`, `__iter__`） |
| [types.py](../data/utils/types.py) | 56 | `DataType`；`DatasetType`；`DatasetMode`；`DatasetSamplingMode`；`ObjDetOutput` |


## frame_construction

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [main.py](../frame_construction/main.py) | 897 | `DataKeys`；`SplitType`；`NoLabelsException`；`H5Writer`（`__init__`, `__enter__`, `__exit__`, `close_callback`, `close`…）；`H5Reader`（`__init__`, `__enter__`, `__exit__`, `_close_callback`, `close`…）；`prophesee_bbox_filter`；`conservative_bbox_filter`；`remove_faulty_huge_bbox_filter`；`crop_to_fov_filter`；其余定义见文件 |
| [main_dsec.py](../frame_construction/main_dsec.py) | 878 | `DataKeys`；`SplitType`；`NoLabelsException`；`H5Writer`（`__init__`, `__enter__`, `__exit__`, `close_callback`, `close`…）；`H5Reader`（`__init__`, `__enter__`, `__exit__`, `_close_callback`, `close`…）；`prophesee_bbox_filter`；`conservative_bbox_filter`；`remove_faulty_huge_bbox_filter`；`crop_to_fov_filter`；其余定义见文件 |


## frame_construction/utils

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [preprocessing.py](../frame_construction/utils/preprocessing.py) | 12 | `_blosc_opts` |
| [representations.py](../frame_construction/utils/representations.py) | 218 | `RepresentationBase`（`construct`, `get_shape`, `get_numpy_dtype`, `get_torch_dtype`, `dtype`…）；`StackedHistogram`（`__init__`, `get_numpy_dtype`, `get_torch_dtype`, `merge_channel_and_bins`, `get_shape`…）；`cumsum_channel`；`MixedDensityEventStack`（`__init__`, `get_numpy_dtype`, `get_torch_dtype`, `get_shape`, `construct`） |


## loggers

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [utils.py](../loggers/utils.py) | 52 | `get_wandb_logger`；`get_ckpt_path` |
| [wandb_logger.py](../loggers/wandb_logger.py) | 379 | `WandbLogger`（`__init__`, `get_checkpoint`, `__getstate__`, `experiment`, `watch`…） |


## models/detection

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init_.py](../models/detection/__init_.py) | 0 | 无顶层类/函数；查看导入、常量或脚本主体 |


## models/detection/recurrent_backbone

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../models/detection/recurrent_backbone/__init__.py) | 25 | `build_recurrent_backbone` |
| [base.py](../models/detection/recurrent_backbone/base.py) | 11 | `BaseDetector`（`get_stage_dims`, `get_strides`） |
| [darknet_rnn_forward_fusion.py](../models/detection/recurrent_backbone/darknet_rnn_forward_fusion.py) | 526 | `RNNDetector`（`__init__`, `get_stage_dims`, `get_strides`, `forward`）；`RNNFusionDetectorStage`（`__init__`, `make_group_layer`, `forward_ssm`, `forward_rnn`, `forward`）；`RNNDetectorStage`（`__init__`, `make_group_layer`, `make_spp_block`, `forward_ssm`, `forward_rnn`…） |
| [darknet_rnn_overall_fusion.py](../models/detection/recurrent_backbone/darknet_rnn_overall_fusion.py) | 293 | `RNNDetector`（`__init__`, `get_stage_dims`, `get_strides`, `forward`）；`MaxVitAttentionPairCl`（`__init__`, `forward`）；`RNNFusionDetectorStage`（`__init__`, `make_group_layer`, `make_spp_block`, `forward`） |
| [maxvit_rnn_forward_fusion.py](../models/detection/recurrent_backbone/maxvit_rnn_forward_fusion.py) | 523 | `RNNDetector`（`__init__`, `get_stage_dims`, `get_strides`, `forward`）；`MaxVitAttentionPairCl`（`__init__`, `forward`）；`RNNFusionDetectorStage`（`__init__`, `forward_ssm`, `forward_rnn`, `forward`）；`RNNDetectorStage`（`__init__`, `forward_ssm`, `forward_rnn`, `forward`） |
| [maxvit_rnn_single_modal.py](../models/detection/recurrent_backbone/maxvit_rnn_single_modal.py) | 248 | `RNNDetector`（`__init__`, `get_stage_dims`, `get_strides`, `forward`）；`MaxVitAttentionPairCl`（`__init__`, `forward`）；`RNNDetectorStage`（`__init__`, `forward_ssm`, `forward_rnn`, `forward`） |
| [resnet_rnn_forward_fusion.py](../models/detection/recurrent_backbone/resnet_rnn_forward_fusion.py) | 529 | `RNNDetector`（`__init__`, `get_stage_dims`, `get_strides`, `forward`）；`RNNFusionDetectorStage`（`__init__`, `forward_ssm`, `forward_rnn`, `forward`）；`RNNDetectorStage`（`__init__`, `forward_ssm`, `forward_rnn`, `forward`） |
| [swin_rnn_forward_fusion.py](../models/detection/recurrent_backbone/swin_rnn_forward_fusion.py) | 546 | `RNNDetector`（`__init__`, `get_stage_dims`, `get_strides`, `forward`）；`SwinAttentionPairCl`（`__init__`, `forward`）；`RNNFusionDetectorStage`（`__init__`, `forward_ssm`, `forward_rnn`, `forward`）；`RNNDetectorStage`（`__init__`, `forward_ssm`, `forward_rnn`, `forward`） |


## models/detection/yolox/models

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../models/detection/yolox/models/__init__.py) | 0 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [losses.py](../models/detection/yolox/models/losses.py) | 55 | `IOUloss`（`__init__`, `forward`） |
| [network_blocks.py](../models/detection/yolox/models/network_blocks.py) | 142 | `SiLU`（`forward`）；`get_activation`；`BaseConv`（`__init__`, `forward`, `fuseforward`）；`DWConv`（`__init__`, `forward`）；`Bottleneck`（`__init__`, `forward`）；`CSPLayer`（`__init__`, `forward`） |
| [yolo_head.py](../models/detection/yolox/models/yolo_head.py) | 614 | `YOLOXHead`（`__init__`, `initialize_biases`, `forward`, `get_output_and_grid`, `decode_outputs`…） |


## models/detection/yolox/utils

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../models/detection/yolox/utils/__init__.py) | 6 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [boxes.py](../models/detection/yolox/utils/boxes.py) | 135 | `filter_box`；`postprocess`；`bboxes_iou`；`matrix_iou`；`adjust_box_anns`；`xyxy2xywh`；`xyxy2cxcywh` |
| [compat.py](../models/detection/yolox/utils/compat.py) | 15 | `meshgrid` |


## models/detection/yolox_extension/models

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../models/detection/yolox_extension/models/__init__.py) | 0 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [build.py](../models/detection/yolox_extension/models/build.py) | 28 | `build_yolox_head`；`build_yolox_fpn` |
| [detector.py](../models/detection/yolox_extension/models/detector.py) | 70 | `YoloXDetector`（`__init__`, `forward_backbone_rnn`, `forward_backbone_ssm`, `forward_detect`） |
| [detector_fusion.py](../models/detection/yolox_extension/models/detector_fusion.py) | 92 | `YoloXDetector`（`__init__`, `forward_backbone_rnn`, `forward_backbone_ssm`, `forward_detect`, `forward`） |
| [yolo_pafpn.py](../models/detection/yolox_extension/models/yolo_pafpn.py) | 139 | `YOLOPAFPN`（`__init__`, `forward`） |


## models/layers

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [rnn.py](../models/layers/rnn.py) | 133 | `DWSConvLSTM2d`（`__init__`, `forward`） |


## models/layers/align_and_fusion

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [EgF.py](../models/layers/align_and_fusion/EgF.py) | 89 | `last_zero_init`；`EgF`（`__init__`, `reset_parameters`, `spatial_pool`, `forward`）；`ALIGN`（`__init__`, `forward`） |
| [EguideF.py](../models/layers/align_and_fusion/EguideF.py) | 69 | `last_zero_init`；`EgF`（`__init__`, `reset_parameters`, `spatial_pool`, `forward`） |
| [adain.py](../models/layers/align_and_fusion/adain.py) | 67 | `calc_mean_std`；`adaptive_instance_normalization`；`_calc_feat_flatten_mean_std`；`_mat_sqrt`；`coral` |
| [align.py](../models/layers/align_and_fusion/align.py) | 51 | `Feature_wrapper`（`__init__`, `forward`）；`Blur_aug`（`__init__`, `forward`） |
| [cbam.py](../models/layers/align_and_fusion/cbam.py) | 84 | `BasicConv`（`__init__`, `forward`）；`Flatten`（`forward`）；`ChannelGate`（`__init__`, `forward`）；`logsumexp_2d`；`ChannelPool`（`forward`）；`SpatialGate`（`__init__`, `forward`） |
| [cross_mamba.py](../models/layers/align_and_fusion/cross_mamba.py) | 216 | `CROSS_Mamba_Fusion`（`__init__`, `dt_init`, `A_log_init`, `D_init`, `forward_corev0`…） |
| [fusion.py](../models/layers/align_and_fusion/fusion.py) | 218 | `Conv_BN_ReLU`（`__init__`, `forward`）；`Cat_Fusion`（`__init__`, `forward`）；`Cross_mamba`（`__init__`, `forward`）；`Cross_cbam`（`__init__`, `forward`）；`Seletive_Feature_fusion`（`__init__`, `forward`）；`Cross_wsam`（`__init__`, `forward`） |
| [fusion_dynamic.py](../models/layers/align_and_fusion/fusion_dynamic.py) | 98 | `ChannelPool`（`forward`）；`BasicConv`（`__init__`, `forward`）；`DCM`（`__init__`, `forward`）；`Fusion_dynamic`（`__init__`, `forward`） |


## models/layers/compat

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [deform_conv.py](../models/layers/compat/deform_conv.py) | 44 | `DeformConv2d`（`__init__`, `reset_parameters`, `forward`） |


## models/layers/darknet

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [backbone.py](../models/layers/darknet/backbone.py) | 684 | `Mlp`（`__init__`, `forward`）；`window_partition`；`window_reverse`；`get_window_size`；`WindowAttention3D`（`__init__`, `forward`）；`SwinTransformerBlock3D`（`__init__`, `forward_part1`, `forward_part2`, `forward`）；`PatchMerging`（`__init__`, `forward`）；`compute_mask`；`BasicLayer`（`__init__`, `forward`）；其余定义见文件 |
| [corr_extract.py](../models/layers/darknet/corr_extract.py) | 163 | `corr3D`（`__init__`, `forward`）；`window_partition`；`window_reverse`；`corrBlock3D`（`__init__`, `forward`）；`corr_BasicLayer`（`__init__`, `forward`） |
| [modules.py](../models/layers/darknet/modules.py) | 453 | `BaseConv`（`__init__`, `forward`, `fuseforward`）；`ResLayer`（`__init__`, `forward`）；`_trunc_normal_`；`trunc_normal_`；`get_activation`；`Focus_yolo`（`__init__`, `forward`）；`Focus`（`__init__`, `patch_and_conv`, `forward`）；`Temporal_Active_Focus_connect`（`__init__`, `init_weights`, `patch`, `forward`）；`Temporal_Active_Focus_corr`（`__init__`, `init_weights`, `forward`）；其余定义见文件 |


## models/layers/maxvit

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../models/layers/maxvit/__init__.py) | 0 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [maxvit.py](../models/layers/maxvit/maxvit.py) | 395 | `PartitionType`；`nChw_2_nhwC`；`nhwC_2_nChw`；`nChw_2_nhwC`；`LayerScale`（`__init__`, `forward`）；`GLU`（`__init__`, `forward`）；`MLP`（`__init__`, `forward`）；`DownsampleBase`（`__init__`, `output_is_normed`）；`get_downsample_layer_Cf2Cl`；其余定义见文件 |


## models/layers/maxvit/layers

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../models/layers/maxvit/layers/__init__.py) | 44 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [activations.py](../models/layers/maxvit/layers/activations.py) | 145 | `swish`；`Swish`（`__init__`, `forward`）；`mish`；`Mish`（`__init__`, `forward`）；`sigmoid`；`Sigmoid`（`__init__`, `forward`）；`tanh`；`Tanh`（`__init__`, `forward`）；`hard_swish`；其余定义见文件 |
| [activations_jit.py](../models/layers/maxvit/layers/activations_jit.py) | 90 | `swish_jit`；`mish_jit`；`SwishJit`（`__init__`, `forward`）；`MishJit`（`__init__`, `forward`）；`hard_sigmoid_jit`；`HardSigmoidJit`（`__init__`, `forward`）；`hard_swish_jit`；`HardSwishJit`（`__init__`, `forward`）；`hard_mish_jit`；其余定义见文件 |
| [activations_me.py](../models/layers/maxvit/layers/activations_me.py) | 218 | `swish_jit_fwd`；`swish_jit_bwd`；`SwishJitAutoFn`（`symbolic`, `forward`, `backward`）；`swish_me`；`SwishMe`（`__init__`, `forward`）；`mish_jit_fwd`；`mish_jit_bwd`；`MishJitAutoFn`（`forward`, `backward`）；`mish_me`；其余定义见文件 |
| [adaptive_avgmax_pool.py](../models/layers/maxvit/layers/adaptive_avgmax_pool.py) | 118 | `adaptive_pool_feat_mult`；`adaptive_avgmax_pool2d`；`adaptive_catavgmax_pool2d`；`select_adaptive_pool2d`；`FastAdaptiveAvgPool2d`（`__init__`, `forward`）；`AdaptiveAvgMaxPool2d`（`__init__`, `forward`）；`AdaptiveCatAvgMaxPool2d`（`__init__`, `forward`）；`SelectAdaptivePool2d`（`__init__`, `is_identity`, `forward`, `feat_mult`, `__repr__`） |
| [attention_pool2d.py](../models/layers/maxvit/layers/attention_pool2d.py) | 131 | `RotAttentionPool2d`（`__init__`, `forward`）；`AttentionPool2d`（`__init__`, `forward`） |
| [blur_pool.py](../models/layers/maxvit/layers/blur_pool.py) | 42 | `BlurPool2d`（`__init__`, `forward`） |
| [bottleneck_attn.py](../models/layers/maxvit/layers/bottleneck_attn.py) | 157 | `rel_logits_1d`；`PosEmbedRel`（`__init__`, `forward`）；`BottleneckAttn`（`__init__`, `reset_parameters`, `forward`） |
| [cbam.py](../models/layers/maxvit/layers/cbam.py) | 112 | `ChannelAttn`（`__init__`, `forward`）；`LightChannelAttn`（`__init__`, `forward`）；`SpatialAttn`（`__init__`, `forward`）；`LightSpatialAttn`（`__init__`, `forward`）；`CbamModule`（`__init__`, `forward`）；`LightCbamModule`（`__init__`, `forward`） |
| [classifier.py](../models/layers/maxvit/layers/classifier.py) | 56 | `_create_pool`；`_create_fc`；`create_classifier`；`ClassifierHead`（`__init__`, `forward`） |
| [cond_conv2d.py](../models/layers/maxvit/layers/cond_conv2d.py) | 123 | `get_condconv_initializer`；`CondConv2d`（`__init__`, `reset_parameters`, `forward`） |
| [config.py](../models/layers/maxvit/layers/config.py) | 115 | `is_no_jit`；`set_no_jit`（`__init__`, `__enter__`, `__exit__`）；`is_exportable`；`set_exportable`（`__init__`, `__enter__`, `__exit__`）；`is_scriptable`；`set_scriptable`（`__init__`, `__enter__`, `__exit__`）；`set_layer_config`（`__init__`, `__enter__`, `__exit__`） |
| [conv2d_same.py](../models/layers/maxvit/layers/conv2d_same.py) | 42 | `conv2d_same`；`Conv2dSame`（`__init__`, `forward`）；`create_conv2d_pad` |
| [conv_bn_act.py](../models/layers/maxvit/layers/conv_bn_act.py) | 88 | `ConvNormAct`（`__init__`, `in_channels`, `out_channels`, `forward`）；`create_aa`；`ConvNormActAa`（`__init__`, `in_channels`, `out_channels`, `forward`） |
| [create_act.py](../models/layers/maxvit/layers/create_act.py) | 154 | `get_act_fn`；`get_act_layer`；`create_act_layer` |
| [create_attn.py](../models/layers/maxvit/layers/create_attn.py) | 89 | `get_attn`；`create_attn` |
| [create_conv2d.py](../models/layers/maxvit/layers/create_conv2d.py) | 36 | `create_conv2d` |
| [create_norm.py](../models/layers/maxvit/layers/create_norm.py) | 56 | `create_norm_layer`；`get_norm_layer` |
| [create_norm_act.py](../models/layers/maxvit/layers/create_norm_act.py) | 91 | `create_norm_act_layer`；`get_norm_act_layer` |
| [drop.py](../models/layers/maxvit/layers/drop.py) | 169 | `drop_block_2d`；`drop_block_fast_2d`；`DropBlock2d`（`__init__`, `forward`）；`drop_path`；`DropPath`（`__init__`, `forward`, `extra_repr`） |
| [eca.py](../models/layers/maxvit/layers/eca.py) | 145 | `EcaModule`（`__init__`, `forward`）；`CecaModule`（`__init__`, `forward`） |
| [evo_norm.py](../models/layers/maxvit/layers/evo_norm.py) | 352 | `instance_std`；`instance_std_tpu`；`instance_rms`；`manual_var`；`group_std`；`group_std_tpu`；`group_rms`；`EvoNorm2dB0`（`__init__`, `reset_parameters`, `forward`）；`EvoNorm2dB1`（`__init__`, `reset_parameters`, `forward`）；其余定义见文件 |
| [fast_norm.py](../models/layers/maxvit/layers/fast_norm.py) | 78 | `is_fast_norm`；`set_fast_norm`；`fast_group_norm`；`fast_layer_norm` |
| [filter_response_norm.py](../models/layers/maxvit/layers/filter_response_norm.py) | 68 | `inv_instance_rms`；`FilterResponseNormTlu2d`（`__init__`, `reset_parameters`, `forward`）；`FilterResponseNormAct2d`（`__init__`, `reset_parameters`, `forward`） |
| [gather_excite.py](../models/layers/maxvit/layers/gather_excite.py) | 90 | `GatherExcite`（`__init__`, `forward`） |
| [global_context.py](../models/layers/maxvit/layers/global_context.py) | 67 | `GlobalContext`（`__init__`, `reset_parameters`, `forward`） |
| [halo_attn.py](../models/layers/maxvit/layers/halo_attn.py) | 233 | `rel_logits_1d`；`PosEmbedRel`（`__init__`, `forward`）；`HaloAttn`（`__init__`, `reset_parameters`, `forward`） |
| [helpers.py](../models/layers/maxvit/layers/helpers.py) | 43 | `_ntuple`；`make_divisible`；`extend_tuple` |
| [inplace_abn.py](../models/layers/maxvit/layers/inplace_abn.py) | 87 | `InplaceAbn`（`__init__`, `reset_parameters`, `forward`） |
| [lambda_layer.py](../models/layers/maxvit/layers/lambda_layer.py) | 133 | `rel_pos_indices`；`LambdaLayer`（`__init__`, `reset_parameters`, `forward`） |
| [linear.py](../models/layers/maxvit/layers/linear.py) | 19 | `Linear`（`forward`） |
| [median_pool.py](../models/layers/maxvit/layers/median_pool.py) | 49 | `MedianPool2d`（`__init__`, `_padding`, `forward`） |
| [mixed_conv2d.py](../models/layers/maxvit/layers/mixed_conv2d.py) | 51 | `_split_channels`；`MixedConv2d`（`__init__`, `forward`） |
| [ml_decoder.py](../models/layers/maxvit/layers/ml_decoder.py) | 156 | `add_ml_decoder_head`；`TransformerDecoderLayerOptimal`（`__init__`, `__setstate__`, `forward`）；`GroupFC`（`__init__`, `__call__`）；`MLDecoder`（`__init__`, `forward`） |
| [mlp.py](../models/layers/maxvit/layers/mlp.py) | 126 | `Mlp`（`__init__`, `forward`）；`GluMlp`（`__init__`, `init_weights`, `forward`）；`GatedMlp`（`__init__`, `forward`）；`ConvMlp`（`__init__`, `forward`） |
| [non_local_attn.py](../models/layers/maxvit/layers/non_local_attn.py) | 145 | `NonLocalAttn`（`__init__`, `forward`, `reset_parameters`）；`BilinearAttnTransform`（`__init__`, `resize_mat`, `forward`）；`BatNonLocalAttn`（`__init__`, `forward`） |
| [norm.py](../models/layers/maxvit/layers/norm.py) | 117 | `GroupNorm`（`__init__`, `forward`）；`GroupNorm1`（`__init__`, `forward`）；`LayerNorm`（`__init__`, `forward`）；`LayerNorm2d`（`__init__`, `forward`）；`_is_contiguous`；`_layer_norm_cf`；`_layer_norm_cf_sqm`；`LayerNormExp2d`（`__init__`, `forward`） |
| [norm_act.py](../models/layers/maxvit/layers/norm_act.py) | 252 | `BatchNormAct2d`（`__init__`, `forward`）；`SyncBatchNormAct`（`forward`）；`convert_sync_batchnorm`；`_num_groups`；`GroupNormAct`（`__init__`, `forward`）；`LayerNormAct`（`__init__`, `forward`）；`LayerNormAct2d`（`__init__`, `forward`） |
| [padding.py](../models/layers/maxvit/layers/padding.py) | 56 | `get_padding`；`get_same_padding`；`is_static_pad`；`pad_same`；`get_padding_value` |
| [patch_embed.py](../models/layers/maxvit/layers/patch_embed.py) | 39 | `PatchEmbed`（`__init__`, `forward`） |
| [pool2d_same.py](../models/layers/maxvit/layers/pool2d_same.py) | 73 | `avg_pool2d_same`；`AvgPool2dSame`（`__init__`, `forward`）；`max_pool2d_same`；`MaxPool2dSame`（`__init__`, `forward`）；`create_pool2d` |
| [pos_embed.py](../models/layers/maxvit/layers/pos_embed.py) | 207 | `pixel_freq_bands`；`inv_freq_bands`；`build_sincos2d_pos_embed`；`build_fourier_pos_embed`；`FourierEmbed`（`__init__`, `forward`）；`rot`；`apply_rot_embed`；`apply_rot_embed_list`；`apply_rot_embed_split`；其余定义见文件 |
| [selective_kernel.py](../models/layers/maxvit/layers/selective_kernel.py) | 119 | `_kernel_valid`；`SelectiveKernelAttn`（`__init__`, `forward`）；`SelectiveKernel`（`__init__`, `forward`） |
| [separable_conv.py](../models/layers/maxvit/layers/separable_conv.py) | 76 | `SeparableConvNormAct`（`__init__`, `in_channels`, `out_channels`, `forward`）；`SeparableConv2d`（`__init__`, `in_channels`, `out_channels`, `forward`） |
| [space_to_depth.py](../models/layers/maxvit/layers/space_to_depth.py) | 53 | `SpaceToDepth`（`__init__`, `forward`）；`SpaceToDepthJit`（`__call__`）；`SpaceToDepthModule`（`__init__`, `forward`）；`DepthToSpace`（`__init__`, `forward`） |
| [split_attn.py](../models/layers/maxvit/layers/split_attn.py) | 84 | `RadixSoftmax`（`__init__`, `forward`）；`SplitAttn`（`__init__`, `forward`） |
| [split_batchnorm.py](../models/layers/maxvit/layers/split_batchnorm.py) | 75 | `SplitBatchNorm2d`（`__init__`, `forward`）；`convert_splitbn_model` |
| [squeeze_excite.py](../models/layers/maxvit/layers/squeeze_excite.py) | 74 | `SEModule`（`__init__`, `forward`）；`EffectiveSEModule`（`__init__`, `forward`） |
| [std_conv.py](../models/layers/maxvit/layers/std_conv.py) | 133 | `StdConv2d`（`__init__`, `forward`）；`StdConv2dSame`（`__init__`, `forward`）；`ScaledStdConv2d`（`__init__`, `forward`）；`ScaledStdConv2dSame`（`__init__`, `forward`） |
| [test_time_pool.py](../models/layers/maxvit/layers/test_time_pool.py) | 52 | `TestTimePoolHead`（`__init__`, `forward`）；`apply_test_time_pool` |
| [trace_utils.py](../models/layers/maxvit/layers/trace_utils.py) | 13 | `_float_to_int` |
| [weight_init.py](../models/layers/maxvit/layers/weight_init.py) | 125 | `_trunc_normal_`；`trunc_normal_`；`trunc_normal_tf_`；`variance_scaling_`；`lecun_normal_` |


## models/layers/s5

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../models/layers/s5/__init__.py) | 1 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [jax_func.py](../models/layers/s5/jax_func.py) | 462 | `safe_map`；`safe_map`；`safe_map`；`safe_map`；`safe_map`；`combine`；`_scan`；`associative_scan`；`test_associative_scan`；其余定义见文件 |
| [s5_init.py](../models/layers/s5/s5_init.py) | 280 | `make_HiPPO`；`make_NPLR_HiPPO`；`make_DPLR_HiPPO`；`make_Normal_S`；`make_Normal_HiPPO`；`log_step_initializer`；`init_log_steps`；`init_VinvB`；`trunc_standard_normal`；其余定义见文件 |
| [s5_model.py](../models/layers/s5/s5_model.py) | 509 | `binary_operator`；`apply_ssm`；`apply_ssm_liquid`；`discretize_bilinear`；`discretize_zoh`；`as_complex`；`S5SSM`（`__init__`, `initial_state`, `get_BC_tilde`, `forward_rnn`, `forward`）；`S5`（`__init__`, `initial_state`, `forward`）；`GEGLU`（`forward`）；其余定义见文件 |
| [s5_model______.py](../models/layers/s5/s5_model______.py) | 551 | `binary_operator`；`apply_ssm`；`apply_ssm_liquid`；`discretize_bilinear`；`as_complex`；`S5SSM`（`__init__`, `initial_state`, `get_BC_tilde`, `forward`）；`S5`（`__init__`, `initial_state`, `forward`）；`GEGLU`（`forward`）；`S5Block`（`__init__`, `forward`） |
| [triton_comparison.py](../models/layers/s5/triton_comparison.py) | 175 | `to_triton`；`to_numpy` |


## models/layers/swins

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [swins.py](../models/layers/swins/swins.py) | 353 | `window_partition`；`window_reverse`；`WindowAttention`（`__init__`, `forward`）；`GLU`（`__init__`, `forward`）；`MLP`（`__init__`, `forward`）；`SwinTransformerBlock`（`__init__`, `_calc_window_shift`, `_attn`, `forward`） |


## models/layers/swins/layers

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../models/layers/swins/layers/__init__.py) | 44 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [activations.py](../models/layers/swins/layers/activations.py) | 145 | `swish`；`Swish`（`__init__`, `forward`）；`mish`；`Mish`（`__init__`, `forward`）；`sigmoid`；`Sigmoid`（`__init__`, `forward`）；`tanh`；`Tanh`（`__init__`, `forward`）；`hard_swish`；其余定义见文件 |
| [activations_jit.py](../models/layers/swins/layers/activations_jit.py) | 90 | `swish_jit`；`mish_jit`；`SwishJit`（`__init__`, `forward`）；`MishJit`（`__init__`, `forward`）；`hard_sigmoid_jit`；`HardSigmoidJit`（`__init__`, `forward`）；`hard_swish_jit`；`HardSwishJit`（`__init__`, `forward`）；`hard_mish_jit`；其余定义见文件 |
| [activations_me.py](../models/layers/swins/layers/activations_me.py) | 218 | `swish_jit_fwd`；`swish_jit_bwd`；`SwishJitAutoFn`（`symbolic`, `forward`, `backward`）；`swish_me`；`SwishMe`（`__init__`, `forward`）；`mish_jit_fwd`；`mish_jit_bwd`；`MishJitAutoFn`（`forward`, `backward`）；`mish_me`；其余定义见文件 |
| [adaptive_avgmax_pool.py](../models/layers/swins/layers/adaptive_avgmax_pool.py) | 118 | `adaptive_pool_feat_mult`；`adaptive_avgmax_pool2d`；`adaptive_catavgmax_pool2d`；`select_adaptive_pool2d`；`FastAdaptiveAvgPool2d`（`__init__`, `forward`）；`AdaptiveAvgMaxPool2d`（`__init__`, `forward`）；`AdaptiveCatAvgMaxPool2d`（`__init__`, `forward`）；`SelectAdaptivePool2d`（`__init__`, `is_identity`, `forward`, `feat_mult`, `__repr__`） |
| [attention_pool2d.py](../models/layers/swins/layers/attention_pool2d.py) | 131 | `RotAttentionPool2d`（`__init__`, `forward`）；`AttentionPool2d`（`__init__`, `forward`） |
| [blur_pool.py](../models/layers/swins/layers/blur_pool.py) | 42 | `BlurPool2d`（`__init__`, `forward`） |
| [bottleneck_attn.py](../models/layers/swins/layers/bottleneck_attn.py) | 157 | `rel_logits_1d`；`PosEmbedRel`（`__init__`, `forward`）；`BottleneckAttn`（`__init__`, `reset_parameters`, `forward`） |
| [cbam.py](../models/layers/swins/layers/cbam.py) | 112 | `ChannelAttn`（`__init__`, `forward`）；`LightChannelAttn`（`__init__`, `forward`）；`SpatialAttn`（`__init__`, `forward`）；`LightSpatialAttn`（`__init__`, `forward`）；`CbamModule`（`__init__`, `forward`）；`LightCbamModule`（`__init__`, `forward`） |
| [classifier.py](../models/layers/swins/layers/classifier.py) | 56 | `_create_pool`；`_create_fc`；`create_classifier`；`ClassifierHead`（`__init__`, `forward`） |
| [cond_conv2d.py](../models/layers/swins/layers/cond_conv2d.py) | 123 | `get_condconv_initializer`；`CondConv2d`（`__init__`, `reset_parameters`, `forward`） |
| [config.py](../models/layers/swins/layers/config.py) | 115 | `is_no_jit`；`set_no_jit`（`__init__`, `__enter__`, `__exit__`）；`is_exportable`；`set_exportable`（`__init__`, `__enter__`, `__exit__`）；`is_scriptable`；`set_scriptable`（`__init__`, `__enter__`, `__exit__`）；`set_layer_config`（`__init__`, `__enter__`, `__exit__`） |
| [conv2d_same.py](../models/layers/swins/layers/conv2d_same.py) | 42 | `conv2d_same`；`Conv2dSame`（`__init__`, `forward`）；`create_conv2d_pad` |
| [conv_bn_act.py](../models/layers/swins/layers/conv_bn_act.py) | 88 | `ConvNormAct`（`__init__`, `in_channels`, `out_channels`, `forward`）；`create_aa`；`ConvNormActAa`（`__init__`, `in_channels`, `out_channels`, `forward`） |
| [create_act.py](../models/layers/swins/layers/create_act.py) | 154 | `get_act_fn`；`get_act_layer`；`create_act_layer` |
| [create_attn.py](../models/layers/swins/layers/create_attn.py) | 89 | `get_attn`；`create_attn` |
| [create_conv2d.py](../models/layers/swins/layers/create_conv2d.py) | 36 | `create_conv2d` |
| [create_norm.py](../models/layers/swins/layers/create_norm.py) | 56 | `create_norm_layer`；`get_norm_layer` |
| [create_norm_act.py](../models/layers/swins/layers/create_norm_act.py) | 91 | `create_norm_act_layer`；`get_norm_act_layer` |
| [drop.py](../models/layers/swins/layers/drop.py) | 169 | `drop_block_2d`；`drop_block_fast_2d`；`DropBlock2d`（`__init__`, `forward`）；`drop_path`；`DropPath`（`__init__`, `forward`, `extra_repr`） |
| [eca.py](../models/layers/swins/layers/eca.py) | 145 | `EcaModule`（`__init__`, `forward`）；`CecaModule`（`__init__`, `forward`） |
| [evo_norm.py](../models/layers/swins/layers/evo_norm.py) | 352 | `instance_std`；`instance_std_tpu`；`instance_rms`；`manual_var`；`group_std`；`group_std_tpu`；`group_rms`；`EvoNorm2dB0`（`__init__`, `reset_parameters`, `forward`）；`EvoNorm2dB1`（`__init__`, `reset_parameters`, `forward`）；其余定义见文件 |
| [fast_norm.py](../models/layers/swins/layers/fast_norm.py) | 78 | `is_fast_norm`；`set_fast_norm`；`fast_group_norm`；`fast_layer_norm` |
| [filter_response_norm.py](../models/layers/swins/layers/filter_response_norm.py) | 68 | `inv_instance_rms`；`FilterResponseNormTlu2d`（`__init__`, `reset_parameters`, `forward`）；`FilterResponseNormAct2d`（`__init__`, `reset_parameters`, `forward`） |
| [gather_excite.py](../models/layers/swins/layers/gather_excite.py) | 90 | `GatherExcite`（`__init__`, `forward`） |
| [global_context.py](../models/layers/swins/layers/global_context.py) | 67 | `GlobalContext`（`__init__`, `reset_parameters`, `forward`） |
| [halo_attn.py](../models/layers/swins/layers/halo_attn.py) | 233 | `rel_logits_1d`；`PosEmbedRel`（`__init__`, `forward`）；`HaloAttn`（`__init__`, `reset_parameters`, `forward`） |
| [helpers.py](../models/layers/swins/layers/helpers.py) | 43 | `_ntuple`；`make_divisible`；`extend_tuple` |
| [inplace_abn.py](../models/layers/swins/layers/inplace_abn.py) | 87 | `InplaceAbn`（`__init__`, `reset_parameters`, `forward`） |
| [lambda_layer.py](../models/layers/swins/layers/lambda_layer.py) | 133 | `rel_pos_indices`；`LambdaLayer`（`__init__`, `reset_parameters`, `forward`） |
| [linear.py](../models/layers/swins/layers/linear.py) | 19 | `Linear`（`forward`） |
| [median_pool.py](../models/layers/swins/layers/median_pool.py) | 49 | `MedianPool2d`（`__init__`, `_padding`, `forward`） |
| [mixed_conv2d.py](../models/layers/swins/layers/mixed_conv2d.py) | 51 | `_split_channels`；`MixedConv2d`（`__init__`, `forward`） |
| [ml_decoder.py](../models/layers/swins/layers/ml_decoder.py) | 156 | `add_ml_decoder_head`；`TransformerDecoderLayerOptimal`（`__init__`, `__setstate__`, `forward`）；`GroupFC`（`__init__`, `__call__`）；`MLDecoder`（`__init__`, `forward`） |
| [mlp.py](../models/layers/swins/layers/mlp.py) | 126 | `Mlp`（`__init__`, `forward`）；`GluMlp`（`__init__`, `init_weights`, `forward`）；`GatedMlp`（`__init__`, `forward`）；`ConvMlp`（`__init__`, `forward`） |
| [non_local_attn.py](../models/layers/swins/layers/non_local_attn.py) | 145 | `NonLocalAttn`（`__init__`, `forward`, `reset_parameters`）；`BilinearAttnTransform`（`__init__`, `resize_mat`, `forward`）；`BatNonLocalAttn`（`__init__`, `forward`） |
| [norm.py](../models/layers/swins/layers/norm.py) | 117 | `GroupNorm`（`__init__`, `forward`）；`GroupNorm1`（`__init__`, `forward`）；`LayerNorm`（`__init__`, `forward`）；`LayerNorm2d`（`__init__`, `forward`）；`_is_contiguous`；`_layer_norm_cf`；`_layer_norm_cf_sqm`；`LayerNormExp2d`（`__init__`, `forward`） |
| [norm_act.py](../models/layers/swins/layers/norm_act.py) | 252 | `BatchNormAct2d`（`__init__`, `forward`）；`SyncBatchNormAct`（`forward`）；`convert_sync_batchnorm`；`_num_groups`；`GroupNormAct`（`__init__`, `forward`）；`LayerNormAct`（`__init__`, `forward`）；`LayerNormAct2d`（`__init__`, `forward`） |
| [padding.py](../models/layers/swins/layers/padding.py) | 56 | `get_padding`；`get_same_padding`；`is_static_pad`；`pad_same`；`get_padding_value` |
| [patch_embed.py](../models/layers/swins/layers/patch_embed.py) | 39 | `PatchEmbed`（`__init__`, `forward`） |
| [pool2d_same.py](../models/layers/swins/layers/pool2d_same.py) | 73 | `avg_pool2d_same`；`AvgPool2dSame`（`__init__`, `forward`）；`max_pool2d_same`；`MaxPool2dSame`（`__init__`, `forward`）；`create_pool2d` |
| [pos_embed.py](../models/layers/swins/layers/pos_embed.py) | 207 | `pixel_freq_bands`；`inv_freq_bands`；`build_sincos2d_pos_embed`；`build_fourier_pos_embed`；`FourierEmbed`（`__init__`, `forward`）；`rot`；`apply_rot_embed`；`apply_rot_embed_list`；`apply_rot_embed_split`；其余定义见文件 |
| [selective_kernel.py](../models/layers/swins/layers/selective_kernel.py) | 119 | `_kernel_valid`；`SelectiveKernelAttn`（`__init__`, `forward`）；`SelectiveKernel`（`__init__`, `forward`） |
| [separable_conv.py](../models/layers/swins/layers/separable_conv.py) | 76 | `SeparableConvNormAct`（`__init__`, `in_channels`, `out_channels`, `forward`）；`SeparableConv2d`（`__init__`, `in_channels`, `out_channels`, `forward`） |
| [space_to_depth.py](../models/layers/swins/layers/space_to_depth.py) | 53 | `SpaceToDepth`（`__init__`, `forward`）；`SpaceToDepthJit`（`__call__`）；`SpaceToDepthModule`（`__init__`, `forward`）；`DepthToSpace`（`__init__`, `forward`） |
| [split_attn.py](../models/layers/swins/layers/split_attn.py) | 84 | `RadixSoftmax`（`__init__`, `forward`）；`SplitAttn`（`__init__`, `forward`） |
| [split_batchnorm.py](../models/layers/swins/layers/split_batchnorm.py) | 75 | `SplitBatchNorm2d`（`__init__`, `forward`）；`convert_splitbn_model` |
| [squeeze_excite.py](../models/layers/swins/layers/squeeze_excite.py) | 74 | `SEModule`（`__init__`, `forward`）；`EffectiveSEModule`（`__init__`, `forward`） |
| [std_conv.py](../models/layers/swins/layers/std_conv.py) | 133 | `StdConv2d`（`__init__`, `forward`）；`StdConv2dSame`（`__init__`, `forward`）；`ScaledStdConv2d`（`__init__`, `forward`）；`ScaledStdConv2dSame`（`__init__`, `forward`） |
| [test_time_pool.py](../models/layers/swins/layers/test_time_pool.py) | 52 | `TestTimePoolHead`（`__init__`, `forward`）；`apply_test_time_pool` |
| [trace_utils.py](../models/layers/swins/layers/trace_utils.py) | 13 | `_float_to_int` |
| [weight_init.py](../models/layers/swins/layers/weight_init.py) | 125 | `_trunc_normal_`；`trunc_normal_`；`trunc_normal_tf_`；`variance_scaling_`；`lecun_normal_` |


## modules

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [detection_event.py](../modules/detection_event.py) | 661 | `Module`（`__init__`, `setup`, `forward`, `get_worker_id_from_batch`, `get_data_from_batch`…） |
| [detection_frame.py](../modules/detection_frame.py) | 664 | `Module`（`__init__`, `setup`, `forward`, `set_model_to_gpus`, `get_worker_id_from_batch`…） |
| [detection_fusion.py](../modules/detection_fusion.py) | 773 | `Module`（`__init__`, `setup`, `set_model_to_gpus`, `forward`, `get_worker_id_from_batch`…） |


## modules/data

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [genx.py](../modules/data/genx.py) | 198 | `get_dataloader_kwargs`；`DataModule`（`__init__`, `get_dataloading_hw`, `set_mixed_sampling_mode_variables_for_train`, `setup`, `train_dataloader`…） |


## modules/utils

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [detection.py](../modules/utils/detection.py) | 162 | `Mode`；`BackboneFeatureSelector`（`__init__`, `reset`, `add_backbone_features`, `get_batched_backbone_features`）；`EventReprSelector`（`__init__`, `reset`, `__len__`, `add_event_representations`, `get_event_representations_as_list`）；`RNNStates`（`__init__`, `_has_states`, `recursive_detach`, `recursive_reset`, `save_states_and_detach`…）；`mixed_collate_fn`；`merge_mixed_batches` |
| [fetch.py](../modules/utils/fetch.py) | 36 | `fetch_model_module`；`fetch_data_module` |


## scripts

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [download_faod_datasets.py](../scripts/download_faod_datasets.py) | 155 | `main` |
| [inspect_faod_model.py](../scripts/inspect_faod_model.py) | 91 | `main` |
| [wait_and_extract_pku.py](../scripts/wait_and_extract_pku.py) | 128 | `status`；`valid_file`；`other_extractors`；`main`：等待迁移并校验续解压 |


## tests/compatibility

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [export_mmcv_reference.py](../tests/compatibility/export_mmcv_reference.py) | 29 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [smoke_training.py](../tests/compatibility/smoke_training.py) | 95 | `Clips`（`__len__`, `__getitem__`）；`Record`（`__init__`, `on_before_optimizer_step`, `on_train_batch_end`）；`main` |
| [test_upgrade.py](../tests/compatibility/test_upgrade.py) | 205 | `Sequence`（`__init__`, `__len__`, `__getitem__`）；`stream_results`；`test_stream_adapters`；`test_stream_no_loss_or_duplication`；`test_rank_worker_ownership`；`test_original_stream_equivalence`；`deform_reference`；`test_deform_output_and_gradient_reference`；`test_zero_offset_and_checkpoint_contract`；其余定义见文件 |


## utils

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [checkpoints.py](../utils/checkpoints.py) | 19 | `resolve_checkpoint_path` |
| [helpers.py](../utils/helpers.py) | 14 | `torch_uniform_sample_scalar`；`clamp` |
| [padding.py](../utils/padding.py) | 74 | `InputPadderFromShape`（`__init__`, `_pad_tensor_impl`, `pad_tensor_repr`, `pad_tensor_ev_repr`, `pad_token_mask`） |
| [preprocessing.py](../utils/preprocessing.py) | 12 | `_blosc_opts` |
| [timers.py](../utils/timers.py) | 95 | `CudaTimer`（`__init__`, `__enter__`, `__exit__`）；`cuda_timer_decorator`；`TimerDummy`（`__init__`, `__enter__`, `__exit__`）；`Timer`（`__init__`, `__enter__`, `__exit__`）；`print_timing_info` |


## utils/evaluation/prophesee

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../utils/evaluation/prophesee/__init__.py) | 0 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [evaluation.py](../utils/evaluation/prophesee/evaluation.py) | 41 | `evaluate_list` |
| [evaluator.py](../utils/evaluation/prophesee/evaluator.py) | 72 | `PropheseeEvaluator`（`__init__`, `_reset_buffer`, `_add_to_buffer`, `_get_from_buffer`, `add_predictions`…） |


## utils/evaluation/prophesee/io

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../utils/evaluation/prophesee/io/__init__.py) | 0 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [box_filtering.py](../utils/evaluation/prophesee/io/box_filtering.py) | 36 | `filter_boxes` |
| [box_loading.py](../utils/evaluation/prophesee/io/box_loading.py) | 103 | `reformat_boxes`；`loaded_label_to_prophesee`；`to_prophesee` |
| [dat_events_tools.py](../utils/evaluation/prophesee/io/dat_events_tools.py) | 227 | `load_td_data`；`_dat_transfer`；`stream_td_data`；`count_events`；`parse_header`；`write_header`；`write_event_buffer` |
| [npy_events_tools.py](../utils/evaluation/prophesee/io/npy_events_tools.py) | 62 | `stream_td_data`；`parse_header` |
| [psee_loader.py](../utils/evaluation/prophesee/io/psee_loader.py) | 252 | `PSEELoader`（`__init__`, `reset`, `event_count`, `get_size`, `__repr__`…） |


## utils/evaluation/prophesee/metrics

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../utils/evaluation/prophesee/metrics/__init__.py) | 0 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [coco_eval.py](../utils/evaluation/prophesee/metrics/coco_eval.py) | 198 | `evaluate_detection`；`_match_times`；`_coco_eval`；`coco_eval_return_metrics`；`_to_coco_format` |


## utils/evaluation/prophesee/visualize

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [__init__.py](../utils/evaluation/prophesee/visualize/__init__.py) | 0 | 无顶层类/函数；查看导入、常量或脚本主体 |
| [vis_utils.py](../utils/evaluation/prophesee/visualize/vis_utils.py) | 110 | `make_binary_histo`；`draw_bboxes_bbv`；`draw_bboxes` |


## visualization

| 文件 | 行数 | 主要符号（自动提取） |
| --- | ---: | --- |
| [dataset_visulization.py](../visualization/dataset_visulization.py) | 166 | `sort_key` |


## 全部 YAML 配置

| 文件 | 所属用途 |
| --- | --- |
| [config/dataset/base.yaml](../config/dataset/base.yaml) | 运行/训练 Hydra 配置 |
| [config/dataset/dsec.yaml](../config/dataset/dsec.yaml) | 运行/训练 Hydra 配置 |
| [config/dataset/pku_fusion.yaml](../config/dataset/pku_fusion.yaml) | 运行/训练 Hydra 配置 |
| [config/experiment/dsec/base.yaml](../config/experiment/dsec/base.yaml) | 运行/训练 Hydra 配置 |
| [config/experiment/dsec/default.yaml](../config/experiment/dsec/default.yaml) | 运行/训练 Hydra 配置 |
| [config/experiment/dsec/small.yaml](../config/experiment/dsec/small.yaml) | 运行/训练 Hydra 配置 |
| [config/experiment/dsec/tiny.yaml](../config/experiment/dsec/tiny.yaml) | 运行/训练 Hydra 配置 |
| [config/experiment/pku_fusion/base.yaml](../config/experiment/pku_fusion/base.yaml) | 运行/训练 Hydra 配置 |
| [config/experiment/pku_fusion/default.yaml](../config/experiment/pku_fusion/default.yaml) | 运行/训练 Hydra 配置 |
| [config/experiment/pku_fusion/small.yaml](../config/experiment/pku_fusion/small.yaml) | 运行/训练 Hydra 配置 |
| [config/experiment/pku_fusion/tiny.yaml](../config/experiment/pku_fusion/tiny.yaml) | 运行/训练 Hydra 配置 |
| [config/general.yaml](../config/general.yaml) | 运行/训练 Hydra 配置 |
| [config/model/base.yaml](../config/model/base.yaml) | 运行/训练 Hydra 配置 |
| [config/model/maxvit_yolox/default.yaml](../config/model/maxvit_yolox/default.yaml) | 运行/训练 Hydra 配置 |
| [config/model/rnndet.yaml](../config/model/rnndet.yaml) | 运行/训练 Hydra 配置 |
| [config/train.yaml](../config/train.yaml) | 运行/训练 Hydra 配置 |
| [config/val.yaml](../config/val.yaml) | 运行/训练 Hydra 配置 |
| [frame_construction/conf_preprocess/extraction/const_count.yaml](../frame_construction/conf_preprocess/extraction/const_count.yaml) | 离线数据构建参数 |
| [frame_construction/conf_preprocess/extraction/const_duration.yaml](../frame_construction/conf_preprocess/extraction/const_duration.yaml) | 离线数据构建参数 |
| [frame_construction/conf_preprocess/filter_dsec.yaml](../frame_construction/conf_preprocess/filter_dsec.yaml) | 离线数据构建参数 |
| [frame_construction/conf_preprocess/filter_gen1.yaml](../frame_construction/conf_preprocess/filter_gen1.yaml) | 离线数据构建参数 |
| [frame_construction/conf_preprocess/filter_gen4.yaml](../frame_construction/conf_preprocess/filter_gen4.yaml) | 离线数据构建参数 |
| [frame_construction/conf_preprocess/filter_pku_fusion.yaml](../frame_construction/conf_preprocess/filter_pku_fusion.yaml) | 离线数据构建参数 |
| [frame_construction/conf_preprocess/representation/mixeddensity_stack.yaml](../frame_construction/conf_preprocess/representation/mixeddensity_stack.yaml) | 离线数据构建参数 |
| [frame_construction/conf_preprocess/representation/stacked_hist.yaml](../frame_construction/conf_preprocess/representation/stacked_hist.yaml) | 离线数据构建参数 |


## 根目录 Shell 包装脚本

这些脚本主要是历史命令缩写，通常没有覆盖当前数据路径。使用主教程中的显式参数命令。

| 文件 | 当前命令 |
| --- | --- |
| [demo.sh](../demo.sh) | `python demo.py dataset=pku_fusion +experiment/pku_fusion='base.yaml'` |
| [test.sh](../test.sh) | `python validation.py dataset=pku_fusion +experiment/pku_fusion='base.yaml'` |
| [test_DSEC.sh](../test_DSEC.sh) | `python validation.py dataset=dsec +experiment/dsec='base.yaml'` |
| [train.sh](../train.sh) | `python train.py dataset=pku_fusion +experiment/pku_fusion='base.yaml'` |
| [train_DSEC.sh](../train_DSEC.sh) | `python train.py dataset=dsec +experiment/dsec='base.yaml'` |
