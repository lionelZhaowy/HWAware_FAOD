> 历史需求归档：用户最初提供的算法交接文本。当前阶段和后续用户修正见 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。本文不作为立即实施全部算法改造的指令。

# FAOD 算法 Agent 交接 Prompt（独立上下文版）

你是负责算法实现的 Agent，运行在 GPU 服务器上。请在用户已 fork 的 FAOD 工程中完成下面的算法改造。你无法访问讨论论文的 Agent 所在电脑，也不需要读取那台电脑的任何 Markdown、RTL、ChipC 编译器或 VCS 工程；必要上下文已经完整写在本消息中。

请把本消息作为实施需求，先检查当前仓库与服务器环境，然后执行代码修改和必要验证。不要重新开展一次大范围算法选型综述，也不要只给计划而不实现。已有讨论包含实验设想，尚未验证的部分在下面明确标出；若源码或实验否定某个设想，请具体指出并给出可验证的最小调整。

## 一、背景、最终目标和当前任务范围

用户在研发 Dremi 多芯视觉向量处理器，面向 ASIC 论文，希望展示 Gray–DVS 多速率融合感知。现有芯片暂不再次流片，因此算法应尽可能使用规整的卷积、矩阵乘、加法、逐元素乘和 ReLU 等计算，不依赖新增复杂硬件。

当前已 fork FAOD，决定以它作为算法工程起点。原 FAOD 的算法成绩、原 EfficientViT 的芯片验证结果，均不能直接当作新融合算法的成绩。

最终希望形成一套统一的软件结构：

```text
MIT EfficientViT 空间编码
        ↓
低频 Gray 特征缓存 + 高频 DVS 编码 + 4/8 帧有限窗口时序融合
        ↓
共享多尺度特征
        ├─ 目标条件模块 + CenterNet 式 heatmap / xy / wh → 检测或 SOT
        └─ 轻量分割 decoder / head → 闭集语义分割
```

不同任务允许使用不同 checkpoint、不同损失和数据集；不要求一套权重同时完成三种任务。首先统一模型类、骨干/融合/状态接口和基础算子结构，不必为了字面上“完全相同”而固定所有任务的输出通道数。

**当前优先级：先完成检测的模型、时序训练、数据适配和推理闭环。** 将 SOT 的目标条件接口及共享中心头纳入结构设计，并提供必要的模型/监督自检；分割保留同一多尺度接口和轻头扩展点。没有对应真实数据时，不要求首轮同时训练好三个任务，也不能把仅有接口写成已经完成 SOT 或分割算法。首版跑通后，再根据实际数据推进 SOT 与分割训练。

跟踪只做 **SOT（单目标跟踪）**；不要引入多目标身份关联和相关评测系统。

## 二、不可混淆的输入与硬件约束

### 2.1 输入条件

- 实验室 Gray：**宽 1024 × 高 720，30 FPS，单通道灰度**。
- 实验室 DVS：**宽 512 × 高 360，输出时间表面帧，240 FPS**。
- Tensor 建议统一 `[B,C,H,W]`，文中尺寸“1024×720”均为宽×高，避免写反。
- 事件 tick 间隔约 `1/240 s = 4.1667 ms`；每个 Gray 周期约有 8 个 DVS tick。
- 240 FPS 是传感输入能力和目标更新预算，不是现有新算法已经达到的吞吐。
- DVS 时间表面的极性通道数、衰减函数、时间常数还未最终确定。请做成配置参数，不要沿用 FAOD 默认 20 通道 voxel 而称为实验室输入。
- 若数据/时间表面参数缺失，可以用明确标注的合成张量开展接口和数值测试；真实训练前必须核验实际事件表示。未经确认的通道数、tau 不能成为隐藏默认的“传感器事实”。

### 2.2 几何与数据集

检测主要面向 **PEOD**。已有讨论记录其为 30 Hz 图像/检测标注的 Event–RGB 数据集，常见发布尺寸为 1280×720、六类；六类名称记录为 car、bus、truck、two-wheeler、three-wheeler、person。**以服务器上实际发布版本的元数据、类别映射和划分为准**，不要硬编码未经验证的 class ID。

PEOD 原生尺寸与实验室传感器尺寸不同。必须明确写出数据变换：原始图像/事件/GT 坐标 → 标定/共同视野 → 裁剪、缩放或 padding → 输出坐标 → 评测逆变换。不得把插值后的事件图叫作原生 512×360 实验室事件数据，也不要把 1280×720 直接当作实验室 Gray 输入。

2:1 的分辨率比例不代表两个实际传感器天然同视野、无视差或严格对齐；标定资料缺失时，用明确的共同视野假设开展软件实验，并在文档中说明限制。

RGB 数据转 Gray 时固定颜色转换、输入范围和标准化，调整不适用的颜色增强。最终单通道入口要实际实现；复制成三通道可以做对照，但不等于已降低输入侧计算。如果初始化单通道卷积以复现“Gray 复制三通道”，在各通道预处理一致的条件下应将原三个输入通道权重相加；原 RGB 标准化不一致时须同时处理缩放/偏置，不能无条件套用权重平均。

可选 padding 为 Gray 1024×768、DVS 512×384，但不是必须值。若采用此方案，保存 valid mask 和逆变换，剔除 padding 区域的训练/预测。Gray /16 和 DVS /8 在此配置下同为 64×48；不要假定任意同编号 stage 都有相同尺寸和语义深度，也不要凭空增加 EfficientViT 不存在的 /64 输出层。

### 2.3 骨干与运算

- 使用 **MIT Han Lab 的 EfficientViT（LiteMLA，高分辨率密集预测版本）**，不是其他同名 EfficientViT。
- 用户已经在 Dremi 上验证过该路线，并且**已经移除了 LiteMLA 的除法归一化，实验效果仍好**。新默认路径按无除法归一化实现；不要重新引入历史 z 或 Qz 分母作为必需项。
- 若服务器没有用户实际部署的 EfficientViT 版本，先检查已有文件/依赖；可基于正确的官方版本建立适配器，记录上游 SHA 和全部差异。此时不能宣称与芯片部署版本逐层相同，后续要留出对齐检查。
- 新时序模块不使用 GRU/LSTM；sigmoid/tanh 在当前 Dremi 算子库尚未实现，因此不要把实现它们设为本轮前提。
- 新部署主路径避免 DCN、Softmax attention、AdaIN、动态 token 剪枝、稀疏图网络和 SNN 专用执行。原论文对照模型可以保留这些运算，但不能混进新模型默认依赖链。
- 默认使用 Conv/DWConv/PWConv、MatMul、Add/Mul、ReLU 和明确的 reshape/固定尺寸操作；并不代表这些运算任意形状都已被芯片验证。导出时列出完整算子清单与张量形状。
- 不要全局删除所有归一化：去掉的是用户指定的 LiteMLA 除法归一化。BN 可在 eval/export 折叠，其他激活/归一化按具体实现逐项说明；训练损失里的 sigmoid、log、交叉熵可以在 GPU 上使用。

## 三、FAOD 工程哪些部分复用，哪些部分替换

本地论文 Agent 曾于 2026-09-13 静态核查官方仓库 `Hatins/FAOD-master`，提交为：

```text
e8666ca536850807173502e6764423135194cf7f
```

用户 fork 可能已有更新，请**先对比实际代码再判断问题是否仍存在**。以下是待核查线索，不是要求盲目打补丁。

### 3.1 值得保留的部分

- `train.py`：训练入口、配置组装。
- `modules/utils/fetch.py`：模型/数据模块选择。
- `data/ev_img_dataloader/sequence_rnd.py`：双模态随机序列采样。
- `data/ev_img_dataloader/sequence_for_streaming.py`：流式序列与边界处理。
- `modules/detection_fusion.py::training_step_with_rnn`：逐 tick 状态更新、有标签特征选择、片段末 detach。
- `modules/utils/detection.py::BackboneFeatureSelector` 和 `RNNStates`：稀疏监督与状态管理思路。

### 3.2 已发现的具体风险

1. `config/model/maxvit_yolox/default.yaml` 名称含 maxvit，但当时默认配置为 `backbone_type: darknet`；工厂 `models/detection/recurrent_backbone/__init__.py` 中 `maxvit` 分支也导入 `darknet_rnn_forward_fusion`。必须记录实际实例化的类，不能只看配置名称。
2. Darknet 版 `forward_rnn` 关闭 align 时出现 `img_input_aligned == None`，并未赋值；MaxViT 版也有关闭对齐后变量未初始化的问题。不能只设 `enable_align=False` 就假定新路径可运行。
3. `models/layers/align_and_fusion/align.py::Feature_wrapper` 调用 EgF、AdaIN 和 ALIGN；`EgF.py::ALIGN` 使用 `DCNv2Pack`，EgF 另含 Softmax/LayerNorm/sigmoid。新模型应该注册独立适配器并避开这条依赖链。
4. `models/detection/yolox_extension/models/detector_fusion.py::YoloXDetector.forward` 当时调用遗留的 `self.forward_backbone(...)`，实际训练走 `forward_backbone_rnn/forward_detect`。不要未经检查就拿原 `forward` 导出。
5. 原融合骨干通常首阶段分开编码两路，然后共用后续阶段，并不是两套完整 backbone。原代码重复接收相同图像，也不等于真正缓存并跳过 Gray 编码。
6. 原状态工具对 Tensor 第 0 维按样本 reset；新 FIFO 若布局为 `[W,B,...]` 会出错。建议 `[B,W,...]`，并扩展有效位、时间戳、指针、跨 worker/序列归属。
7. 部分顶层 import 会加载未启用的 DCN/Mamba/S5 等实验依赖。新模型不要为无关代码引入大量安装负担。

**最终默认预测头改为 CenterNet 式三分支，因此 YOLOX head/loss 不再是必须保留项。** 原 `labels_yolox`、`forward_detect`、PAFPN/head 的输入输出适配需要同步修改。可以保留原 FAOD/YOLOX 配置作为原始对照，但不要让两种 target/loss 格式混用。

## 四、空间结构：先完成小改动版本，预留双骨干布局

第一版建议实现：

```text
新 Gray 到达 → Gray 浅层编码 → 缓存 ─┐
                                   ├→ Add/Concat + 投影 → 共享 EfficientViT 主体
每 DVS tick → DVS 浅层编码 ─────────┘                          ↓
                                              多尺度时序模块 → 统一中心点头
```

这保留 FAOD 早期融合、后续共享的组织，减少双完整骨干开销。Gray 浅层缓存只在新的 Gray 真正可用时更新；不得每 tick 重算 Gray 后再称其为缓存方案。

另预留较晚融合接口：Gray EfficientViT 与 DVS EfficientViT 各自编码，多尺度对齐后融合。它更适合研究模态并行，但不是强制第一版同时实现的两套大网络。请把骨干布局配置和接口设计清楚，先把早期融合的检测闭环跑通，再决定是否推进较晚融合对照。

空间 EfficientViT 尽量保持前馈结构，把时序模块放在选定多尺度输出或公共特征层之后。不要机械地把 FAOD 每个 LSTM stage 改成另一种 RNN，也不要声称共享主体布局完全复现已部署整网。

## 五、时序核心：无归一化 LiteMLA 的 4/8 帧 S-FIFO

### 5.1 数学定义

以下为每个 batch、head、可选空间分区的定义。对第 t 个事件 tick 的当前特征产生：

```text
Q_t ∈ R^(Nq × dk)
K_t ∈ R^(Nk × dk)
V_t ∈ R^(Nk × dv)
φ = ReLU
S_t = φ(K_t)^T V_t ∈ R^(dk × dv)
```

FIFO 保留最近 W 个 S，W 支持 1/4/8，默认先 W=8。每次只计算新帧 S，复用旧项，不重算窗口内全部旧 backbone 或旧 K/V。

当前 Q 应包含当前 DVS 信息，也可融合缓存 Gray。保留当前空间特征的残差/跳连，使有限大小的 S 不承担全部定位信息。S 不保留显式逐 token 位置，必要时采用固定空间分区/位置特征；分区数量、head 数、dk/dv 尚未通过新算法验证，请配置化并报告预算，不要把某个历史示例数值当成硬件规格。

不要构建随视频时长增长的完整 KV cache。保留 KᵀV 统计是线性注意力的重排/压缩，不是直接移植 DeepSeek MLA，也不是 GRU 式隐藏状态。

### 5.2 必须支持的两种读取语义

**A：合并后读取，作为低计算量对照。**

```text
M_t = Σ_r a_r S_(t-r)
Y_t = φ(Q_t) M_t
```

等权和可以维护滚动和，新项加入、旧项减去。非均匀时间位置权重在窗口移动后会变化，不能直接照抄等权滚动加减。固定常量缩放可由参数/量化尺度处理，但不要悄悄加入依赖当前数据的除法归一化。

**B：逐帧读取，再保留时间轴融合，作为显式时间信息候选。**

```text
Y_(t,r) = φ(Q_t) S_(t-r), r=0...W−1
Y_t = TemporalFuse([Y_(t,0), ..., Y_(t,W−1)])
```

旧 7 个 S 加当前 S 构成 W=8 窗口，每个时间槽仍可辨识；明确新→旧或旧→新的排列。TemporalFuse 可用固定尺寸通道拼接、小卷积/线性投影和 ReLU，使用相对年龄/真实时间差时记录定义。不要拼接后立即等权求和，却声称保留了时间顺序。

固定纯线性融合有时可以提前合并 S；包含非线性或输入相关选择的融合不能普遍折成等权和。请提供参考实现与必要的等价性测试，避免错误优化。先写清楚可靠的两种模式，再考虑降低中间 `[B,W,...]` 特征的驻留开销。

两种模式均 **没有 z，也没有 Qz 除法分母**。模式 B 读取成本通常随 W 增加，即使存储已压缩，也不能声称其计算量与单次读取相同。

单个序列、一个特征尺度的 S-FIFO 理论容量为 `W × 分区数 × head数 × dk × dv × 每元素字节数`；多尺度需要求和，还要加 Gray 缓存、当前特征、模板、有效位及读出临时张量。分别报告实际 dtype 和完整峰值显存，不能只报告 S 的容量。尚无服务器可读取的 Dremi 精确容量预算时，报告配置间的取舍，不编造片上内存余量。

### 5.3 流式状态与梯度

- 显式状态建议包含 `gray_features/gray_timestamp`、各尺度 `S_fifo/valid_mask/timestamps/write_index`。
- SOT 目标模板是另一类持久状态，不能放入 8 帧 FIFO 自动淘汰。
- 序列开始不足 W 帧时用有效位，确保无效槽不会因时间偏置或带 bias 的融合模块伪造历史。
- 跨视频、场景重启、尺寸改变时重置必要状态；独立样本 shuffle 不能共用跨样本 FIFO。
- batch 内不同样本允许 Gray 刷新不同步、序列结束不同步、有效长度不同，不能只用一个全局 Python 指针解决所有样本。
- clip 内保留所需梯度，clip 边界按 TBPTT 策略 detach。不要每 tick detach S，使所有历史帧失去训练信号；也不要保留整段长视频计算图导致内存无限增长。
- GPU 训练避免 inplace 覆盖仍被 autograd 使用的历史值；推理可采用固定大小 ring buffer。流式和离线参考需要在 eval 模式数值一致。
- 多 worker/DDP 保证状态按正确样本和序列归属，不要把 worker ID 当成永远不变的视频 ID。
- 不在 optimizer 更新后无限持有旧权重生成的带梯度 Gray cache；明确训练片段边界/重计算和截断策略。eval 下“重复编码”与“缓存复用”等价不代表训练 BN/dropout 任意设置下也等价。

## 六、统一 CenterNet 式检测/SOT 预测头

### 6.1 输出与解码

默认使用轻量公共 decoder/共享卷积 stem，再分为：

```text
heatmap_logits: [B, K, Hh, Wh]
xy_offset:      [B, 2, Hh, Wh]
wh_size:        [B, 2, Hh, Wh]
```

检测 K 按实际类别数，SOT K=1；如需固定形状可配置 K_max 并屏蔽无效通道，不作为第一版必要条件。

建议统一以 head 网格单位回归偏移和宽高。stride 为 r、选中格点为 `(u,v)`：

```text
cx = r × (u + dx), cy = r × (v + dy)
w_img = r × w,     h_img = r × h
```

再生成角点框，并按数据几何变换回原评测坐标。xy 是中心的网格量化偏移，**不是上一帧到当前帧的运动位移**。

检测按类别热图做局部峰值提取、top-k 和阈值筛选，不是把所有超过阈值的像素都输出一个框。SOT 对目标条件响应取最大位置；目标不在场时 argmax 仍会有结果，因此需按短期/长期协议定义无效预测处理，不能自行宣称具备重捕获能力。

实现 Gaussian 中心 target、heatmap focal 类损失、中心 xy/wh 回归损失；SOT 可选 IoU/GIoU。回归 mask、空目标帧、多目标同格点冲突、边界和 padding 均需明确定义。**已标注的空场景是负样本，无标注时刻不是负样本**。

heatmap 最后一层输出线性 logits，不加 ReLU。由于 sigmoid 单调，纯 argmax/top-k/局部峰值可用 logits，概率阈值 τ 可预先换算为 `log(τ/(1−τ))`，减少芯片整图 sigmoid。不要把 `sigmoid(L)×Hann` 改写成 `L×Hann`；这一般不等价。宽高若参考 OSTrack 的 sigmoid 参数化，也不能直接删除 sigmoid 后沿用原解码，应改成完整一致的直接回归/非负方案并训练。

### 6.2 SOT 必须有目标条件

同一场景指定跟踪 A 或 B，系统应产生不同输出。没有模板、首帧框初始化或目标状态，仅靠换数据集和 argmax 不能做到这一点。

推荐公共接口：检测使用固定可学习条件 `c_det`，SOT 使用首帧框提取的目标特征 `c_sot`。用固定大小的逐元素乘/MatMul 与当前特征交互，再进入同一个中心头。一个可检验的最小候选是：

```text
U = Conv1×1(ReLU(Conv1×1(Concat(F, F⊙Broadcast(c), Broadcast(c)))))
```

这只是尚未训练验证的候选，不是已知最佳结构。单向量模板可能丢失形状；可预留小网格模板匹配选项。首轮先固定首帧模板，不自动引入复杂在线更新。

推荐场景 backbone/fuser/S-FIFO 不依赖目标模板，把条件交互放在后端；这样更换模板不必重算全部场景缓存。优先全图 SOT 后端匹配，减少“上一帧最终框决定下一输入 ROI”的流水反馈。ROI 模式可以另做配置，但需明确计算量和反馈差异。

SOT 训练只把指定目标标成正样本，不能把场景所有对象都标正后让最大值自行选人。目标模板只使用允许的初始/过去信息，禁止每 tick 用未来 GT 重建模板。

### 6.3 分割接口

复用同一多尺度骨干/Gray–DVS 融合/S-FIFO，接轻量语义分割 decoder，保留当前空间特征和浅层跳连，输出 `[B,C_seg,Hout,Wout]` logits。仅需类别图时逐像素 argmax 可省去推理 softmax。

PEOD 检测框不能直接监督语义分割；有 DSEC-Semantic 等数据时按其真实类别、标签来源和视野变换训练。首轮缺分割数据则明确交付的是接口或合成自检，不编造 mIoU。

## 七、240 Hz 更新、30 Hz 标签的训练方式

1. 以真实时间戳建立事件 tick，保留事件截止时间、Gray 时间和 GT 时间。不要通过反复累加取整后的 4167 微秒造成长序列时钟漂移。
2. 每 tick 更新模型状态，只在对应人工标注的时刻计算直接损失。不要只跑有标签帧，也不要把 30 Hz GT 复制到另外 7 个时刻当真值。
3. 初版优先 W=8，clip 长度按显存选取并记录 warm-up/TBPTT。W=4 且每个 S 只依赖本帧时，部分无标签 tick 可能不进入任何标签时刻的窗口；检查梯度覆盖，不能声称它们都受到了间接监督。
4. Gray 使用当前时刻以前已经可用的最近图像。Time Shift 只选择更旧 Gray，不移动当前 GT；不盲用 FAOD 的 `label_shift/image_shift` 索引规则。
5. 时间表面若已经含历史衰减，再跨帧记忆会重复呈现旧事件；记录表面时间常数、采样步长与 FIFO 的关系，通过消融判断，不能把每张表面都当作互不重叠的新事件。
6. 训练/验证/测试按序列划分，不能将相邻片段或生成的模板泄漏到不同划分。量化/阈值校准不用测试集。
7. 先评测 Gray 30/15/7.5 Hz、保持原 30 Hz 检测标注，以验证旧 Gray 期间事件是否有效；240 Hz 更新的 30 Hz AP 不能证明全部中间预测精度。
8. 伪标签/插值标签如后续启用，单独标识训练用途，不能充当额外人工测试真值。本轮不以实现 LEOD/FlexTune 整套自训练为前提。

## 八、最小但有判别力的验证要求

先检查服务器可用 GPU、显存、现有环境、依赖与数据，避免影响用户正在运行的其他作业。优先独立环境和当前 fork 的新配置；保留用户已有修改，不覆盖原始 checkpoint/数据。可执行必要的单 batch、短序列、短训练验证；不要自行启动长时间全数据多组 sweep。记录实际执行命令和资源，给出完整训练命令。

请优先完成这些有实际意义的验证：

- **原工程定位：**实际选中的 backbone、训练/验证入口及现存问题，不把旧核查 SHA 的问题当作用户 fork 已证实的 bug。
- **双分辨率/单通道：**目标尺寸或等比例小尺寸的前向、损失与反向；至少检查目标尺寸真实 forward，资源不足如实记录，不能仅测试方形低分辨率输入。
- **缓存数值：**短序列逐帧重算参考与 FIFO 复用一致；均匀合并模式与合法矩阵重排一致；拼接模式的时间顺序正确。
- **状态隔离：**两段不同视频连续处理与各自独立处理一致；batch 内单独 reset 不影响其他样本；不足 W 帧和异步 Gray 刷新正确。
- **训练梯度：**有标签时刻对窗口内历史事件编码有梯度；只在 clip 边界截断；无标签步骤不误参与负样本损失。
- **因果性：**改变预测时刻之后的 Gray/事件，不影响该时刻输出；修改未来 GT 不影响推理结果。
- **头与几何：**合成已知中心/偏移/宽高的框解码能回到预期坐标；padding、裁剪、非方形输入和逆变换正确；局部峰值避免同一响应邻域重复框。
- **SOT 条件：**若已实现 SOT 小训练验证，用同一合成/真实片段中的不同初始目标检查目标条件是否被利用；未经训练随机权重输出不同不能证明学会跟踪。至少检查条件数据链与 target 构造，并明确功能测试和算法精度测试的区别。
- **训练可学习性：**真实检测数据可用时完成小子集过拟合/短训练，验证损失、框和热图变化；如果数据不可用，用合成数据验证代码链路并明确限制，不把合成结果写成 PEOD 成绩。

不要为每个 getter 写形式化测试；重点测试上面的数学、状态、监督和坐标契约。模型规模、shape、W、dtype、batch、warm-up、GPU 同步方式与测试范围必须随性能结果记录。

## 九、输出成果与持续交接

请在服务器工程内保存自包含文档，例如 `docs/ALGORITHM_HANDOFF.md`、`docs/IMPLEMENTATION_STATUS.md` 和实验记录，使后续 Agent 不依赖聊天历史。不要引用本地电脑的 V1–V16 文件路径作为必要背景。

应交付：

1. 新模型、状态接口、中心头、target/loss、训练和推理配置；原模型对照入口保留。
2. 数据适配说明和实际参数：尺寸、通道、时间表面定义、标注时刻、共同视野/坐标变换、序列划分。
3. 复现命令：环境、数据预处理、单 batch、短序列、训练、评测、推理；配置默认能看出当前模型到底是哪一版。
4. 必要验证的结果/日志路径，以及未完成或被数据/依赖阻塞的事项。不能写“全部通过”但没有实际运行。
5. 模型结构、各层/各尺度 shape、参数量/MACs、缓存字节数和完整推理算子清单。GPU 性能与 Dremi 性能分开，FP32 S 缓存不等于已验证 INT32 定点实现。
6. 记录已确认事实、临时工程默认值和后续消融项：窗口聚合方式、空间分区、模板表示、早/晚融合、head stride、decoder 通道等。

初次回应简要说明：当前代码/数据状态、计划修改的主要模块、真正缺失的信息，然后继续工作。对可配置的非阻塞问题采用明确的临时参数做接口验证；只有影响真实数据解释或长训练决策的必要事项才询问用户，并继续不依赖答案的工作。

## 十、可直接联网获取的主要参考

以下链接用于核对实现，不是要求完整复现所有论文。遵守所用源码许可证，记录上游提交及本项目改动。

- FAOD：<https://github.com/Hatins/FAOD-master>
- MIT EfficientViT：<https://github.com/mit-han-lab/efficientvit>；重点看 `efficientvit/models/nn/ops.py` 的 LiteMLA 和对应多尺度 backbone。
- CenterNet / Objects as Points：<https://github.com/xingyizhou/CenterNet>。已核查提交 `4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c`；重点 `src/lib/models/decode.py`、`src/lib/datasets/sample/ctdet.py`、`src/lib/trains/ctdet.py`。
- OSTrack：<https://github.com/botaoye/OSTrack>。已核查提交 `33b5e12586216b7fd0e95d255bd01ba44cbec759`；重点 `lib/models/layers/head.py::CenterPredictor`、`lib/models/ostrack/ostrack.py`、`lib/train/actors/ostrack.py`、`lib/test/tracker/ostrack.py`。
- PEOD 正式论文与协议入口：<https://ojs.aaai.org/index.php/AAAI/article/view/38883>。具体数据文件以用户服务器上的发布版为准。
- 分割工程参考 CMX：<https://github.com/huaaaliu/RGBX_Semantic_Segmentation>；DSEC-Semantic 官方标签说明：<https://dsec.ifi.uzh.ch/dsec-semantic/>。其官方标签由图像分割模型生成，不要写成全量人工像素 GT。

现在请从当前 fork 的仓库检查开始，按上述范围实施。不要请求用户把论文 Agent 的所有历史文档重新搬到服务器；如果某项必要信息确实缺失，请指出具体字段、代码接口或数据格式。
