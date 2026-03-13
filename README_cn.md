# BP-SR

[English](README.md) | [简体中文](README_cn.md)

BP-SR 是 `NTIRE 2025 / CVPR 2025 Workshops, Track 2: KwaiSR` 的两阶段解决方案，隶属于 `NTIRE 2025 Challenge on Short-form UGC Video Quality Assessment and Enhancement`。

该公开版本的流程由三部分组成：

- 使用 `DRCT` 对 synthetic 数据进行上采样
- 使用 `FaithDiff` 生成同分辨率增强先验
- 使用 `BPSR_DualStreamCrossAttention` 进行双输入后处理细化

官方参考链接：

- Challenge 页面: https://codalab.lisn.upsaclay.fr/competitions/21346
- 项目主页: https://lixinustc.github.io/NTIRE2025-KVQE-KwaSR-KVQ.github.io/
- Challenge report: https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/A_Li_NTIRE_2025_Challenge_on_Short-form_UGC_Video_Quality_Assessment_and_CVPRW_2025_paper.pdf
- KwaiSR 数据集论文: https://arxiv.org/html/2504.15003v1

## 项目概览

Track 2 并不是一个标准的 synthetic-only 4x 超分 benchmark。

根据官方 KwaiSR 数据说明：

- 数据集包含 `1800` 对 synthetic paired images 和 `1900` 张 wild low-quality images
- 数据划分比例为 `8:1:1`
- synthetic 子集更接近 `4x` SR 问题
- wild 子集已经处于目标分辨率，但仍然带有真实 UGC 退化

因此，这个赛题同时包含：

- `synthetic LR -> HR` 的监督恢复
- `wild 同分辨率低质图像 -> 增强结果` 的真实感知增强

官方开发阶段评分公式为：

`PSNR + 10*SSIM - 10*LPIPS + 0.1*MUSIQ + 10*ManIQA + 10*CLIPIQA`

也就是说，这个任务并不只是提高 PSNR，而是需要同时兼顾：

- 结构保真
- 感知质量
- synthetic / wild 双域泛化能力

## 如何理解官方数据集

官方的 KwaiSR 不能简单理解成一个普通的 paired SR 数据集。

- `synthetic` 子集是有配对监督的 source domain。
  - 提供对齐的 `LR / HR`
  - 更接近一个可控的 `4x` 恢复问题

- `wild` 子集是无配对 GT 的 real target domain。
  - 只包含真实 UGC 低质量图像
  - 不提供 paired GT
  - 主要用于考察真实域下的感知增强能力与泛化能力

官方材料没有规定唯一的 wild 利用方式，但有三个信号非常明确：

- wild 没有 paired GT
- synthetic 和 wild 需要由同一套方法处理
- challenge 页面明确鼓励 diffusion-based 方法

因此，wild 更合理的理解不是“第二套 supervised train set”，而是“无标注真实域数据”。它更适合被用于：

- 先验生成
- pseudo target / pseudo label
- self-training 或 consistency regularization
- no-reference 质量引导
- 面向真实 UGC 退化的 domain adaptation

公开的 BP-SR 采用的是一种更稳健的实现方式：

- 后处理 refinement 网络仍然只在 paired synthetic triplet 上做监督训练
- wild 图像用于同分辨率先验生成与最终推理
- 经过预处理后，同一条 refinement pipeline 同时作用于 synthetic 和 wild

## 设计动机

BP-SR 主要针对 KwaiSR 中的三个实际问题进行设计。

### 1. synthetic 和 wild 输入不在同一个空间

- 原始 synthetic LR 输入为 `270x480`
- 目标分辨率为 `1080x1920`
- wild 图像本身已经是 `1080x1920`，但视觉质量较差

这意味着一部分数据更像传统 4x SR，另一部分数据更像同分辨率增强。直接用一个统一模型处理这两类输入，训练会不稳定。

### 2. 原图与 diffusion 输出各有优缺点

- 原始 `lq` 更可靠地保留了结构和内容锚点
- diffusion 增强结果通常能恢复更丰富的感知细节，但也可能偏离原始输入

只使用其中一路都不理想：

- 只用 `lq`，结果通常偏保守，细节不足
- 只用 diffusion 结果，视觉上更丰富，但控制性更弱

### 3. 赛题本身是多目标优化

官方评分同时包含 fidelity、perceptual 和 no-reference IQA 指标，因此需要设计一个能够在多类目标之间折中的方案。

### 4. wild 子集不能被当作普通 paired supervision

- wild 图像没有 paired GT
- 方法仍然必须从 synthetic supervision 泛化到真实 UGC 输入
- 因此需要把 wild 视为真实域先验和泛化约束来源，而不是直接的像素监督来源

## 方法概述

BP-SR 是一个两阶段流程。

### 阶段 A：分辨率统一与先验生成

- 使用 `DRCT` 将原始 synthetic 数据从 `270x480` 上采样到 `1080x1920`
- 使用 `FaithDiff` 对 synthetic 和 wild 数据在同一 `1080x1920` 分辨率上进行预处理
- 将 FaithDiff 输出作为 `diff` 分支，作为感知先验输入后续模型

这一阶段的作用是把混合输入赛题转换为统一的同分辨率 refinement 问题。

### 阶段 B：双输入后处理细化

最终 refinement 网络联合输入：

- 原始低质量图像 `lq`
- diffusion 增强先验 `diff`

并学习输出一个更稳定、更可靠的最终增强结果。

整体流程可以概括为：

`synthetic LR (270x480) -> DRCT -> 1080x1920 -> FaithDiff -> diff`

`wild LQ (1080x1920) -> FaithDiff -> diff`

`[lq, diff] -> BPSR_DualStreamCrossAttention -> final enhanced image`

## 代码级流程

根目录公开代码的实际张量流如下。

### 1. 数据集构建

`BPSRAlignedTripletDataset` 会扫描三个同名文件夹，并构建严格对齐的 triplet：

- `lq_path`
- `diff_path`
- `gt_path`

训练阶段会执行：

- 对 `lq / diff / gt` 做同位置随机裁剪
- 对三路图像做同步翻转 / 旋转增强
- 可选的 `RGB -> Y` 转换
- 转成 `RGB / CHW / float32` 张量

由于公开配置使用的是 `scale: 1`，三路图像的 patch 尺寸完全相同。对当前训练配置中的 `gt_size: 192`，实际进入模型的 patch 为：

- `lq`: `3 x 192 x 192`
- `diff`: `3 x 192 x 192`
- `gt`: `3 x 192 x 192`

### 2. 包装层输入打包

`BPSRRefinementModel.optimize_parameters()` 和 `BPSRInferenceModel.pre_process()` 都会先执行：

`torch.stack([lq, diff], dim=1)`

因此 generator 的输入张量形状是：

`[B, 2, 3, H, W]`

推理阶段在堆叠之前，还会先对 `lq` 和 `diff` 做 reflect padding，使其尺寸补齐到 `window_size` 的整数倍。

### 3. 共享浅层编码

在 `BPSR_DualStreamCrossAttention.forward()` 中：

- 先对双输入做 mean normalization
- 再把 `[B, 2, 3, H, W]` 展平成 `[2B, 3, H, W]`
- 然后用共享的 `3x3` 卷积 `conv_first` 将两路输入提升到 `embed_dim`

这意味着两路分支共享浅层特征提取，而真正的分支交互发生在后续 attention 空间里。

### 4. 深层特征提取

公开主干的关键配置为：

- `patch_size = 1`
- `embed_dim = 180`
- `depths = [6, 6, 6]`
- `num_heads = [6, 6, 6]`
- `window_size = 16`

其中 `patch_size = 1` 很关键，它表示 tokenization 不会降低空间分辨率。因此，这个网络本质上是一个同分辨率 transformer refinement 网络，而不是低分辨率 latent SR 模型。

深层部分由 `3` 个 `RHAG` 组成，每个 `RHAG` 含 `6` 个 `HAB`，因此当前公开配置总计有 `18` 个 hybrid attention blocks。

### 5. 跨分支注意力

`WindowAttention` 是 BP-SR 的核心交互模块。

它为两路分支分别定义了独立投影：

- 原图分支使用 `qkv`
- diffusion 分支使用 `qkv_diff`

在每个局部窗口内，两路不是各自 self-attention，而是显式跨流交互：

- `attn_ori = q_ori @ k_diff^T`
- `attn_diff = q_diff @ k_ori^T`
- `x_ori = attn_ori @ v_diff`
- `x_diff = attn_diff @ v_ori`

同时仍然保留 relative position bias 和 shifted-window mask，因此整体仍然保持 HAT 风格的局部窗口归纳偏置，只是把单流注意力改成了双流 cross interaction。

### 6. 局部分支与 block 组成

每个 `HAB` 内部还实例化了一个 `CAB` 局部卷积分支，用于通道感知的局部特征建模。

从当前公开 forward 路径看，block 的残差更新主要由以下两部分主导：

- cross-window attention
- MLP refinement

`CAB` 分支会被实际计算，但当前发布实现中的最终残差更新保留的是 attention 主路径。这一描述严格对应当前公开代码，而不是理想化的结构图版本。

### 7. 分支融合与输出重建

深层特征提取完成后：

- 先把原图分支和 diffusion 分支重新拆开
- 再沿通道维拼接
- 使用 `1x1` 的 `fusion_conv` 将 `2C -> C`
- 通过 `conv_last` 映射回 RGB

最终输出是建立在 diffusion 先验上的残差修正：

`output = reconstruction + x_diff`

这一点很关键。BP-SR 不是从零独立合成最终图像，而是学习对 FaithDiff 结果进行可控修正。

## 官方数据集与归档比赛工作区的关系

根目录 README 说明的是整理后的正式方法命名与代码结构，而 `BP-code/` 保留了比赛期间真实运行的工作区组织方式。

在归档工作区中，可以把数据角色理解为：

- synthetic paired 训练子集
  - `BP-code/datasets/HR`
  - `BP-code/datasets/LR`
  - `BP-code/datasets/LR_Diff`
  - 共 `1440` 组对齐 triplet，实际用于后处理监督训练

- 本地训练监控子集
  - `BP-code/datasets/vali_HR`
  - `BP-code/datasets/vali_LR`
  - `BP-code/datasets/vali_Diff`
  - 共 `7` 组样本，用于比赛期间的快速验证监控

- 开发阶段预处理目录
  - `FaithDiff-main/LR_SR` 与 `FaithDiff-main/LR_SR_enhance`
  - `FaithDiff-main/wild_val` 与 `FaithDiff-main/wild_val_enhance`
  - 这些目录对应开发阶段测试时使用的 synthetic / wild 预处理结果

- 最终测试与提交目录
  - `BP-code/datasets/DRCT_syn_test`
  - `BP-code/datasets/DRCT_syn_test-diff`
  - `BP-code/datasets/wild_test`
  - `BP-code/datasets/wild_test-diff`
  - 这些目录对应最终 synthetic / wild 推理与提交所使用的输入和先验

这一归档结构也解释了为什么比赛时期的命名会显得混乱：

- 训练目录里的 `LR` 实际已经是经过 DRCT 处理后的同分辨率输入
- `FaithDiff-main` 中的 `wild_val` 是历史命名，不应被机械地理解成根仓库语义上的“验证集定义”
- 官方 benchmark 的完整划分，与本地保留下来的比赛工作区目录，并不是一一同构的

## 实际代码结构

本 README 对应仓库根目录当前真实结构。

```text
BP-SR/
├── baseline/
│   ├── archs/bpsr_dualstream_cross_attention_arch.py
│   ├── data/bpsr_aligned_triplet_dataset.py
│   ├── models/bpsr_inference_model.py
│   └── models/bpsr_refinement_model.py
├── options/
│   ├── train/train_bpsr_dualstream_refinement_ntire.yml
│   └── test/test_bpsr_dualstream_refinement_ntire.yml
├── FaithDiff-main/
│   └── FaithDiff 预处理代码
├── BP-code/
│   └── 比赛期间的归档工作区
├── inference.py
└── README_origin.md
```

核心实现文件：

- `baseline/data/bpsr_aligned_triplet_dataset.py`
  - 正式数据集名称 `BPSRAlignedTripletDataset`
  - 负责对齐加载 `lq / diff / gt`
- `baseline/models/bpsr_refinement_model.py`
  - 正式训练包装名称 `BPSRRefinementModel`
- `baseline/models/bpsr_inference_model.py`
  - 正式验证与测试包装名称 `BPSRInferenceModel`
- `baseline/archs/bpsr_dualstream_cross_attention_arch.py`
  - 正式双输入主干名称 `BPSR_DualStreamCrossAttention`
- `options/train/train_bpsr_dualstream_refinement_ntire.yml`
  - 训练配置，其中 `scale: 1`，`model_type: BPSRRefinementModel`，`network_g: BPSR_DualStreamCrossAttention`

兼容别名：

- `NtireImageDataset`
- `PPV5Model`
- `HATModel`
- `PostProcess_V3`

`BP-code/` 保留为比赛期间的归档工作区，根目录的 `baseline/` 和 `options/` 才是此 README 所对应的主要代码路径。

## wild 子集在训练中的角色

由于官方 wild 子集没有 paired GT，公开的 BP-SR refinement 阶段并不会在 wild 图像上直接优化 supervised pixel loss。

公开流程对应的设计是：

- `DRCT` 先解决 synthetic 的 `4x` 分辨率失配
- `FaithDiff` 为 synthetic 和 wild 都生成同分辨率感知先验
- 最终 BP-SR refinement 网络只在对齐的 synthetic triplet 上监督训练，然后泛化到两个域

这种设计和 benchmark 的结构是一致的：

- synthetic 负责提供稳定监督
- wild 负责提供真实域先验与最终评测目标
- 方法必须在没有 paired wild GT 的前提下，仍然能够处理真实 UGC 输入

## 训练配置细节

根目录公开训练配置为：

- `options/train/train_bpsr_dualstream_refinement_ntire.yml`

当前公开配置中的关键参数包括：

- `model_type: BPSRRefinementModel`
- `network_g: BPSR_DualStreamCrossAttention`
- `scale: 1`
- `gt_size: 192`
- `batch_size_per_gpu: 2`
- `num_worker_per_gpu: 8`
- `optimizer: Adam`
- `lr: 2e-4`
- `betas: [0.9, 0.99]`
- `ema_decay: 0.999`
- `total_iter: 100000`

同时还启用了：

- `find_unused_parameters: true`
- tile 验证逻辑
- 验证阶段保存图像

## 训练目标

BP-SR 公开配置中使用的是单一复合目标 `CombinedLoss`，在训练包装器中的调用形式为：

`self.cri_pix(self.gt, self.output)`

项目中实际使用的损失为：

`total = 1.00 * SmoothL1 + 0.06 * Perceptual + 0.05 * Histogram + 0.50 * MS-SSIM + 0.0083 * PSNRLoss + 0.25 * Color`

其中：

- `SmoothL1` 用于稳定像素级回归
- `Perceptual` 由 `VGGPerceptualLoss` 计算
- `Histogram` 用于约束整体色调 / 分布一致性
- `MS-SSIM` 用于强化结构相似性
- `PSNRLoss` 用于保留保真约束
- `ColorLoss` 用于抑制增强过程中的颜色漂移

这一点很关键。BP-SR 并不是只用简单的 `L1` 或 `L2` 做重建，而是从训练目标层面显式平衡：

- 像素保真
- 感知相似性
- 结构保持
- 色调与颜色一致性

这与 KwaiSR 的赛题设定是吻合的，因为最终成绩本来就不是由单一重建指标决定的。

训练过程中的验证指标为：

- `PSNR`
- `LPIPS`

这与 BP-SR “同时兼顾保真和感知质量”的设计目标是一致的。

## 推理细节

根目录公开测试配置为：

- `options/test/test_bpsr_dualstream_refinement_ntire.yml`

当前推理配置中的关键点包括：

- `model_type: BPSRInferenceModel`
- `param_key_g: params_ema`
- `tile_size: 1024`
- `tile_pad: 32`
- `window_size: 16`
- `save_img: true`

推理流程为：

1. 对 `lq` 和 `diff` 做窗口对齐的 reflect padding
2. 按 tile 切分进行显存友好的前向
3. 将各 tile 输出重新拼回整图
4. 去掉 pad 区域
5. 保存最终结果

当 `val.suffix` 为空时，保存文件名规则为：

`<img_name>_<experiment_name>.png`

这也解释了为什么归档工作区中的提交结果会带有如下后缀：

- `..._testSyn_PostProcessV5_NTIRE.png`
- `..._testWild_PostProcessV5_NTIRE.png`

## 为什么这套设计适合 KwaiSR

BP-SR 并不是把一个通用 restoration 模型简单套到比赛数据上，而是围绕 KwaiSR 的数据结构专门组织出来的。

synthetic 和 wild 本身对应两种不同困难：

- synthetic 更强调从已知退化中恢复出可信内容
- wild 更强调在没有 paired GT 的前提下做真实感知增强

BP-SR 最终能成立，依赖于三个相互配合的设计：

1. `FaithDiff` 先生成感知先验
   - 这样最终阶段不必只靠 `lq` 自己去“猜”高频细节。

2. `BPSR_DualStreamCrossAttention` 不会直接照单全收 diffusion 输出
   - 它强制原图分支和 diffusion 分支在局部窗口内相互查询、相互校正。
   - 因此最终结果比“直接拿 diffusion 结果当输出”更可控。

3. 网络学习的是“对 diffusion 先验的残差修正”
   - 这让最终阶段更聚焦于纠偏、去伪影、补结构，而不是从头生成一张全新的图。
   - 对同分辨率 refinement 来说，这是一个很务实的设计。

损失函数和结构也是配套的：

- `MS-SSIM` 与 `SmoothL1` 保证结构和内容稳定
- `Perceptual` 与 `Histogram` 促进更自然的纹理与色调分布
- `ColorLoss` 抑制激进增强导致的颜色偏移
- `PSNRLoss` 让优化过程始终保留保真约束

综合来看，这个项目可以被理解成一个“可控的感知增强系统”：

- diffusion 提供候选细节
- cross-stream attention 负责筛选和对齐这些细节
- residual 输出负责修正先验
- composite loss 则约束结果在视觉观感和保真之间取得平衡

## 模型细节

### 三路对齐数据流

`BPSRAlignedTripletDataset` 每个样本会读取三张对齐图像：

- `lq`
- `diff`
- `gt`

训练时三者会进行同步裁剪和数据增强，保证模型训练在严格对齐的 triplet 上，而不是松散匹配的样本上。

### 双输入训练与推理

训练包装和测试包装都会在送入 generator 前，将 `lq` 和 `diff` 显式堆叠：

- 训练阶段：`baseline/models/bpsr_refinement_model.py`
- 验证/测试阶段：`baseline/models/bpsr_inference_model.py`

这说明 BP-SR 不是单图像 post-filter，而是真正的双分支 refinement pipeline。

### 跨分支窗口注意力

核心融合逻辑位于 `baseline/archs/bpsr_dualstream_cross_attention_arch.py`。

其中 `WindowAttention` 为两路分支分别定义了：

- 原图分支的 `qkv`
- diffusion 分支的 `qkv_diff`

随后执行跨分支注意力交互：

- `attn_ori = q_ori @ k_diff^T`
- `attn_diff = q_diff @ k_ori^T`

并分别利用对方分支的信息进行重建：

- `x_ori <- attn_ori @ v_diff`
- `x_diff <- attn_diff @ v_ori`

这一点是 BP-SR 的关键。模型不是简单拼接两张图像，而是在局部窗口内显式地让原图和 diffusion 先验相互查询、相互修正。

### 为什么最终阶段使用 `scale = 1`

最终 refinement 模型配置为 `scale: 1` / `upscale: 1`。

这是刻意设计的：

- `DRCT` 负责 synthetic 数据的 4x 上采样
- `FaithDiff` 提供同分辨率增强先验
- `BPSR_DualStreamCrossAttention` 只聚焦在同分辨率后处理细化

因此最终阶段的容量主要用于：

- 伪影抑制
- 感知细节筛选
- 保结构的视觉增强

而不是大尺度几何上采样。

## 官方结果

根据官方 Track 2 结果表，BP-SR 的公开成绩为：

- `Objective Score = 46.4382`
- `PSNR = 27.2520`
- `SSIM = 0.7744`
- `LPIPS = 0.2257`
- `MUSIQ = 58.9467`
- `ManIQA = 0.3387`
- `CLIPIQA = 0.4418`
- `Objective Rank = 7`

这一结果更适合被理解为：BP-SR 在真实 UGC 图像增强 benchmark 上取得了较有竞争力的多目标综合表现。

其中 `LPIPS = 0.2257` 也反映出这套流程在感知质量上的明显导向，但最终排名仍由 fidelity、perceptual 和 no-reference IQA 指标共同决定，而不是单看 LPIPS。

## 安装

先安装 PyTorch，然后执行：

```bash
pip install -r requirements.txt
python setup.py develop
```

仓库中列出的依赖包括：

- `torch>=1.7`
- `basicsr==1.3.4.9`
- `einops`

## 预处理数据

- `DRCT` 用于将原始 synthetic 数据从 `270x480` 上采样到 `1080x1920`
- `FaithDiff` 用于在相同目标分辨率上预处理 synthetic 和 wild 数据

## 训练

公开训练配置使用 `2 x A6000`。

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=4321 \
  baseline/train.py \
  -opt options/train/train_bpsr_dualstream_refinement_ntire.yml \
  --launcher pytorch
```

## 测试

公开测试配置使用 `1 x A6000`。

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/test.py \
  -opt options/test/test_bpsr_dualstream_refinement_ntire.yml
```

## 权重与可视化结果

- 可视化结果: https://drive.google.com/drive/folders/1cbT7aaKb5FCxvlnaDIMWhgYphJg9h822?usp=drive_link
- 预训练权重: https://drive.google.com/file/d/1N0C01YGVzEFclERoiORE1QSF60QAHgLI/view?usp=drive_link

## 外部组件

- FaithDiff: https://github.com/jychen9811/FaithDiff
- DRCT: https://github.com/ming053l/DRCT
