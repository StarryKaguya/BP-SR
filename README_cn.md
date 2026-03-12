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
