# BP-SR 技术说明（NTIRE 2025 / KwaiSR）

## 1. 项目与赛题背景

- 赛题：**NTIRE 2025 Challenge on Short-form UGC Image Super-Resolution (4x)**（CodaLab 竞赛页）。
- 赛程：Development 阶段开始于 **2025-02-06 23:00 UTC**，比赛结束于 **2025-03-22 16:00 UTC**。
- 官方开发阶段评分公式（CodaLab 说明）：
  - `Final Score = PSNR + 10*SSIM - 10*LPIPS + 0.1*MUSIQ + 10*ManIQA + 10*CLIPIQA`
- 你参与的是 NTIRE 2025 的 **KwaiSR 赛道（Track 2）**，项目方案名为 **BP-SR**。

参考链接：
- https://codalab.lisn.upsaclay.fr/competitions/21346
- https://lixinustc.github.io/NTIRE2025-KVQE-KwaSR-KVQ.github.io/

---

## 2. 命名升级（已改代码）

为了让网络名直接体现创新点（Diffusion 引导 + 双流跨注意力），将原命名：
- `PostProcess_V3`

升级为：
- `BPSR_DualStreamCrossAttention`

并保留兼容别名：
- `PostProcess_V3`（防止旧权重/旧配置失效）

### 已修改文件

- `NTIRE2025/HAT/hat/archs/post_process_v3_arch.py`
  - 主类改名为 `BPSR_DualStreamCrossAttention`
  - 保留 `PostProcess_V3` 兼容别名
- `NTIRE2025/HAT/options/train/train_PPV5_SRx1_finetune_from_NTIRE.yml`
  - `network_g.type` 改为 `BPSR_DualStreamCrossAttention`
- `NTIRE2025/HAT/options/train/train_PPV3_SRx1_finetune_from_NTIRE.yml`
  - `network_g.type` 改为 `BPSR_DualStreamCrossAttention`
- `NTIRE2025/HAT/options/test/Test_PostProcessV5_SRx1_NTIRE.yml`
  - `network_g.type` 改为 `BPSR_DualStreamCrossAttention`
- `NTIRE2025/HAT/options/test/Test_PostProcessV3_SRx1_NTIRE.yml`
  - `network_g.type` 改为 `BPSR_DualStreamCrossAttention`

---

## 3. BP-SR 方法总览（Diffusion 双流）

BP-SR 的核心不是单网络“直接修复”，而是分成两个阶段：

1. **Diffusion 先验生成（FaithDiff）**
- 输入 LR 图像，利用 FaithDiff 生成增强先验图（本工程中对应 `diff` 分支图像）。
- 输出先验图写入 `LR_Diff` 或对应验证/测试 `*_Diff` 目录。

2. **双流跨注意力后处理（BPSR_DualStreamCrossAttention）**
- 将 `lq`（原始输入）与 `diff`（Diffusion 先验）拼成 5 维输入：`[N, 2, 3, H, W]`。
- 网络执行双流跨窗口注意力融合，输出恢复图像。

可理解为：

`LR(保真基线) + Diffusion Prior(细节先验) -> Cross-Attn Fusion -> 重建输出`

---

## 4. 你的核心创新（可直接写简历）

### 创新 1：双流输入范式（不是简单后处理）

- 数据层面明确三元组：`(lq, diff, gt)`。
- 训练/验证时同步读取并对齐裁剪 `lq` 与 `diff`，保证先验与原图空间一致。
- 模型输入显式做 `torch.stack([lq, diff], dim=1)`，让网络从结构上“感知两路信息来源”。

### 创新 2：双向跨流注意力（Cross-stream Window Attention）

在 `WindowAttention` 中，不是单流 self-attention，而是：

- 对原图流和先验流分别建立投影：`qkv` 与 `qkv_diff`
- 做**双向跨流匹配**：
  - `attn_ori = q_ori @ k_diff^T`
  - `attn_diff = q_diff @ k_ori^T`
- 取值时也跨流：
  - `x_ori <- attn_ori @ v_diff`
  - `x_diff <- attn_diff @ v_ori`

这使得网络不只是“拼接再卷积”，而是让两路特征在注意力域内相互校正。

### 创新 3：轻量融合 + 残差回注

- 深层输出将两路特征拼接后用 `1x1 Conv + ReLU` 融合（`fusion_conv`）。
- 最终输出采用残差形式回注到 `x_diff`（先验分支），增强稳定性并保留高频细节。

### 创新 4：面向竞赛落地的工程化细节

- 支持 tile 推理（避免高分辨率 OOM）
- 训练中启用 EMA（`ema_decay=0.999`）
- 使用分布式训练（日志记录为 2 卡）
- 训练与验证配置完全可复现（yml + log + checkpoints）

---

## 5. 训练与验证配置（来自实际代码/日志）

以 `train_PPV5_SRx1_finetune_from_NTIRE.yml` 与训练日志为准：

- Backbone：`BPSR_DualStreamCrossAttention`（原 `PostProcess_V3`）
- 参数量：**12,043,003 (~12.04M)**
- 输入尺度：`scale=1`（同分辨率质量增强范式）
- 深度配置：`depths=[6,6,6]`, `embed_dim=180`, `num_heads=[6,6,6]`, `window_size=16`
- patch：`gt_size=192`
- 优化器：Adam，`lr=2e-4`，`betas=(0.9, 0.99)`
- 迭代：`total_iter=100000`
- 损失：`CombinedLoss`
- 训练验证指标：PSNR + LPIPS（离线验证）

日志中记录：
- 训练图像数：`1440`
- 验证图像数：`7`
- best PSNR：`26.4780 @ 35000 iter`
- best LPIPS：`0.2739 @ 10000 iter`

---

## 6. 其他模型详细介绍（你方案中提到的）

### 6.1 FaithDiff（Diffusion 先验生成器）

你的工程实际调用方式：
- `FaithDiff-main/FaithDiff/create_FaithDiff_model.py`
  - 加载 SDXL UNet
  - 初始化 VAE encoder / information transformer / control embedding
  - 加载 FaithDiff 权重
  - 使用 DDPM scheduler
- `FaithDiff-main/test_wo_llava.py`
  - 典型推理参数：`num_inference_steps=20`, `guidance_scale=5`
  - 支持 `start_point`、tile VAE、color fix
  - 输出生成增强图供 BP-SR 的 `diff` 分支使用

FaithDiff 项目侧说明（其官方 README）：
- 定位是 CVPR 2025 图像超分/增强方向扩散先验方法
- 代码依赖 diffusers / SUPIR / TLC

### 6.2 DRCT（你提到的 DCRT，建议统一写 DRCT）

命名建议：
- 业界与官方仓库名称是 **DRCT**（不是 DCRT）。

官方 DRCT（CVPRW/NTIRE 2024）核心点：
- 主题是缓解 SwinIR 类网络中的信息瓶颈
- 强调将 dense connection 引入 Transformer SR 主干以稳定信息流

你当前仓库里的实际使用方式：
- 在测试配置中，`datasets/DRCT_syn_test` / `datasets/DRCT_syn_test-diff` 被用作合成测试数据目录命名
- 这表示你有使用“DRCT 相关数据/结果域”进行验证
- 但当前仓库未直接集成 DRCT 的训练/推理代码本体（是数据侧接入而非模型侧实现）

---

## 7. 可放简历的项目描述（长版）

`BP-SR：Diffusion先验引导的双流跨注意力图像增强框架（NTIRE 2025 KwaiSR）`

- 面向短视频 UGC 增强任务，提出并落地“FaithDiff 先验生成 + 双流跨注意力后处理”的两阶段框架。
- 设计 `BPSR_DualStreamCrossAttention` 主干，显式建模原图流与 diffusion 先验流，在窗口注意力中实现双向跨流 Q/K/V 交互，提升细节恢复与结构一致性。
- 构建三元组数据流水线 `(lq, diff, gt)`，并实现分布式训练、EMA、tile 推理等工程化能力，支撑高分辨率稳定推理。
- 在本地验证中达到 `PSNR=26.4780`、`LPIPS=0.2739`（best），模型参数约 `12.04M`。

---

## 8. 关键代码索引

- 双流模型封装：`NTIRE2025/HAT/hat/models/postprocess_v5_model.py`
- 双流主干：`NTIRE2025/HAT/hat/archs/post_process_v3_arch.py`
- 三元组数据集：`NTIRE2025/HAT/hat/data/ntire_image_dataset.py`
- 训练配置：`NTIRE2025/HAT/options/train/train_PPV5_SRx1_finetune_from_NTIRE.yml`
- 测试配置：`NTIRE2025/HAT/options/test/Test_PostProcessV5_SRx1_NTIRE.yml`
- FaithDiff 入口：`FaithDiff-main/FaithDiff/create_FaithDiff_model.py`

---

## 9. 外部参考

- NTIRE 2025 CodaLab 页面：
  - https://codalab.lisn.upsaclay.fr/competitions/21346
- NTIRE 2025 项目页：
  - https://lixinustc.github.io/NTIRE2025-KVQE-KwaSR-KVQ.github.io/
- FaithDiff：
  - https://github.com/jychen9811/FaithDiff
- DRCT：
  - https://github.com/ming053l/DRCT

