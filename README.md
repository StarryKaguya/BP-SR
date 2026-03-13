# BP-SR

[English](README.md) | [简体中文](README_cn.md)

BP-SR is a two-stage solution for `NTIRE 2025 / CVPR 2025 Workshops, Track 2: KwaiSR`, under the broader `NTIRE 2025 Challenge on Short-form UGC Video Quality Assessment and Enhancement`.

The released pipeline combines:

- `DRCT` for synthetic-data upsampling
- `FaithDiff` for same-resolution prior generation
- `BPSR_DualStreamCrossAttention` for dual-input post-processing refinement

Official references:

- Challenge page: https://codalab.lisn.upsaclay.fr/competitions/21346
- Project page: https://lixinustc.github.io/NTIRE2025-KVQE-KwaSR-KVQ.github.io/
- Challenge report: https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/A_Li_NTIRE_2025_Challenge_on_Short-form_UGC_Video_Quality_Assessment_and_CVPRW_2025_paper.pdf
- KwaiSR dataset paper: https://arxiv.org/html/2504.15003v1

## Overview

Track 2 mixes two different regimes:

- synthetic paired images that behave like `4x` SR
- wild UGC images that are already at target resolution but still visually degraded

According to the official KwaiSR description, the dataset contains `1800` synthetic paired images and `1900` wild low-quality images, with an `8:1:1` split. The development score is a composite objective:

`PSNR + 10*SSIM - 10*LPIPS + 0.1*MUSIQ + 10*ManIQA + 10*CLIPIQA`

This setting makes the task different from a standard synthetic-only super-resolution benchmark. The main challenge is to balance:

- structure fidelity
- perceptual quality
- robustness across synthetic and wild domains

## How To Interpret The Official Dataset

The official KwaiSR benchmark should not be interpreted as a conventional paired SR dataset.

- The `synthetic` subset is the paired source domain.
  - It provides aligned `LR / HR` supervision.
  - It behaves like a controllable `4x` restoration problem.

- The `wild` subset is the unpaired real target domain.
  - It contains only low-quality UGC images without paired GT.
  - It is intended to measure real-domain perceptual enhancement and generalization.

The challenge materials do not prescribe a single recipe for exploiting the wild subset. However, three signals are explicit:

- wild images do not come with paired GT
- the same method is expected to process both synthetic and wild inputs
- the competition page explicitly encourages diffusion-based methods

Therefore, the wild subset should be understood as unlabeled real-domain data rather than as a second supervised training set. In practice, it is suitable for:

- prior generation
- pseudo-target construction
- self-training or consistency regularization
- no-reference quality guidance
- domain adaptation to real UGC artifacts

The released BP-SR pipeline follows a conservative version of this logic:

- supervised refinement training is performed on the paired synthetic subset
- wild images are used for same-resolution prior generation and final inference
- one unified refinement pipeline is applied to both domains after preprocessing

## Motivation

BP-SR is designed around three practical issues in KwaiSR:

1. Synthetic and wild inputs do not start from the same spatial setting.
   - The original synthetic LR data are `270x480`.
   - The target resolution is `1080x1920`.
   - Wild images are already at `1080x1920`, but they contain realistic UGC artifacts.

2. Original inputs and diffusion outputs have complementary strengths.
   - The original `lq` image preserves structure and content anchors.
   - The diffusion-enhanced result provides stronger perceptual details, but may drift from the original signal.

3. The official score is multi-objective rather than PSNR-only.
   - Improving perceptual quality alone is insufficient.
   - Improving fidelity alone is also insufficient.

4. The wild subset cannot be used as ordinary paired supervision.
   - There is no paired GT for wild images.
   - The method must still generalize from synthetic supervision to real UGC inputs.

## Method

BP-SR is a two-stage pipeline.

### Stage A: Resolution unification and prior generation

- `DRCT` upsamples the original synthetic inputs from `270x480` to `1080x1920`.
- `FaithDiff` preprocesses both synthetic and wild images at the same `1080x1920` resolution.
- The FaithDiff output is stored as the `diff` branch and used as a perceptual prior.

This converts the mixed-input challenge into a unified same-resolution refinement setting.

### Stage B: Dual-input post-processing refinement

The final refinement model jointly takes:

- the original low-quality image `lq`
- the diffusion-enhanced prior `diff`

and learns a same-resolution refined output.

The root-level implementation uses:

- `baseline/data/bpsr_aligned_triplet_dataset.py` for aligned `lq / diff / gt` loading
- `baseline/models/bpsr_refinement_model.py` for the `BPSRRefinementModel` training wrapper
- `baseline/models/bpsr_inference_model.py` for validation and test
- `baseline/archs/bpsr_dualstream_cross_attention_arch.py` for the `BPSR_DualStreamCrossAttention` backbone

### Dual-input fusion

Both training and test wrappers explicitly stack `lq` and `diff` before forwarding them to the generator.

Inside `BPSR_DualStreamCrossAttention`, `WindowAttention` defines separate projections for the two branches and performs cross-stream interaction:

- original-stream queries attend to the diffusion branch
- diffusion-stream queries attend to the original branch

This allows the model to use diffusion priors as a controllable perceptual cue instead of directly trusting the diffusion output.

## Code-Level Pipeline

The released root-level code follows the tensor flow below.

### 1. Dataset construction

`BPSRAlignedTripletDataset` scans three folders with identical filenames and constructs aligned triplets:

- `lq_path`
- `diff_path`
- `gt_path`

At training time, the dataset performs:

- same-location random cropping on `lq`, `diff`, and `gt`
- synchronized flip / rotation augmentation
- optional RGB-to-Y conversion
- tensor conversion in `RGB / CHW / float32`

Because the released BP-SR configuration uses `scale: 1`, the crop size is the same for all three branches. In the provided training config, `gt_size: 192` therefore produces:

- `lq`: `3 x 192 x 192`
- `diff`: `3 x 192 x 192`
- `gt`: `3 x 192 x 192`

### 2. Wrapper-level input packing

`BPSRRefinementModel.optimize_parameters()` and `BPSRInferenceModel.pre_process()` both pack the input as:

`torch.stack([lq, diff], dim=1)`

which gives a generator input with shape:

`[B, 2, 3, H, W]`

During inference, both branches are reflect-padded to a multiple of `window_size` before stacking.

### 3. Shared shallow encoding

Inside `BPSR_DualStreamCrossAttention.forward()`:

- the dual input is mean-normalized
- the two branches are flattened from `[B, 2, 3, H, W]` to `[2B, 3, H, W]`
- a shared `3x3` convolution `conv_first` lifts both branches to `embed_dim`

This means the model uses a shared shallow encoder for both branches, while branch interaction happens later in attention space.

### 4. Deep feature extraction

The released backbone uses:

- `patch_size = 1`
- `embed_dim = 180`
- `depths = [6, 6, 6]`
- `num_heads = [6, 6, 6]`
- `window_size = 16`

With `patch_size = 1`, tokenization preserves the native spatial resolution. The network therefore behaves as a same-resolution transformer refiner rather than a low-resolution latent model.

Deep features are processed by `3` residual hybrid attention groups (`RHAG`), each containing `6` hybrid attention blocks (`HAB`), for a total of `18` attention blocks in the released configuration.

### 5. Cross-stream attention

`WindowAttention` is the core BP-SR interaction module.

It defines separate projections for the two branches:

- `qkv` for the original `lq` stream
- `qkv_diff` for the diffusion-prior stream

Inside each window, the two branches attend to each other rather than only to themselves:

- `attn_ori = q_ori @ k_diff^T`
- `attn_diff = q_diff @ k_ori^T`
- `x_ori = attn_ori @ v_diff`
- `x_diff = attn_diff @ v_ori`

Relative position bias and shifted-window masking are still used, so the architecture preserves the HAT-style local-window inductive bias while replacing single-stream attention with dual-stream cross interaction.

### 6. Local branch and block composition

Each `HAB` also instantiates a `CAB` local convolutional branch for channel-aware local processing.

In the released code path, the block residual update is dominated by:

- cross-window attention
- MLP refinement

The `CAB` branch is computed inside the block, but the residual fusion line currently keeps the attention path active in the final update. This matches the current released implementation rather than a hypothetical ablation variant.

### 7. Feature fusion and output reconstruction

After deep feature extraction:

- branch features are split back into original and diffusion halves
- the two branch feature maps are concatenated along channels
- a `1x1` fusion layer `fusion_conv` reduces `2C -> C`
- `conv_last` projects features back to RGB

The final output is formed as a residual over the original diffusion prior:

`output = reconstruction + x_diff`

This design is important. The network does not synthesize a fully independent image from scratch. Instead, it learns a residual correction on top of the FaithDiff-enhanced input.

## Official Dataset Versus Archived Competition Workspace

The root repository documents the cleaned method names and code structure. The archived `BP-code/` workspace preserves how the competition run was actually organized.

In the archived workspace, the data usage can be interpreted as follows:

- synthetic paired training subset
  - `BP-code/datasets/HR`
  - `BP-code/datasets/LR`
  - `BP-code/datasets/LR_Diff`
  - `1440` aligned triplets actually used for supervised refinement training

- local monitoring subset
  - `BP-code/datasets/vali_HR`
  - `BP-code/datasets/vali_LR`
  - `BP-code/datasets/vali_Diff`
  - `7` locally selected validation triplets used for quick training-time monitoring

- development-phase staging directories
  - `FaithDiff-main/LR_SR` and `FaithDiff-main/LR_SR_enhance`
  - `FaithDiff-main/wild_val` and `FaithDiff-main/wild_val_enhance`
  - these directories correspond to preprocessed synthetic and wild inputs used during development-stage testing

- final test-time directories
  - `BP-code/datasets/DRCT_syn_test`
  - `BP-code/datasets/DRCT_syn_test-diff`
  - `BP-code/datasets/wild_test`
  - `BP-code/datasets/wild_test-diff`
  - these directories correspond to the final synthetic and wild inference sets used for challenge submission

This archived layout also explains why the naming can look inconsistent:

- `LR` in the archived training folders already refers to DRCT-upsampled same-resolution inputs
- `wild_val` in the FaithDiff workspace is a historical competition-era name, not a repository-level semantic definition
- the full official benchmark split and the locally retained competition snapshot are not identical directory trees

### Why `scale = 1`

The final refinement model is configured with `scale: 1` and `upscale: 1`.

This is intentional:

- `DRCT` handles synthetic 4x upsampling
- `FaithDiff` provides same-resolution priors
- `BPSR_DualStreamCrossAttention` focuses on same-resolution refinement

As a result, the final stage can spend its capacity on:

- artifact suppression
- perceptual detail selection
- structure-preserving enhancement

instead of large-scale geometric upsampling.

## Official Result

According to the official Track 2 results table in the challenge report, BP-SR achieved:

- `Objective Score = 46.4382`
- `PSNR = 27.2520`
- `SSIM = 0.7744`
- `LPIPS = 0.2257`
- `MUSIQ = 58.9467`
- `ManIQA = 0.3387`
- `CLIPIQA = 0.4418`
- `Objective Rank = 7`

This result is best interpreted as a competitive multi-objective submission on a realistic UGC enhancement benchmark. The LPIPS value also reflects the perceptual emphasis of the pipeline, while the final rank is determined jointly by fidelity, perceptual, and no-reference IQA metrics.

## Repository Layout

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
│   └── FaithDiff preprocessing code
├── BP-code/
│   └── archived competition workspace
├── inference.py
└── README_origin.md
```

Notes:

- The formal method names in the root repository are:
  - `BPSRAlignedTripletDataset`
  - `BPSRRefinementModel`
  - `BPSRInferenceModel`
  - `BPSR_DualStreamCrossAttention`
- The legacy competition-era names are kept as compatibility aliases:
  - `NtireImageDataset`
  - `PPV5Model`
  - `HATModel`
  - `PostProcess_V3`
- `BP-code/` is kept as an archived workspace snapshot. The root-level `baseline/` and `options/` directories are the primary code paths documented here.

## Training Role Of The Wild Subset

Since the official wild subset has no paired GT, the released BP-SR refinement stage does not optimize a direct supervised pixel loss on wild images.

Instead, the repository reflects the following design:

- `DRCT` resolves the synthetic `4x` scale mismatch
- `FaithDiff` generates same-resolution perceptual priors for both synthetic and wild images
- the final BP-SR refinement model is trained on aligned synthetic triplets and then applied to both domains

This design choice matches the benchmark structure:

- synthetic data provide stable supervision
- wild data provide real-domain priors and evaluation targets
- the final method must remain usable on real UGC inputs without access to paired wild GT

## Training Configuration

The released root-level training config is:

- `options/train/train_bpsr_dualstream_refinement_ntire.yml`

Key settings in the released configuration are:

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

The released training config also enables:

- `find_unused_parameters: true`
- tiled validation logic
- image saving during validation

## Training Objective

The released BP-SR configuration uses a single composite objective through `CombinedLoss`, called in the wrapper as:

`self.cri_pix(self.gt, self.output)`

The actual loss used in the project is:

`total = 1.00 * SmoothL1 + 0.06 * Perceptual + 0.05 * Histogram + 0.50 * MS-SSIM + 0.0083 * PSNRLoss + 0.25 * Color`

where:

- `SmoothL1` stabilizes pixel-level regression
- `Perceptual` is computed by `VGGPerceptualLoss`
- `Histogram` encourages global tone / distribution consistency
- `MS-SSIM` enforces structural similarity
- `PSNRLoss` keeps fidelity pressure in the optimization
- `ColorLoss` reduces color drift across the restored image

This objective is important for understanding BP-SR. The project does not rely on a plain `L1` or `L2` reconstruction target. Instead, the training signal is explicitly designed to balance:

- pixel fidelity
- perceptual similarity
- structure preservation
- tone and color consistency

This is directly aligned with the challenge setting, where the final score is not determined by a single reconstruction metric.

During training-time validation, the released config tracks:

- `PSNR`
- `LPIPS`

This is consistent with the overall motivation of balancing fidelity and perceptual quality.

## Inference Details

The released root-level test config is:

- `options/test/test_bpsr_dualstream_refinement_ntire.yml`

Important inference details:

- `model_type: BPSRInferenceModel`
- `param_key_g: params_ema`
- `tile_size: 1024`
- `tile_pad: 32`
- `window_size: 16`
- `save_img: true`

Inference proceeds as:

1. reflect-pad `lq` and `diff` to window-aligned spatial sizes
2. optionally split the input into tiles for memory-safe processing
3. merge output tiles back to a full-resolution image
4. crop away padding
5. save the restored image

When `val.suffix` is unset, the saved filename follows:

`<img_name>_<experiment_name>.png`

This explains why archived competition outputs contain names such as:

- `..._testSyn_PostProcessV5_NTIRE.png`
- `..._testWild_PostProcessV5_NTIRE.png`

## Why This Design Works For KwaiSR

BP-SR is not merely a generic image-restoration model applied to a competition dataset. The released design is tightly coupled to the structure of KwaiSR.

The synthetic subset and the wild subset stress different failure modes:

- synthetic requires faithful recovery from a degraded but paired source
- wild requires perceptually convincing enhancement without paired GT

The final BP-SR design addresses this mismatch through three interacting choices:

1. `FaithDiff` generates a perceptual prior before the final stage.
   - This injects plausible high-frequency details that a conservative refiner would struggle to hallucinate from `lq` alone.

2. `BPSR_DualStreamCrossAttention` does not blindly trust the diffusion output.
   - It forces the original image branch and the diffusion branch to query each other inside local windows.
   - This makes the final output more controllable than directly using the diffusion result.

3. The network predicts a residual over the diffusion prior rather than a fully new image.
   - This keeps the final stage focused on correction, cleanup, and structural adjustment.
   - It is a pragmatic choice for same-resolution refinement.

The loss complements the architecture:

- `MS-SSIM` and `SmoothL1` preserve structure and content stability
- `Perceptual` and `Histogram` encourage visually plausible textures and tonal distribution
- `ColorLoss` suppresses color shifts introduced by aggressive enhancement
- `PSNRLoss` keeps the optimization anchored to fidelity

Taken together, the project can be understood as a controlled perceptual refinement system:

- diffusion contributes candidate details
- cross-stream attention filters and aligns those details
- the residual output corrects the prior
- the composite loss keeps the result visually strong without fully giving up fidelity

## Installation

Install PyTorch first, then:

```bash
pip install -r requirements.txt
python setup.py develop
```

Environment dependencies listed in this repository:

- `torch>=1.7`
- `basicsr==1.3.4.9`
- `einops`

## Pre-Processed Datasets

- `DRCT` is used to upsample the original synthetic datasets from `270x480` to `1080x1920`.
- `FaithDiff` is used to preprocess both synthetic and wild datasets at the same target resolution.

## Training

The released training setup uses `2 x A6000`.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=4321 \
  baseline/train.py \
  -opt options/train/train_bpsr_dualstream_refinement_ntire.yml \
  --launcher pytorch
```

## Testing

The released test setup uses `1 x A6000`.

```bash
CUDA_VISIBLE_DEVICES=0 python baseline/test.py \
  -opt options/test/test_bpsr_dualstream_refinement_ntire.yml
```

## Weights and Visual Results

- Visual results: https://drive.google.com/drive/folders/1cbT7aaKb5FCxvlnaDIMWhgYphJg9h822?usp=drive_link
- Pretrained weight: https://drive.google.com/file/d/1N0C01YGVzEFclERoiORE1QSF60QAHgLI/view?usp=drive_link

## External Components

- FaithDiff: https://github.com/jychen9811/FaithDiff
- DRCT: https://github.com/ming053l/DRCT
