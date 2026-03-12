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
