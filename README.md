# BP-SR

[Python 3.8]
[PyTorch 2.4]

The *official* implementation for the [BP-SR].

### Installation

Install Pytorch first. Then,
``` bash
pip install -r requirements.txt 
python setup.py develop
```

### Pre-Processed Datasets
1. the [DRCT](https://github.com/ming053l/DRCT) model is utilized to upsample the resolution of original synthetic datasets from 270×480 to 1080×1920.

2. the [FaithDiff](https://github.com/JyChen9811/FaithDiff) model is utilized to preprocess the synthetic and wild datasets at the same resolution.

### Train
We utilize 2 A6000 GPUs for training.
``` bash
cd BP-SR
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port=4321 baseline/train.py -opt options/train/train_PPV5_SRx1_finetune_from_NTIRE.yml --launcher pytorch

```

### Test
We utilize 1 A6000 GPU for testing.
Test the trained model with best performance by
```bash
cd BP-SR
CUDA_VISIBLE_DEVICES=0 python baseline/test.py -opt options/test/Test_PostProcessV5_SRx1_NTIRE.yml
```

### Results
All visual results of BP-SR can be downloaded [here](https://drive.google.com/drive/folders/1cbT7aaKb5FCxvlnaDIMWhgYphJg9h822?usp=drive_link).