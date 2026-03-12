import torch.cuda
import argparse
from PIL import Image
from llava.llm_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH, SDXL_PATH, FAITHDIFF_PATH, VAE_FP16_PATH
import os
from torch.nn.functional import interpolate
import numpy as np
import cv2
import json


# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
args = parser.parse_args()
print(args)
use_llava = not args.no_llava


# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device='cuda', load_8bit=args.load_8bit_llava, load_4bit=False)
else:
    llava_agent = None



os.makedirs(args.save_dir, exist_ok=True)
import json
for file_name in sorted(os.listdir(args.img_dir)):
    img_name = file_name.split('.')[0]

    image = Image.open(os.path.join(args.img_dir,file_name)).convert('RGB')


    # step 1: LLaVA
    if use_llava:
        captions = llava_agent.gen_image_caption([image])
    else:
        captions = ['']

    data = {
        "caption": captions[0]
    }
    json_name = img_name+'.json'
    file_path = os.path.join(
        args.save_dir, json_name)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
