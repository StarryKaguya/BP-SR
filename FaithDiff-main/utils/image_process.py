from PIL import Image
import cv2
import numpy as np
def check_image_size(x, padder_size=8):
    # 获取图像的宽高
    width, height = x.size
    padder_size = padder_size
    # 计算需要填充的高度和宽度
    mod_pad_h = (padder_size - height % padder_size) % padder_size
    mod_pad_w = (padder_size - width % padder_size) % padder_size
    x_np = np.array(x)
    # 使用 ImageOps.expand 进行填充
    x_padded = cv2.copyMakeBorder(x_np, top=0, bottom=mod_pad_h, left=0, right=mod_pad_w, borderType=cv2.BORDER_REPLICATE)

    x = Image.fromarray(x_padded)
    # x = x.resize((width + mod_pad_w, height + mod_pad_h))
    
    return x, width, height, width + mod_pad_w, height + mod_pad_h