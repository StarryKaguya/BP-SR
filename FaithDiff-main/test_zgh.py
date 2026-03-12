import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_image_into_patches(image, patch_size):
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append((x, y, patch))  # 返回patch的坐标
    return patches

def calculate_patch_variance(patch):
    return np.var(patch)

def classify_patches(patches, threshold=100):
    detail_patches = []
    flat_patches = []
    patch_info = []  # 存储每个patch的类别和坐标
    
    for x, y, patch in patches:
        variance = calculate_patch_variance(patch)
        if variance > threshold:
            detail_patches.append((x, y, patch))
            patch_info.append((x, y, 'detail'))
        else:
            flat_patches.append((x, y, patch))
            patch_info.append((x, y, 'flat'))
    
    return patch_info, detail_patches, flat_patches

def create_mask(image, patch_info, patch_size):
    """
    根据patch分类的结果，创建一个掩膜图像，掩膜为半透明的
    :param image: 输入图像
    :param patch_info: 每个patch的分类信息
    :param patch_size: 每个patch的大小
    :return: 掩膜图像
    """
    mask = np.zeros_like(image)  # 创建一个空白的掩膜图像
    
    for x, y, label in patch_info:
        # 用不同颜色填充mask
        if label == 'detail':
            mask[y:y+patch_size, x:x+patch_size] = [0, 0, 255]  # 红色
        else:
            mask[y:y+patch_size, x:x+patch_size] = [0, 255, 0]  # 绿色
    
    return mask

def add_transparent_overlay(image, mask, alpha=0.5):
    """
    将透明掩膜叠加到原图上
    :param image: 原图
    :param mask: 掩膜
    :param alpha: 掩膜透明度（0-1之间，0为完全透明，1为不透明）
    :return: 叠加后的图像
    """
    return cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

def draw_patch_boundaries(image, patch_info, patch_size):
    """
    在每个patch的边缘画上分界线
    :param image: 原图
    :param patch_info: 每个patch的分类信息
    :param patch_size: 每个patch的大小
    :return: 添加分界线后的图像
    """
    for x, y, _ in patch_info:
        cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), (255, 255, 255), 1)  # 白色分界线
    return image

# 读取图像并转为灰度图
image = cv2.imread('/opt/data/private/gzy/FaithDiff-main/wild_val/134113096143_284_cg.png')
if image is None:
    raise FileNotFoundError("Image file not found.")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 设置patch大小
patch_size = 96
if gray_image.shape[0] < patch_size or gray_image.shape[1] < patch_size:
    raise ValueError("Image size must be larger than patch size.")

# 分割图像成patch
patches = split_image_into_patches(gray_image, patch_size)

# 分类patches为细节patch和平坦patch
patch_info, detail_patches, flat_patches = classify_patches(patches, threshold=100)

# 创建掩膜图像
mask_image = create_mask(image, patch_info, patch_size)

# 将掩膜叠加到原图上
overlay_image = add_transparent_overlay(image, mask_image, alpha=0.5)

# 在原图上绘制分界线
overlay_with_boundaries = draw_patch_boundaries(overlay_image.copy(), patch_info, patch_size)

# 保存最终图像为overlay_mask.png
cv2.imwrite('overlay_mask.png', overlay_with_boundaries)

# 可选：显示保存的掩膜图像
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(overlay_with_boundaries, cv2.COLOR_BGR2RGB))  # 转换颜色格式用于显示
plt.title('Overlay with Mask and Patch Boundaries')
plt.axis('off')
plt.show()
