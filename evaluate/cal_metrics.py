import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os 
import numpy as np
import torch
import lpips
loss_fn = lpips.LPIPS(net='alex')


def remove_suffix(text, suffix):
    return text[:-len(suffix)] if text.endswith(suffix) else text

def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Normalize to [0, 1]
    img1_rgb /= 255.0
    img2_rgb /= 255.0

    # L2 (MSE)
    mse = np.mean((img1_rgb - img2_rgb) ** 2)

    # SSIM
    ssim_val, _ = ssim(img1_rgb, img2_rgb, data_range=1.0, channel_axis=2, full=True)

    # LPIPS (requires tensor and normalization)
    img1_tensor = torch.tensor(img1_rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    img2_tensor = torch.tensor(img2_rgb).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    lpips_val = loss_fn(img1_tensor, img2_tensor).item()

    return mse, ssim_val, lpips_val


def cal_content_loss():
    content_dir = './output/content/'
    reversed_content_dir = './output/reverse/'
    content_names = [f for f in os.listdir(content_dir)]
    reversed_content_names = [f for f in os.listdir(reversed_content_dir)]
    lpips_list = []
    ssim_list = []
    l2_list = []
    for content_name in content_names:
        for reversed_content_name in reversed_content_names:
            if remove_suffix(content_name, '.png') == remove_suffix(reversed_content_name, '.png') .split('__')[0]:
                l2_value, ssim_value, lpips_value  = compare_images(os.path.join(content_dir, content_name),
                                                       os.path.join(reversed_content_dir, reversed_content_name))
                lpips_list.append(lpips_value)
                ssim_list.append(ssim_value)
                l2_list.append(l2_value)
    print(len(ssim_list))
    print('Decoded Content Loss')
    print('Average LPIPS:', sum(lpips_list) / len(lpips_list))
    print('Average SSIM:', sum(ssim_list) / len(ssim_list))
    print('Average L2:', sum(l2_list) / len(l2_list))

def cal_stega_loss():
    regular_dir = './output/regular/'
    stega_regular_dir = './output/stega_regular/'
    regular_names = [f for f in os.listdir(regular_dir)]
    stega_regular_names = [f for f in os.listdir(stega_regular_dir)]
    lpips_list = []
    ssim_list = []
    l2_list = []
    for regular_name in regular_names:
        for stega_regular_name in stega_regular_names:
            if remove_suffix(regular_name, '.png')  == remove_suffix(stega_regular_name, '.png'):
                l2_value, ssim_value, lpips_value = compare_images(os.path.join(regular_dir, regular_name),
                                                       os.path.join(stega_regular_dir, stega_regular_name))
                lpips_list.append(lpips_value)
                ssim_list.append(ssim_value)
                l2_list.append(l2_value)
    print(len(ssim_list))
    print('Stega Loss')
    print('Average LPIPS:', sum(lpips_list) / len(lpips_list))
    print('Average SSIM:', sum(ssim_list) / len(ssim_list))
    print('Average L2:', sum(l2_list) / len(l2_list))

def cal_serial_loss():
    serial_dir = './output/serial/'
    regular_dir = './output/regular/'
    serial_names = [f for f in os.listdir(serial_dir)]
    regular_names = [f for f in os.listdir(regular_dir)]
    lpips_list = []
    ssim_list = []
    l2_list = []
    for serial_name in serial_names:
        for regular_name in regular_names:
            ss_serial_name = remove_suffix(serial_name, '.png').split('__')
            ss_regular_name = remove_suffix(regular_name, '.png').split('__')
            if ss_serial_name[0] + ss_serial_name[-1] == ss_regular_name[0] + ss_regular_name[-1]:
                l2_value, ssim_value, lpips_value = compare_images(os.path.join(serial_dir, serial_name),
                                                       os.path.join(regular_dir, regular_name))
                lpips_list.append(lpips_value)
                ssim_list.append(ssim_value)
                l2_list.append(l2_value)
    print(len(ssim_list))
    print('Serial Loss')
    print('Average LPIPS:', sum(lpips_list) / len(lpips_list))
    print('Average SSIM:', sum(ssim_list) / len(ssim_list))
    print('Average L2:', sum(l2_list) / len(l2_list))

if __name__ == '__main__':
    cal_content_loss()
    cal_stega_loss()
    cal_serial_loss()