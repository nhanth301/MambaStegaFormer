from PIL import Image
import torch
import numpy as np
import os
import sys
import io
import torchvision.utils as vutils
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn as nn
import argparse
import time
# Set path for imports
sys.path.append(os.path.abspath('../mambast'))
sys.path.append(os.path.abspath('../steganography'))

# Import models
from util.utils import load_pretrained
from stega_models.HidingUNet import UnetGenerator
from stega_models.RevealNet import RevealNet

def remove_suffix(text, suffix):
    return text[:-len(suffix)] if text.endswith(suffix) else text

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_style_transfer_model():
    """Load the style transfer model"""
    class Args:
        mamba_path = '../checkpoints/mamba_iter_160000.pth'
        embedding_path = '../checkpoints/embedding_iter_160000.pth'
        decoder_path = '../checkpoints/decoder_iter_160000.pth'
        vgg = '../checkpoints/vgg_normalised.pth'
        d_state = 16
        img_size = 256
        use_pos_embed = False
        rnd_style = False
    model = load_pretrained(Args()).to(device).eval()
    return model, Args()

def load_steganography_models():
    """Load the hiding and reveal networks"""
    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid).to(device)
    try:
        hnet_checkpoint = torch.load("../checkpoints/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth", 
                                   map_location=device)
        Hnet.load_state_dict(hnet_checkpoint)
    except FileNotFoundError:
        print("Hiding network checkpoint not found. Using uninitialized model.")
    Hnet.eval()

    Rnet = RevealNet(output_function=nn.Sigmoid).to(device)
    try:
        rnet_checkpoint = torch.load("../checkpoints/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth", 
                                   map_location=device)
        Rnet.load_state_dict(rnet_checkpoint)
    except FileNotFoundError:
        print("Reveal network checkpoint not found. Using uninitialized model.")
    Rnet.eval()
    
    return Hnet, Rnet

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def check_output_subdir_exist(output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    subdir_names = ['content', 'style', 'regular', 'stega_regular', 'reverse', 'serial']
    for dir_name in subdir_names:
        subdir_path = os.path.join(output_dir,dir_name)
        if not os.path.isdir(subdir_path):
            os.mkdir(subdir_path)

def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image"""
    # Remove batch dimension and move to CPU
    img = tensor.squeeze(0).cpu().detach()
    # Convert to numpy and transpose to (H, W, C)
    img = img.numpy().transpose(1, 2, 0)
    # Clip to [0, 1] and convert to uint8
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def argumnet():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir', type=str, default='./input/content/',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='./input/style/',
                        help='Directory path to a batch of style images')

    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='Directory path for output images')

    parser.add_argument('--img_size', type=int, default=256,
                        help='New (minimum) size for the images, \
                        keeping the original size if set to 0')

    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = argumnet()
    tf = test_transform(args.img_size, True)
    check_output_subdir_exist(args.output_dir)
    st_model, _ = load_style_transfer_model()
    Hnet, Rnet = load_steganography_models()
    content_names = [f for f in os.listdir(args.content_dir)]
    style_names = [f for f in os.listdir(args.style_dir)]
    content_names = [f for f in os.listdir(args.content_dir)]
    style_names = [f for f in os.listdir(args.style_dir)]
    start_time = time.time()
    times = []
    for content_name in content_names:
        print(content_name)
        content_image = tf(Image.open(args.content_dir + content_name).convert('RGB')).unsqueeze(0)
        if not args.cpu:
            content_image = content_image.cuda()
        output = content_image.cpu()
        output_name = args.output_dir + 'content/' + remove_suffix(content_name, '.jpg') + '.png'
        save_image(output, output_name, normalize=True)
        for style_name in style_names:
            print(style_name)
            style_image = tf(Image.open(args.style_dir + style_name).convert('RGB')).unsqueeze(0)
            if not args.cpu:
                style_image = style_image.cuda()
            output = style_image.cpu()
            output_name = args.output_dir + 'style/' + remove_suffix(style_name, '.jpg') + '.png'
            save_image(output, output_name, normalize=True)
            # style transfer
            stime = time.time()
            stylized = st_model(content_image, style_image)[0].clamp(0, 1)
            etime = time.time()
            delta = etime - stime

            output = stylized.cpu()
            output_name = args.output_dir + 'regular/' + remove_suffix(content_name, '.jpg') + '__' + remove_suffix(style_name, '.jpg') + '.png'
            save_image(output, output_name, normalize=True)
            stime = time.time()
            container_image = Hnet(torch.cat((stylized, content_image.clone()), dim=1))
            etime = time.time()
            times.append(etime - stime + delta)
            output = container_image.cpu()
            output_name = args.output_dir + 'stega_regular/' + remove_suffix(content_name, '.jpg') + '__' + remove_suffix(style_name, '.jpg') + '.png'
            save_image(output, output_name, normalize=True)

            # reverse style transfer
            reveal_content_image = Rnet(container_image)
            output = reveal_content_image.cpu()
            output_name = args.output_dir + 'reverse/' + remove_suffix(content_name, '.jpg') + '__' + remove_suffix(style_name, '.jpg') + '.png'
            save_image(output, output_name, normalize=True)

            print(output_name)
            for serial_style_name in style_names:
                print(serial_style_name)
                serial_style_image = tf(Image.open(args.style_dir + serial_style_name).convert('RGB')).unsqueeze(0)
                if not args.cpu:
                    serial_style_image = serial_style_image.cuda()
                serial_stylized = st_model(reveal_content_image, serial_style_image)[0].clamp(0, 1)
                output = serial_stylized.cpu()
                output_name = args.output_dir + 'serial/' + remove_suffix(content_name, '.jpg') + '__' + remove_suffix(style_name, '.jpg') + '__' + remove_suffix(serial_style_name, '.jpg') + '.png'   
                save_image(output, output_name, normalize=True)
                print(output_name)
    end_time = time.time()
    print('Total time: ', end_time - start_time)
    print('Average time per image: ', sum(times) / len(times))
if __name__ == "__main__":
    with torch.no_grad():
        main()
