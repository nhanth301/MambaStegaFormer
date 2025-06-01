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
        mamba_path = '/home/nhan/Downloads/mamba_iter_65000.pth'
        embedding_path = '../checkpoints/embedding_iter_160000.pth'
        decoder_path = '../checkpoints/decoder_iter_160000.pth'
        vgg = '../checkpoints/vgg_normalised.pth'
        d_state = 16
        img_size = 512
        use_pos_embed = True
        rnd_style = True
    model = load_pretrained(Args()).to(device).eval()
    return model, Args()

def load_steganography_models():
    """Load the hiding and reveal networks"""
    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid).to(device)
    try:
        hnet_checkpoint = torch.load("../checkpoints/netH_epoch_121,sumloss=0.000686,Hloss=0.000388.pth", 
                                   map_location=device)
        Hnet.load_state_dict(hnet_checkpoint)
    except FileNotFoundError:
        print("Hiding network checkpoint not found. Using uninitialized model.")
    Hnet.eval()

    Rnet = RevealNet(output_function=nn.Sigmoid).to(device)
    try:
        rnet_checkpoint = torch.load("../checkpoints/netR_epoch_121,sumloss=0.000686,Rloss=0.000397.pth", 
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



if __name__ == "__main__":
    cnt_dir = "Content"
    sty_dir = "Style"
    output_dir = "serial_output"
    st_model, _ = load_style_transfer_model()
    Hnet, Rnet = load_steganography_models()
    tf = test_transform(512, True)
    content_names = [f for f in os.listdir(cnt_dir)]
    style_names = [f for f in os.listdir(sty_dir)]
    content_names.sort()
    style_names.sort()
    style_name = ['contrast_of_forms.jpg', 'garden.jpg', 'sketch.png', 'starry.jpg', 'impronte_d_artista.jpg']
    print(style_names)
    # Check if output directory exists, if not create it
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for name in content_names:
        content_path = os.path.join(cnt_dir, name)
        content_image = tf(Image.open(content_path).convert('RGB'))
        # fix_content_image = content_image.clone()
        # fix_content_image = fix_content_image.unsqueeze(0).to(device)
        for style_name in style_names:
            style_path = os.path.join(sty_dir, style_name)
            style_image = tf(Image.open(style_path).convert('RGB'))
            # Apply style transfer
            with torch.no_grad():
                content_image = content_image.unsqueeze(0).to(device)
                style_image = style_image.unsqueeze(0).to(device)
                stylized_image = st_model(content_image, style_image)[0]
                # stylized_image = stylized_image.squeeze(0).cpu()
            # Save stylized image
            path = os.path.join(output_dir, style_name[:-4])
            if not os.path.isdir(path):
                os.mkdir(path)
            # stylized_image_path = os.path.join(path, name)
            # save_image(stylized_image, stylized_image_path)
            # Prepare for steganography
            # content_image = stylized_image.clone()
            # content_image = content_image.unsqueeze(0).to(device)
            # Create input for hiding network
            input_image = torch.cat((stylized_image, content_image), dim=1)
            # Hide the stylized image
            with torch.no_grad():
                hidden_image = Hnet(input_image)
                hidden_image = hidden_image.squeeze(0).cpu()
            # Save hidden image
            hidden_image_path = os.path.join(path, name[:-4] + '.png')
            save_image(hidden_image, hidden_image_path)
            print(f"Hidden image saved to {hidden_image_path}")
            # Load hidden image
            hidden_image = tf(Image.open(hidden_image_path).convert('RGB'))
            #remove hidden image 
            # os.remove(hidden_image_path)
            hidden_image = hidden_image.unsqueeze(0).to(device)
            # Reveal the hidden image
            with torch.no_grad():
                revealed_image = Rnet(hidden_image)
                revealed_image = revealed_image.squeeze(0).cpu()
            content_image = revealed_image


