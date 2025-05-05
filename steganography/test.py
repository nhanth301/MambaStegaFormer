import torch
import torch.nn as nn
from torchvision import transforms, utils as vutils
from stega_models.HidingUNet import UnetGenerator
from stega_models.RevealNet import RevealNet
from PIL import Image
import os

# Load checkpoint
hnet_checkpoint = torch.load("./checkPoint/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth")
rnet_checkpoint = torch.load("./checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid).to(device)
    Hnet.load_state_dict(hnet_checkpoint)
    Hnet.eval()

    Rnet = RevealNet(output_function=nn.Sigmoid).to(device)
    Rnet.load_state_dict(rnet_checkpoint)
    Rnet.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load & preprocess cover and secret
    cover_img = transform(Image.open("./test/cover.jpg").convert("RGB")).unsqueeze(0).to(device)
    secret_img = transform(Image.open("./test/secret.jpg").convert("RGB")).unsqueeze(0).to(device)

    concat_img = torch.cat((cover_img, secret_img), dim=1)

    # Step 1: Hiding
    with torch.no_grad():
        container_img = Hnet(concat_img).clamp(0, 1)
        vutils.save_image(container_img.data, "./test/container.png", normalize=True)

    # Step 2: Load saved container from file
    loaded_container_img = transform(Image.open("./test/container.png").convert("RGB")).unsqueeze(0).to(device)

    # Step 3: Reveal
    with torch.no_grad():
        rev_secret_img = Rnet(loaded_container_img).clamp(0, 1)
        vutils.save_image(rev_secret_img.data, "./test/rev_secret.png", normalize=True)

    print("✅ Container image was saved, loaded again, and secret was revealed from it.")
