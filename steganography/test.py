# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com
@version: 1.0
@file: test.py
@time: 2025/04/21
"""

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
import utils.transformed as transforms
from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet
import time

# Define paths and configurations
DATA_DIR = '/n/liyz/data/deep-steganography-dataset/'  # Adjust to your dataset path
PRETRAINED_HNET_PATH = "./checkPoint/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth"  # Path to pretrained Hnet
PRETRAINED_RNET_PATH = "./checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth"  # Path to pretrained Rnet

parser = argparse.ArgumentParser(description='Test script for deep steganography with pretrained models')
parser.add_argument('--test_dir', default='./test_data/', help='directory containing test images')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='image size for testing')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--testPics', default='./test_results/', help='folder to output test images')
parser.add_argument('--beta', type=float, default=0.75, help='hyper parameter of beta')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def print_log(log_info, log_path, console=True):
    if console:
        print(log_info)
    if not opt.debug:
        if not os.path.exists(log_path):
            with open(log_path, "w") as fp:
                fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')

def save_result_pic(this_batch_size, originalLabelv, ContainerImg, secretLabelv, RevSecImg, epoch, i, save_path):
    if not opt.debug:
        originalFrames = originalLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        containerFrames = ContainerImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        secretFrames = secretLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        revSecFrames = RevSecImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

        showContainer = torch.cat([originalFrames, containerFrames], 0)
        showReveal = torch.cat([secretFrames, revSecFrames], 0)
        resultImg = torch.cat([showContainer, showReveal], 0)
        resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(resultImg, resultImgName, nrow=this_batch_size, padding=1, normalize=True)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test(test_loader, epoch, Hnet, Rnet, criterion):
    print("#################################################### test begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    
    for i, data in enumerate(test_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data
        this_batch_size = int(all_pics.size()[0] / 2)

        cover_img = all_pics[0:this_batch_size, :, :, :]
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        concat_img = torch.cat([cover_img, secret_img], dim=1)

        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img, volatile=True)
        cover_imgv = Variable(cover_img, volatile=True)

        container_img = Hnet(concat_imgv)
        errH = criterion(container_img, cover_imgv)
        Hlosses.update(errH.data.item(), this_batch_size)

        rev_secret_img = Rnet(container_img)
        secret_imgv = Variable(secret_img, volatile=True)
        errR = criterion(rev_secret_img, secret_imgv)
        Rlosses.update(errR.data.item(), this_batch_size)
        save_result_pic(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i, opt.testPics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "test[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t test time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    print("#################################################### test end ########################################################")
    return val_hloss, val_rloss, val_sumloss

def main():
    global opt, logPath

    opt = parser.parse_args()
    logPath = './test_logs/test_log.txt'

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    cudnn.benchmark = True

    # Create output directories
    if not opt.debug:
        if not os.path.exists(opt.testPics):
            os.makedirs(opt.testPics)
        if not os.path.exists(os.path.dirname(logPath)):
            os.makedirs(os.path.dirname(logPath))

    # Load test dataset
    test_dataset = MyImageFolder(
        opt.test_dir,
        transforms.Compose([
            transforms.Resize([opt.imageSize, opt.imageSize]),
            transforms.ToTensor(),
        ]))
    assert test_dataset
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    # Initialize models
    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=torch.nn.Sigmoid)
    Rnet = RevealNet(output_function=torch.nn.Sigmoid)

    # Load pretrained weights
    Hnet.load_state_dict(torch.load(PRETRAINED_HNET_PATH))
    Rnet.load_state_dict(torch.load(PRETRAINED_RNET_PATH))

    # Move models to GPU if enabled
    if opt.cuda:
        Hnet.cuda()
        Rnet.cuda()
    
    # Apply weight initialization (optional, as pretrained weights are loaded)
    # Hnet.apply(weights_init)
    # Rnet.apply(weights_init)

    # Support multi-GPU if specified
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
        Rnet = torch.nn.DataParallel(Rnet).cuda()

    # Define loss criterion
    criterion = torch.nn.MSELoss().cuda() if opt.cuda else torch.nn.MSELoss()

    # Run test
    test(test_loader, 0, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
    print(f"Test completed. Results saved in {opt.testPics}")

if __name__ == '__main__':
    main()