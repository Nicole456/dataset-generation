#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from model.acgan import _netG_CIFAR10
import os
import torchvision
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batchsize', type=int, default=10, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--out-dir', default='./output/acgan/gen_time:{}/',
                    help='folder to output images')
parser.add_argument('--gpu', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--gen-time', type=int, default=1, help='The ID of the specified GPU')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

save_path = args.out_dir.format(args.gen_time)

if not os.path.exists(save_path):
    os.mkdir(save_path)
    for i in range(10):
        os.mkdir(os.path.join(save_path, str(i)))


def main():
    noise = torch.FloatTensor(args.batchsize, args.nz)
    noise = noise.cuda()
    noise = Variable(noise)
    labels = torch.arange(0, 10, dtype=torch.long, device=device)

    generator = _netG_CIFAR10(args.ngpu, args.nz)
    generator.load_state_dict(torch.load(args.netG))
    generator.cuda()

    print('load generator successfully!')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        for i in range(10):
            os.makedirs(save_path + '{}/'.format(i))

    print('makedir successfully!')

    for k in range(5000):
        noise = noise.normal_(0, 1)
        images = generator(noise.cuda(), labels)
        images = images / 2 + 0.5
        for i in range(0, 10):
            torchvision.utils.save_image(images[i], os.path.join(save_path, str(i), str(k) + '.png'))


if __name__ == '__main__':
    main()
