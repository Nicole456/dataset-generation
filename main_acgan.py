"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model.acgan import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10
from other.folder import ImageFolder

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--out-dir', default='./output/acgan/model/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--tmpf', default='./output/acgan/tmp',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

parser.add_argument('--train-time', type=int, default=1, help='The ID of the specified GPU')

args = parser.parse_args()
print(args)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


# specify the gpu id if using only 1 gpu
if args.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

try:
    out_path = args.out_dir
    os.makedirs(os.path.join(out_path, 'train_time:{}'.format(args.train_time)))
except OSError:
    pass

try:
    tmp_path = args.tmpf
    os.makedirs(os.path.join(tmp_path, 'train_time:{}'.format(args.train_time)))
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# datase t
if args.dataset == 'imagenet':
    # folder dataset
    dataset = ImageFolder(
        root=args.dataroot,
        transform=transforms.Compose([
            transforms.Scale(args.imageSize),
            transforms.CenterCrop(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        classes_idx=(10, 20)
    )
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(
        root=args.dataroot, download=True,
        transform=transforms.Compose([
            transforms.Scale(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
else:
    raise NotImplementedError("No such dataset {}".format(args.dataset))

assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers))

# some hyper parameters
ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
num_classes = int(args.num_classes)

# Define the generator and initialize the weights
if args.dataset == 'imagenet':
    netG = _netG(ngpu, nz)
else:
    netG = _netG_CIFAR10(ngpu, nz)
netG.apply(weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
# print(netG)

# Define the discriminator and initialize the weights
if args.dataset == 'imagenet':
    netD = _netD(ngpu, num_classes)
else:
    netD = _netD_CIFAR10(ngpu, num_classes)
netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))


def main():
    # loss functions
    dis_criterion = nn.BCELoss()
    aux_criterion = nn.NLLLoss()

    # tensor placeholders
    input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
    noise = torch.FloatTensor(args.batchSize, nz)
    eval_label = torch.IntTensor([np.random.randint(0, num_classes, args.batchSize)])
    eval_noise = torch.FloatTensor(args.batchSize, nz).normal_(0, 1)
    dis_label = torch.FloatTensor(args.batchSize)
    aux_label = torch.LongTensor(args.batchSize)
    real_label = 1
    fake_label = 0

    # if using cuda
    if args.cuda:
        netD.cuda()
        netG.cuda()
        dis_criterion.cuda()
        aux_criterion.cuda()
        input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
        noise, eval_noise = noise.cuda(), eval_noise.cuda()

    # define variables
    input = Variable(input)
    noise = Variable(noise)
    eval_noise = Variable(eval_noise)
    dis_label = Variable(dis_label)
    aux_label = Variable(aux_label)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    avg_loss_D = 0.0
    avg_loss_G = 0.0
    avg_loss_A = 0.0

    for epoch in range(args.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, label = data
            batch_size = real_cpu.size(0)
            if args.cuda:
                real_cpu = real_cpu.cuda()
            with torch.no_grad():
                input.resize_as_(real_cpu).copy_(real_cpu)
                dis_label.resize_(batch_size).fill_(real_label)
                aux_label.resize_(batch_size).copy_(label)
            dis_output, aux_output = netD(input)

            dis_errD_real = dis_criterion(dis_output, dis_label)
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = dis_errD_real + aux_errD_real
            errD_real.backward()
            D_x = dis_output.data.mean()

            # compute the current classification accuracy
            accuracy = compute_acc(aux_output, aux_label)

            # train with fake
            noise.normal_(0, 1)
            fake = netG(noise, label)
            dis_label.data.fill_(fake_label)
            dis_output, aux_output = netD(fake.detach())
            dis_errD_fake = dis_criterion(dis_output, dis_label)
            aux_errD_fake = aux_criterion(aux_output, aux_label)
            errD_fake = dis_errD_fake + aux_errD_fake
            errD_fake.backward()
            D_G_z1 = dis_output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            dis_label.data.fill_(real_label)  # fake labels are real for generator cost
            dis_output, aux_output = netD(fake)
            dis_errG = dis_criterion(dis_output, dis_label)
            aux_errG = aux_criterion(aux_output, aux_label)
            errG = dis_errG + aux_errG
            errG.backward()
            D_G_z2 = dis_output.data.mean()
            optimizerG.step()

            # compute the average loss
            curr_iter = epoch * len(dataloader) + i
            all_loss_G = avg_loss_G * curr_iter
            all_loss_D = avg_loss_D * curr_iter
            all_loss_A = avg_loss_A * curr_iter
            all_loss_G += errG.item()
            all_loss_D += errD.item()
            all_loss_A += accuracy
            avg_loss_G = all_loss_G / (curr_iter + 1)
            avg_loss_D = all_loss_D / (curr_iter + 1)
            avg_loss_A = all_loss_A / (curr_iter + 1)

            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
                    % (epoch, args.niter, i, len(dataloader),
                       errD.item(), avg_loss_D, errG.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
            if i % 100 == 0:
                vutils.save_image(
                    real_cpu, '%s/real_samples.png' % args.out_dir)
                fake = netG(eval_noise.cuda(), eval_label.cuda())
                vutils.save_image(
                    fake.data / 2 + 0.5,
                    '%s/train_time:%d/fake_samples_epoch_%03d.png' % (args.tmpf, args.train_time, epoch)
                )

    torch.save(netG.state_dict(), '%s/train_time:%d/netG_epoch_%d.pth' % (args.out_dir, args.train_time, args.niter))
    torch.save(netD.state_dict(), '%s/train_time:%d/netD_epoch_%d.pth' % (args.out_dir, args.train_time, args.niter))


if __name__ == '__main__':
    main()
