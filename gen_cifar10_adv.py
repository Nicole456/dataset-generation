#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import os
from robustness import model_utils, datasets
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchsize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--representation_size', type=int, default=2048,
                    help='the height / width of the input image to network')
parser.add_argument('--noise-scala', type=int, default=20, help='size of the latent z vector')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--arch', default='resnet50', help="model architecture")
parser.add_argument('--resume-path', default='.', help="trained model path")
parser.add_argument('--out-dir', default='./output/adv/gen_time:{}/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--gen-time', type=int, default=1, help='The ID of the specified GPU')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

DATA = 'CIFAR'  # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']

# Load dataset
dataset_function = getattr(datasets, DATA)
dataset = dataset_function(args.dataroot)
train_loader, test_loader = dataset.make_loaders(workers=args.workers,
                                                 batch_size=args.batchsize,
                                                 data_aug=False)
data_iterator = enumerate(test_loader)

# Load model

model_kwargs = {
    'arch': args.arch,
    'dataset': dataset,
    'resume_path': args.resume_path,
    'parallel': True
}
model, _ = model_utils.make_and_restore_model(**model_kwargs)
model.eval()

save_path = args.out_dir.format(args.gen_time)
if not os.path.exists(save_path):
    os.mkdir(save_path)
    for i in range(10):
        os.mkdir(os.path.join(save_path, str(i)))


# Custom loss for inversion
def inversion_loss(model, inp, targ):
    _, rep = model(inp, with_latent=True, fake_relu=True)
    loss = torch.div(torch.norm(rep - targ, dim=1), torch.norm(targ, dim=1))
    return loss, None


def save_robust_dataset(data, labels, cnt):
    dataset = TensorDataset(data, labels)
    dloader = DataLoader(dataset, batch_size=1)

    for i, (image, label) in enumerate(dloader):
        save_image(image,
                   os.path.join(save_path,
                                '{}/{}.png'.format(label.item(), cnt * 1000 + i)))


def main():
    # PGD parameters
    kwargs = {
        'custom_loss': inversion_loss,
        'constraint': '2',
        'eps': 0.5,
        'step_size': 0.1,
        'iterations': 1000,
        'do_tqdm': True,
        'targeted': True,
        'use_best': False
    }

    start_time = time.time()
    for i, (image, label) in enumerate(train_loader):
        inter_time = time.time()

        rand_data = torch.randn_like(image) / args.noise_scala + 0.5  # Seed for inversion (x_0)

        with torch.no_grad():
            (_, rep), _ = model(image.cuda(), with_latent=True)  # Corresponding representation

        _, xadv = model(rand_data.cuda(), rep.clone(), make_adv=True, **kwargs)  # Image inversion using PGD

        save_robust_dataset(xadv, label, i)

        # Measure the time
        if (i + 1) % 10 == 1:
            elapsed = time.time() - start_time
            elapsed_tracking = time.time() - inter_time
            print(
                f'Robustified {(i + 1) * args.batchsize} images in {elapsed:0.3f} seconds; '
                f'Took {elapsed_tracking:0.3f} seconds for this particular iteration')
            # break


if __name__ == '__main__':
    main()
