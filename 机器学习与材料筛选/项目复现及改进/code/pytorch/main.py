from __future__ import print_function
import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vtuils
from torch.autograd import Variable
from dataset_ziyong import HDF5Dataset
from hdf5_io_ziyong import save_hdf5
import dcgan_ziyong

np.random.seed(43)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='3D')
parser.add_argument('--dataroot', default=r'D:\test_MistGPU\training_images', help='path to dataset')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to netwrok')
parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--niter', type=int, default=1000, help='number of epoches to train for')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate, default=0.0002, 1e-3~1e-5')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help='path to netG (to continue trainging)')
parser.add_argument('--netD', default='', help='path to netD (to continue trainging)')
parser.add_argument('--outf', default=r'D:\output\output', help='folder to output images and model checkpoints')

opt = parser.parse_args()
#opt = parser.parse_known_args()[0]
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

opt.manualSeed = 43
print('Random Seed:', opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and opt.cuda:
    print('WARNING: You have a CUDA device, so you should probably run with --cuda')

if opt.dataset in ['3D']:
    dataset = HDF5Dataset(opt.dataroot, input_transform=transforms.Compose([transforms.ToTensor]))

assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1

#权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0,0.02)
        m.bias.data.fill_(0)

netG = dcgan_ziyong.DCGAN3D_G(opt.imageSize, nz, nc, ngf, ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = dcgan_ziyong.DCGAN3D_D(opt.imageSize, nz, nc, ndf, ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input, noise, fixed_noise, fixed_noise_TI = None, None, None, None
input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize, opt.imageSize)
input = input.squeeze(-1)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1, 1)
fixed_noise = torch.FloatTensor(1, nz, 7, 7, 7).normal_(0, 1)
fixed_noise_TI = torch.FloatTensor(1, nz, 1, 1, 1).normal_(0, 1)

label = torch.FloatTensor(opt.batchSize)
real_label = 0.9
fake_label = 0

if not opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    fixed_noise_TI = fixed_noise_TI.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
fixed_noise_TI = Variable(fixed_noise_TI)

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

epoch = 0
gen_iterations = 0
for epoch in range(opt.niter + 1):
    for i, data in enumerate(dataloader, 0):
        f = open(r'D:\output\output\output.csv', "a")
        # (1) Update D network:maxisize log(D(x)) + log(1-D(G(z)))
        #train with real
        netD.zero_grad()

        real_cpu = data

        batch_size = real_cpu.size(0)
        input.resize_(real_cpu.size()).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)

        output = netD(input)
        output = output.squeeze(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        #train with fake
        noise.data.resize_(batch_size, nz, 1, 1, 1)
        noise.data.normal_(0, 1)
        fake = netG(noise).detach()
        label.data.fill_(fake_label)
        output = netD(fake)
        output = output.squeeze(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        # (2) Update G network: maxisize log(D(G(z)))

        g_iter = 1
        while g_iter != 0:
            netG.zero_grad()
            label.data.fill_(1.0)# fake labels are real for generator cost
            noise.data.normal_(0, 1)
            fake = netG(noise)
            output = netD(fake)
            output = output.squeeze(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
            g_iter -= 1

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        f.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, opt.niter, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        f.write('\n')
        f.close()


    if epoch % 2 == 0:
        fake = netG(fixed_noise)
        fake_TI = netG(fixed_noise_TI)
        save_hdf5(fake.data,r'D:\output\output\fake_samples_{0}.hdf5'.format(gen_iterations))
        save_hdf5(fake_TI.data, r'D:\output\output\fake_TI_{0}.hdf5'.format(gen_iterations))
        gen_iterations += 1

    if epoch % 50 == 0:
        torch.save(netG.state_dict(), r'D:\output\output\netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), r'D:\output\output\netD_epoch_%d.pth' % (epoch))

    epoch += 1
