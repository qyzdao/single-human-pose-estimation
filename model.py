import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import utils
from utils import *
import matplotlib.cm as cm
from module import *

IMG_HEIGHT, IMG_WIDTH = 320,320

parser = argparse.ArgumentParser()
parser.add_argument("--lamda", type=int, default=0.02, help="coefficient for loss")
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of z")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--data_length", type=int, default=200, help="interation per epoch")
parser.add_argument("--grid", type=int, default=32, help="scene graph is a x by x graph")
parser.add_argument("--anchor", type=int, default=13, help="number of joints")
parser.add_argument("--start", type=int, default=0, help="joints start")
parser.add_argument("--end", type=int, default=13, help="joints end")
parser.add_argument("--art_size", type=int, default=320, help="the size of kids art")
parser.add_argument("--patch_size", type=int, default=128, help="the size of local patch around joint")
parser.add_argument("--exp", type=int, default=1, help="i-th experiment")
opt = parser.parse_args()
print(opt)

EVAL_ROOT = "eval%d" % opt.exp
IMAGE_ROOT = "images%d" % opt.exp
WEIGHT_ROOT = "weights%d" % opt.exp

os.makedirs(EVAL_ROOT, exist_ok=True)
os.makedirs(IMAGE_ROOT, exist_ok=True)
os.makedirs(WEIGHT_ROOT, exist_ok=True)

def seed_all(seed):
    if not seed:
        seed = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.en_block1 = nn.Sequential(
            #CoordConv(3, 3, with_r=False),
            ResiBlock(opt.channels, (8, 8, 32), True),        # 320
            ResiBlock(32, (8, 8, 32)),
        )
        #self.en_block1 = nn.DataParallel(self.en_block1)

        self.en_block2 = nn.Sequential(
            DownResiBlock(32, (16, 16, 64)),       # 160
            ResiBlock(64, (16, 16, 64)),
            ResiBlock(64, (16, 16, 64)),
        )
        #self.en_block2 = nn.DataParallel(self.en_block2)

        self.en_block3 = nn.Sequential(
            DownResiBlock(64, (32, 32, 128)),      # 80
            ResiBlock(128, (32, 32, 128)),
            ResiBlock(128, (32, 32, 128)),
        )
        #self.en_block3 = nn.DataParallel(self.en_block3)

        self.en_block4 = nn.Sequential(
            DownResiBlock(128, (64, 64, 256)),     # 40
            ResiBlock(256, (64, 64, 256)),
            ResiBlock(256, (64, 64, 256)),
        )
        #self.en_block4 = nn.DataParallel(self.en_block4)

        self.en_block5 = nn.Sequential(
            DownResiBlock(256, (64, 64, 256)),     # 20
            ResiBlock(256, (64, 64, 256)),
            ResiBlock(256, (64, 64, 256)),
        )
        #self.en_block5 = nn.DataParallel(self.en_block5)

        self.bottle = nn.Sequential(
            DownResiBlock(256, (128, 128, 256)),   # 10
            ResiBlock(256, (128, 128, 256)),
            ResiBlock(256, (128, 128, 256)),
        )
        #self.bottle = nn.DataParallel(self.bottle)

        self.embedding = nn.Conv2d(128, 256, kernel_size=1)

        self.en_block6 = nn.Sequential(
            #CoordConv(256, 256, with_r=False),
            ResiBlock(512, (128, 128, 512)),
            ResiBlock(512, (128, 128, 512)),
            ResiBlock(512, (128, 128, 512)),
            ResiBlock(512, (128, 128, 512)),
            #SelfAttention(512),
        )
        #self.en_block6 = nn.DataParallel(self.en_block6)

        # u1
        self.UpResiBlock1 = UpResiBlock(512, (64, 64, 256))             # 20
        #self.SEblock1 = SEblock(256, 256)
        self.de_block1 = nn.Sequential(
            #CoordConv(512, 512, with_r=False),
            ResiBlock(512, (64, 64, 256), True),
            ResiBlock(256, (64, 64, 256)),
            #SelfAttention(256),
        )
        #self.de_block1 = nn.DataParallel(self.de_block1)

        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=16)
        self.conv1 = nn.Conv2d(256, opt.anchor, kernel_size=1)

        # u2
        self.UpResiBlock2 = UpResiBlock(256, (64, 64, 256))             # 40
        #self.SEblock2 = SEblock(256, 256)
        self.de_block2 = nn.Sequential(
            #CoordConv(512, 512, with_r=False),
            ResiBlock(512, (64, 64, 256), True),
            ResiBlock(256, (64, 64, 256)),
            #SelfAttention(256),
        )
        #self.de_block2 = nn.DataParallel(self.de_block2)

        # u3
        self.UpResiBlock3 = UpResiBlock(256, (32, 32, 128))            # 80
        #self.SEblock3 = SEblock(128, 128)
        self.de_block3 = nn.Sequential(
            #CoordConv(256, 256, with_r=False),
            ResiBlock(256, (32, 32, 128), True),
            ResiBlock(128, (32, 32, 128)),
            #SelfAttention(128),
        )
        #self.de_block3 = nn.DataParallel(self.de_block3)

        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=4)
        self.conv2 = nn.Conv2d(128, opt.anchor, kernel_size=1)

        # u4
        self.UpResiBlock4 = UpResiBlock(128, (16, 16, 64))             # 160
        #self.SEblock4 = SEblock(64, 64)
        self.de_block4 = nn.Sequential(
            #CoordConv(128, 128, with_r=False),
            ResiBlock(128, (16, 16, 64), True),
            ResiBlock(64, (16, 16, 64)),
            #SelfAttention(64),
        )
        #self.de_block4 = nn.DataParallel(self.de_block4)

        # u5
        self.UpResiBlock5 = UpResiBlock(64, (8, 8, 32))               # 320
        #self.SEblock5 = SEblock(32, 32)
        self.de_block5 = nn.Sequential(
            #CoordConv(64, 64, with_r=False),
            ResiBlock(64, (8, 8, 32), True),
            ResiBlock(32, (8, 8, 32)),
        )
        #self.de_block5 = nn.DataParallel(self.de_block5)

        # u6
        self.de_block6 = nn.Sequential(
            ResiBlock(32, (4, 4, 16), True),
            ResiBlock(16, (4, 4, 16)),
            ResiBlock(16, (4, 4, 16)),
        )
        #self.de_block6 = nn.DataParallel(self.de_block6)

        self.conv2d = nn.Conv2d(16, opt.anchor, kernel_size=1)

    def forward(self, input_image, input_feature):

        d1 = self.en_block1(input_image) #320
        d2 = self.en_block2(d1) #160
        d3 = self.en_block3(d2) #80
        d4 = self.en_block4(d3) #40
        d5 = self.en_block5(d4) #20
        d6 = self.bottle(d5) #10

        f = self.embedding(input_feature)
        d6 = torch.cat((d6, f), 1)
        d6 = self.en_block6(d6) #10

        u1 = self.UpResiBlock1(d6) #20
        u1 = torch.cat((u1, d5), 1)
        u1 = self.de_block1(u1)

        s1 = self.upsample1(u1)
        s1 = torch.tanh(self.conv1(s1))

        u2 = self.UpResiBlock2(u1) #40
        u2 = torch.cat((u2, d4), 1)
        u2 = self.de_block2(u2)

        u3 = self.UpResiBlock3(u2) #80
        u3 = torch.cat((u3, d3), 1)
        u3 = self.de_block3(u3)

        s2 = self.upsample2(u3)
        s2 = torch.tanh(self.conv2(s2))

        u4 = self.UpResiBlock4(u3) #160
        u4 = torch.cat((u4, d2), 1)
        u4 = self.de_block4(u4)

        u5 = self.UpResiBlock5(u4)
        u5 = torch.cat((u5, d1), 1)
        u5 = self.de_block5(u5)

        u6 = self.de_block6(u5)
        output_img = torch.tanh(self.conv2d(u6))
        #output_img = nn.Softmax()(self.conv2d(u6))

        return output_img, s1, s2


def mse_loss(predict, target):
    mse = nn.MSELoss()
    loss = 0
    for i in range(opt.anchor):
        loss = loss + mse(predict[:,i], target[:,i])
    return loss

if __name__ == '__main__':
    #seed_all(0)

    # Initialize generator and discriminator
    generator = Generator().cuda()
    #generator.load_state_dict(torch.load('../share/output/arxiv_0708_1/weights1/g_549.pth'))

    # print('----------- Networks architecture -------------')
    # utils.print_network(generator)
    # print('----------- Networks architecture -------------')

    # Optimizers
    optimizer = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    train_generator = data_generator(is_train=True, opt=opt, seed=1)
    test_generator = data_generator(is_train=False, opt=opt, seed=1)

    train_hist, metric_hist = {}, {}
    train_hist['loss'] = []
    train_hist['s1'] = []
    train_hist['s2'] = []
    metric_hist['train'] = []
    metric_hist['test'] = []

    theta = torch.zeros((opt.anchor,))
    scale_x, scale_y = torch.ones(opt.anchor)*opt.patch_size/IMG_HEIGHT, torch.ones(opt.anchor)*opt.patch_size/IMG_HEIGHT
    background = np.array([0,0,0])
    background = np.concatenate((background, [1.])) #add alpha
    background = torch.from_numpy(background).float().cuda()
    background = background.unsqueeze(1).unsqueeze(1).repeat(1,IMG_HEIGHT,IMG_WIDTH)

    for epoch in range(opt.n_epochs):
        # ---------------------
        #  Eval
        # ---------------------
        with torch.no_grad():
            # eval mode
            generator.eval()
            img, heatmap_gt, features = next(test_generator)
            features = F.interpolate(features, [10,10]).cuda()
            heatmap_gt = heatmap_gt[:, opt.start:opt.end]

            save_image(img.data[:4],
                os.path.join(EVAL_ROOT, "%d_img_GT.png" % epoch), nrow=2, normalize=True)
            save_image(create_heatmap(img, heatmap_gt).data[:4],
                os.path.join(EVAL_ROOT, "%d_overlap_GT.png" % epoch), nrow=2, normalize=True)
            save_image(utils.merge_heatmap(heatmap_gt).data[:4],
                os.path.join(EVAL_ROOT, "%d_heatmap_GT.png" % epoch), nrow=2, normalize=True)

            confidence_map,_,_ = generator(img, features)
            for k in range(opt.anchor):
                save_image(utils.merge_heatmap(confidence_map[:,k,None]).data[:4],
                    os.path.join(EVAL_ROOT, "%d_heatmap_%d.png" % (epoch,k)), nrow=2, normalize=True)
            # output all joints together
            save_image(create_heatmap(img, confidence_map).data[:4],
                    os.path.join(EVAL_ROOT, "%d_overlap.png" % epoch), nrow=2, normalize=True)
            save_image(utils.merge_heatmap(confidence_map).data[:4],
                os.path.join(EVAL_ROOT, "%d_heatmap.png" % epoch), nrow=2, normalize=True)

            # record metric
            metric = mse_loss(confidence_map, heatmap_gt)
            metric_hist['test'].append(metric.item())

        # -------------
        #  Training
        # -------------
        # train mode
        generator.train()
        metric_train = []
        loss_hist = []
        s1_hist = []
        s2_hist = []
        for i in range(0, opt.data_length):
            img, heatmap_gt, features = next(train_generator)
            heatmap_gt = heatmap_gt[:, opt.start:opt.end]

            # Configure input
            img = Variable(img, requires_grad=True)
            features = F.interpolate(features, [10,10]).cuda()
            features = Variable(features, requires_grad=True)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer.zero_grad()

            # Generate a batch of images
            confidence_map, s1, s2 = generator(img, features)

            loss = mse_loss(confidence_map, heatmap_gt)
            s1_loss = mse_loss(s1, heatmap_gt)
            s2_loss = mse_loss(s2, heatmap_gt)

            total_loss = 10 * loss + 5 * s1_loss + 5 * s2_loss
            loss_hist.append(loss.item())
            s1_hist.append(s1_loss.item())
            s2_hist.append(s2_loss.item())

            total_loss.backward()
            optimizer.step()

            # compute metric
            with torch.no_grad():
                metric = mse_loss(heatmap_gt, confidence_map)
                metric_train.append(metric.item())

                # save training imgs
                if i % (opt.data_length//2) == 0:
                    save_image(img.data[:4],
                        os.path.join(IMAGE_ROOT, "%d_%d_img_GT.png" % (epoch,i)), nrow=2, normalize=True)
                    save_image(create_heatmap(img, heatmap_gt).data[:4],
                        os.path.join(IMAGE_ROOT, "%d_%d_overlap_GT.png" % (epoch,i)), nrow=2, normalize=True)
                    save_image(utils.merge_heatmap(heatmap_gt).data[:4],
                        os.path.join(IMAGE_ROOT, "%d_%d_heatmap_GT.png" % (epoch,i)), nrow=2, normalize=True)

                    for k in range(opt.anchor):
                        save_image(utils.merge_heatmap(confidence_map[:,k,None]).data[:4],
                            os.path.join(IMAGE_ROOT, "%d_%d_heatmap_%d.png" % (epoch,i,k)), nrow=2, normalize=True)
                    # output all joints together
                    save_image(create_heatmap(img, confidence_map).data[:4],
                            os.path.join(IMAGE_ROOT, "%d_%d_overlap.png" % (epoch,i)), nrow=2, normalize=True)
                    save_image(utils.merge_heatmap(confidence_map).data[:4],
                        os.path.join(IMAGE_ROOT, "%d_%d_heatmap.png" % (epoch,i)), nrow=2, normalize=True)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [s1: %f] [s2: %f] [Metric: %f]"
                % (epoch, opt.n_epochs, i, opt.data_length,
                loss.item(), s1_loss.item(), s2_loss.item(), metric.item())
            )
        train_hist['loss'].append(np.array(loss_hist).mean())
        train_hist['s1'].append(np.array(s1_hist).mean())
        train_hist['s2'].append(np.array(s2_hist).mean())
        metric_hist['train'].append(np.array(metric_train).mean())
        if epoch > 99:
            torch.save(generator.state_dict(), os.path.join(WEIGHT_ROOT, 'g_%d.pth' % epoch))

    utils.loss_plot(train_hist, './', 'loss_%d' % opt.exp)
    utils.loss_plot(metric_hist, './', 'metric_%d' % opt.exp)
