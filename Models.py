from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
import torch

#----------------------------------------------
# Spade Block Model Class
class SPADE(nn.Module):
    def __init__(self, size):
        super(SPADE, self).__init__()
        self.prev_input = nn.BatchNorm2d(size, affine=True)
        self.shared_active = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU()
        )
        self.lambda_conv = nn.Conv2d(128, size, 3, padding=1)
        self.beta_conv = nn.Conv2d(128, size, 3, padding=1)

    def forward(self, input1, input2):
        bn = self.prev_input(input1)
        shared = self.shared_active(input2)
        lc = self.lambda_conv(shared)
        bc = self.beta_conv(shared)
        out = torch.add(torch.mul(bn, (1+lc)), bc)
        return out

#----------------------------------------------
# Spade Block Model Class
class ResBlock(nn.Module):
    def __init__(self, size, size_out):
        super(ResBlock, self).__init__()
        if size_out != size:
            self.skip = True
        else:
            self.skip = False

        size_middle = (size + size_out) // 2
        self.SpadeA = SPADE(size)
        self.SpadeB = SPADE(size_middle)
        self.SpadeC = SPADE(size)
        self.PostA = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(size, size_middle, kernel_size=3, padding=1),
        )
        self.PostB = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(size_middle, size_out, kernel_size=3, padding=1),
        )
        if self.skip:
            self.PostC = nn.Sequential(
                nn.Conv2d(size, size_out, kernel_size=1, bias=False),
            )

    def forward(self, input1, input2):
        a_out = self.SpadeA(input1, input2)
        a_out = self.PostA(a_out)
        b_out = self.SpadeB(a_out, input2)
        b_out = self.PostB(b_out)
        if self.skip:
            c_out = self.SpadeC(input1, input2)
            c_out = self.PostC(c_out)
            res_out = torch.add(b_out, c_out)
            return res_out
        else:
            return b_out

#----------------------------------------------
#Generator Model Class
class Generator(nn.Module):
    def __init__(self, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.IM_ENCODER = Encoder(img_size)
        self.KLD = KLDLoss()
        self.input_linear = torch.nn.Linear(img_size*2, img_size * 4 * img_size//16 * img_size//16)
        self.r1 = ResBlock(512, 512)
        self.r2 = ResBlock(512, 512)
        self.r3 = ResBlock(512, 256)
        self.r4 = ResBlock(256, 256)
        self.r5 = ResBlock(256, 128)
        self.r6 = ResBlock(128, 128)
        self.r7 = ResBlock(128, 64)
        self.r8 = ResBlock(64, 32)
        self.CT3 = nn.Sequential(nn.Conv2d(32, 3, 3, padding=1),
                                 nn.Tanh())
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, img, large_map, medium_map, small_map, tiny_map, micro_map):
        mean, var = self.IM_ENCODER(img)
        z = reparameterize(mean, var)
        kld_loss = self.KLD(mean, var) * 0.05
        z = self.input_linear(z)
        z = z.view(-1, self.img_size * 4, self.img_size // 16, self.img_size // 16)
        out = self.r1(z, micro_map)
        out = self.up(out)
        out = self.r2(out, tiny_map)
        out = self.r3(out, tiny_map)
        out = self.up(out)
        out = self.r4(out, small_map)
        out = self.r5(out, small_map)
        out = self.up(out)
        out = self.r6(out, medium_map)
        out = self.r7(out, medium_map)
        out = self.up(out)
        out = self.r8(out, large_map)
        out = self.CT3(out)
        return out, kld_loss

#Multi Part Discriminator ------------------
class MultiDiscriminator(nn.Module):
    def __init__(self, img_size):
        super(MultiDiscriminator, self).__init__()
        self.d1 = Discriminator(img_size)
        self.d2 = Discriminator2(img_size)
    def forward(self, img, large_map, medium_map, small_map, tiny_map, micro_map):
        out1 = self.d1(img, large_map)
        out2 = self.d2(img)
        return ([out1], [out2])

#----------------------------------------------
#Discriminator 1 Used
class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        layers = OrderedDict()
        self.img_size = img_size
        layer_filter = 0.5
        self.inlayer = nn.Sequential(nn.Conv2d(6, int(img_size * layer_filter), kernel_size=4, stride=2, padding=1,
                                               bias=True), nn.LeakyReLU(0.2, False))
        q = 0
        for i in range(int(np.log(img_size) / np.log(2)) - 3):
            stride = 1 if layer_filter > 2 else 2
            layers[str(q)] = nn.Conv2d(int(img_size * layer_filter), int(img_size * layer_filter * 2), kernel_size=4,
                                       stride=stride, padding=1, bias=True)
            layers[str(q+1)] = nn.InstanceNorm2d(int(img_size * layer_filter * 2))
            layers[str(q+2)] = nn.LeakyReLU(0.2, False)
            q=q+3
            layer_filter = layer_filter * 2
        self.disc = nn.Sequential(layers)
        self.outlayer1 = nn.Sequential(nn.Conv2d(int(img_size*layer_filter), 1, kernel_size=4, stride=1, padding=0,
                                                 bias=True))

    def forward(self, img, map):
        concat_tensors = torch.cat((map, img), 1)
        out = self.inlayer(concat_tensors)
        out = self.disc(out)
        out = self.outlayer1(out)
        return out.view(-1)

#----------------------------------------------
# Discriminator 2 Used in PatchGan Discriminator, without concatenating sem-map, used to replace relying on an L1 Loss
class Discriminator2(nn.Module):
    def __init__(self, img_size):
        super(Discriminator2, self).__init__()
        layers = OrderedDict()
        self.img_size = img_size
        layer_filter = 0.5
        self.inlayer = nn.Sequential(nn.Conv2d(3, int(img_size*layer_filter), kernel_size=4, stride=2, padding=1,
                                               bias=False), nn.LeakyReLU(0.2, True))
        q=0
        for i in range(int(np.log(img_size) / np.log(2)) - 3):
            stride = 1 if layer_filter > 2 else 2
            layers[str(q)] = nn.Conv2d(int(img_size * layer_filter), int(img_size * layer_filter * 2), kernel_size=4,
                                       stride=stride, padding=1, bias=False)
            layers[str(q+1)] = nn.BatchNorm2d(int(img_size * layer_filter * 2))
            layers[str(q+2)] = nn.LeakyReLU(0.2, True)
            q=q+3
            layer_filter = layer_filter * 2
        self.disc = nn.Sequential(layers)
        self.outlayer1 = nn.Sequential(nn.Conv2d(int(img_size*layer_filter), 1, kernel_size=4, stride=1, padding=0,
                                                 bias=False))

    def forward(self, img):
        out = self.inlayer(img)
        out = self.disc(out)
        out = self.outlayer1(out)
        return out.view(-1)

#----------------------------------------------
# Image Encoder Class
class Encoder(nn.Module):
    def __init__(self, img_size):
        super(Encoder, self).__init__()
        self.img_size = img_size
        self.e1 = nn.Sequential(
            nn.Conv2d(3, img_size//2, 3, 2, 1),
            nn.InstanceNorm2d(img_size//2),
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(img_size//2, img_size, 3, 2, 1),
            nn.InstanceNorm2d(img_size),
            nn.LeakyReLU(0.2, False),
        )
        self.e3 = nn.Sequential(
            nn.Conv2d(img_size, img_size*2, 3, 2, 1),
            nn.InstanceNorm2d(img_size*2),
            nn.LeakyReLU(0.2, False),
        )
        self.e4 = nn.Sequential(
            nn.Conv2d(img_size*2, img_size*4, 3, 2, 1),
            nn.InstanceNorm2d(img_size*4),
            nn.LeakyReLU(0.2, False),
        )
        self.e5 = nn.Sequential(
            nn.Conv2d(img_size*4, img_size*4, 3, 2, 1),
            nn.InstanceNorm2d(img_size*4),
            nn.LeakyReLU(0.2, False),
        )
        self.meanlin = nn.Linear(img_size*4*4*4, 256)
        self.varlin = nn.Linear(img_size*4*4*4, 256)

    def forward(self, img):
        out = self.e1(img)
        out = self.e2(out)
        out = self.e3(out)
        out = self.e4(out)
        out = self.e5(out)
        out = out.view(out.size(0), -1)
        mean = self.meanlin(out)
        var = self.varlin(out)
        return mean, var

# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std) + mu

# Class implementing Hinge Loss
class CustomHingeLoss(torch.nn.Module):
    def __init__(self):
        super(CustomHingeLoss, self).__init__()
    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss

# Class implementing Relativistic Loss taken from https://github.com/AlexiaJM/RelativisticGAN
class RealativisticLoss(torch.nn.Module):
    def __init__(self):
        super(RealativisticLoss, self).__init__()
    def forward(self, output, target, Gan=False):
        if Gan:
            err = (torch.mean((target - torch.mean(output) - 1) ** 2) + torch.mean(
                (output - torch.mean(target) + 1) ** 2)) / 2
        else:
            err = (torch.mean((target - torch.mean(output) + 1) ** 2) + torch.mean(
                (output - torch.mean(target) - 1) ** 2)) / 2
        return err

# Adds noise sampled from a gaussian to an input image
def noiseer(image):
    row,col,ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = torch.tensor((gauss.reshape(row,col,ch))).float().to(device='cuda')
    noisy = image + gauss
    return noisy

# Weight initialization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    for x in m.children():
        weights_init_normal(x)