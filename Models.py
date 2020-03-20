import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
from collections import OrderedDict

def noiseer(image):
    row,col,ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = torch.tensor((gauss.reshape(row,col,ch))).float().to(device='cuda')
    noisy = image + gauss
    return noisy

#----------------------------------------------
# Spade Block Model Class
class SPADE(nn.Module):
    def __init__(self, size):
        super(SPADE, self).__init__()
        self.prev_input = nn.InstanceNorm2d(size)
        self.shared_active = nn.Sequential(
            nn.Conv2d(8, 128, 3, padding=1),
            nn.ReLU()
        )
        self.lambda_conv = nn.Conv2d(128, size, 3, padding=1)
        self.beta_conv = nn.Conv2d(128, size, 3, padding=1)

    def forward(self, input1, input2):
        bn = self.prev_input(input1)
        input2 = F.interpolate(input2, size=input1.size()[2:], mode='nearest')
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
        size_middle = (size + size_out) // 2
        self.SpadeA = SPADE(size)
        self.SpadeB = SPADE(size_middle)
        self.SpadeC = SPADE(size)

        self.PostA = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(size, size_middle, kernel_size=3, padding=1),
        )
        self.PostB = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(size_middle, size_out, kernel_size=3, padding=1)
        )
        self.PostC = nn.Sequential(
            nn.Conv2d(size, size_out, kernel_size=1, bias=False)
        )

    def forward(self, input1, input2):
        a_out = self.SpadeA(input1, input2)
        a_out = self.PostA(a_out)
        b_out = self.SpadeB(a_out, input2)
        b_out = self.PostB(b_out)
        c_out = self.SpadeC(input1, input2)
        c_out = self.PostC(c_out)
        res_out = torch.add(b_out, c_out)
        return res_out

#----------------------------------------------
#Generator Model Class
class Generator(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.IM_ENCODER = Encoder(img_size)
        self.KLD = KLDLoss()
        self.input_linear = torch.nn.Linear(img_size*2, img_size * 8 * img_size//16 * img_size//16)
        self.r1 = ResBlock(1024, 512)
        self.r2 = ResBlock(512, 256)
        self.r3 = ResBlock(256, 128)
        self.r4 = ResBlock(128, 64)
        self.r5 = ResBlock(64, 32)
        self.CT3 = nn.Sequential(nn.ReLU(True),
                                 nn.Conv2d(32, 3, 3, padding=1, bias=False),
                                 nn.Tanh())
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, img, large_map, medium_map, small_map, tiny_map, micro_map):
        mean, var = self.IM_ENCODER(img)
        z = reparameterize(mean, var)
        kld_loss = self.KLD(mean, var) * 0.05
        z = self.input_linear(z)
        z = z.view(-1, self.img_size * 8, self.img_size // 16, self.img_size // 16)
        out = self.r1(z, micro_map)
        out = self.up(out)
        out = self.r2(out, tiny_map)
        out = self.up(out)
        out = self.r3(out, small_map)
        out = self.up(out)
        out = self.r4(out, medium_map)
        out = self.up(out)
        out = self.r5(out, large_map)
        out = self.CT3(out)
        return out, kld_loss

#Multi Discriminator ------------------
class MultiDiscriminator(nn.Module):
    def __init__(self, img_size):
        super(MultiDiscriminator, self).__init__()
        self.d1 = Discriminator(img_size)
        self.d6 = Discriminator2(img_size)

    def forward(self, img, large_map, medium_map, small_map, tiny_map, micro_map):
        out1 = self.d1(img, large_map)
        out6 = self.d6(img)
        return ([out1], [out6])

#----------------------------------------------
#Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        layers = OrderedDict()
        self.img_size = img_size
        layer_filter = 0.5
        self.inlayer = nn.Sequential(nn.Conv2d(11, int(img_size * layer_filter), 4, 2, 1),
                                     nn.LeakyReLU(0.2, False),
                                     nn.Dropout2d(0.2))
        q = 0
        for i in range(int(np.log(img_size) / np.log(2)) - 3):
            if layer_filter == 4:
                layers[str(q)] = nn.Conv2d(int(img_size * layer_filter), int(img_size * layer_filter), 4, 2, 1)
                layers[str(q + 1)] = nn.InstanceNorm2d(int(img_size * layer_filter))
                layers[str(q + 2)] = nn.LeakyReLU(0.2, False)
                layers[str(q + 3)] = nn.Dropout2d(0.2)
                q=q+4
            else:
                layers[str(q)] = nn.Conv2d(int(img_size * layer_filter), int(img_size * layer_filter * 2), 4, 2, 1)
                layers[str(q + 1)] = nn.InstanceNorm2d(int(img_size * layer_filter * 2))
                layers[str(q + 2)] = nn.LeakyReLU(0.2, False)
                layers[str(q + 3)] = nn.Dropout2d(0.2)
                layer_filter = layer_filter * 2
                q=q+4
        self.disc = nn.Sequential(layers)
        self.outlayer1 = nn.Sequential(nn.Conv2d(int(img_size*layer_filter), 1, 4, 1, 0),
                                       nn.Sigmoid())

    def forward(self, img, map):
        concat_tensors = torch.cat((img, map), 1)
        out = self.inlayer(concat_tensors)
        out = self.disc(out)
        out = self.outlayer1(out)
        return out.view(-1)

# Discriminator 2
class Discriminator2(nn.Module):
    def __init__(self, img_size):
        super(Discriminator2, self).__init__()
        layers = OrderedDict()
        self.img_size = img_size
        layer_filter = 0.5
        self.inlayer = nn.Sequential(nn.Conv2d(3, int(img_size*layer_filter), 4, 2, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True))
        q=0
        for i in range(int(np.log(img_size) / np.log(2)) - 3):
            if layer_filter == 4:
                layers[str(q)] = nn.Conv2d(int(img_size*layer_filter), int(img_size*layer_filter), 4, 2, 1, bias=False)
                layers[str(q+1)] = nn.BatchNorm2d(int(img_size*layer_filter))
                layers[str(q+2)] = nn.LeakyReLU(0.2, inplace=True)
                q=q+3
            else:
                layers[str(q)] = nn.Conv2d(int(img_size*layer_filter), int(img_size*layer_filter*2), 4, 2, 1, bias=False)
                layers[str(q+1)] = nn.BatchNorm2d(int(img_size*layer_filter*2))
                layers[str(q+2)] = nn.LeakyReLU(0.2, inplace=True)
                layer_filter = layer_filter*2
                q=q+3
        self.disc = nn.Sequential(layers)
        self.outlayer1 = nn.Sequential(nn.Conv2d(int(img_size*layer_filter), 1, 4, 1, 0, bias=False),
                                       nn.Sigmoid())

    def forward(self, img):
        out = self.inlayer(img)
        out = self.disc(out)
        out = self.outlayer1(out)
        return out.view(-1)

class Encoder(nn.Module):
    def __init__(self, img_size):
        super(Encoder, self).__init__()
        self.img_size = img_size
        self.p1 = nn.Sequential(
            nn.Conv2d(3, img_size//2, 3, 2, 1),
            nn.InstanceNorm2d(img_size//2),
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(img_size//2, img_size, 3, 2, 1),
            nn.InstanceNorm2d(img_size),
            nn.LeakyReLU(0.2, False),
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(img_size, img_size*2, 3, 2, 1),
            nn.InstanceNorm2d(img_size*2),
            nn.LeakyReLU(0.2, False),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(img_size*2, img_size*4, 3, 2, 1),
            nn.InstanceNorm2d(img_size*4),
            nn.LeakyReLU(0.2, False),
        )
        self.p5 = nn.Sequential(
            nn.Conv2d(img_size*4, img_size*4, 3, 2, 1),
            nn.InstanceNorm2d(img_size),
            nn.LeakyReLU(0.2, False),
        )
        self.meanlin = nn.Linear(img_size*4*4*4, 256)
        self.varlin = nn.Linear(img_size*4*4*4, 256)

    def forward(self, img):
        out = self.p1(img)
        out = self.p2(out)
        out = self.p3(out)
        out = self.p4(out)
        out = self.p5(out)
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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    for x in m.children():
        weights_init_normal(x)