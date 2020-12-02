import  torch
import torch.nn as nn
from .blocks import *

class G(nn.Module):
  def __init__(self, output_dim_a, output_dim_b, nz):
    super(G, self).__init__()
    self.nz = nz
    ini_tch = 256
    tch_add = ini_tch
    tch = ini_tch
    self.tch_add = tch_add
    self.decA1 = MisINSResBlock(tch, tch_add)
    self.decA2 = MisINSResBlock(tch, tch_add)
    self.decA3 = MisINSResBlock(tch, tch_add)
    self.decA4 = MisINSResBlock(tch, tch_add)

    decA5 = []
    decA5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    decA5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    decA5 += [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]
    decA5 += [nn.Tanh()]
    self.decA5 = nn.Sequential(*decA5)

    tch = ini_tch
    self.decB1 = MisINSResBlock(tch, tch_add)
    self.decB2 = MisINSResBlock(tch, tch_add)
    self.decB3 = MisINSResBlock(tch, tch_add)
    self.decB4 = MisINSResBlock(tch, tch_add)
    decB5 = []
    decB5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    decB5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
    tch = tch//2
    decB5 += [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)]
    decB5 += [nn.Tanh()]
    self.decB5 = nn.Sequential(*decB5)

    self.mlpA = nn.Sequential(
        nn.Linear(8, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, tch_add*4))
    self.mlpB = nn.Sequential(
        nn.Linear(8, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, tch_add*4))
    return

  def forward_a(self, x, z):
    z = self.mlpA(z)
    z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
    z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
    out1 = self.decA1(x, z1)
    out2 = self.decA2(out1, z2)
    out3 = self.decA3(out2, z3)
    out4 = self.decA4(out3, z4)
    out = self.decA5(out4)
    return out

  def forward_b(self, x, z):
    z = self.mlpB(z)
    z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
    z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
    out1 = self.decB1(x, z1)
    out2 = self.decB2(out1, z2)
    out3 = self.decB3(out2, z3)
    out4 = self.decB4(out3, z4)
    out = self.decB5(out4)
    return out

class G_concat(nn.Module):
  def __init__(self, output_dim_a, output_dim_b, nz):
    super(G_concat, self).__init__()
    self.nz = nz
    tch = 256
    dec_share = []
    dec_share += [INSResBlock(tch, tch)]
    self.dec_share = nn.Sequential(*dec_share)
    tch = 256+self.nz
    decA1 = []
    for i in range(0, 3):
      decA1 += [INSResBlock(tch, tch)]
    tch = tch + self.nz
    decA2 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decA3 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decA4 = [nn.ConvTranspose2d(tch, output_dim_a, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
    self.decA1 = nn.Sequential(*decA1)
    self.decA2 = nn.Sequential(*[decA2])
    self.decA3 = nn.Sequential(*[decA3])
    self.decA4 = nn.Sequential(*decA4)

    tch = 256+self.nz
    decB1 = []
    for i in range(0, 3):
      decB1 += [INSResBlock(tch, tch)]
    tch = tch + self.nz
    decB2 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decB3 = ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
    tch = tch//2
    tch = tch + self.nz
    decB4 = [nn.ConvTranspose2d(tch, output_dim_b, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
    self.decB1 = nn.Sequential(*decB1)
    self.decB2 = nn.Sequential(*[decB2])
    self.decB3 = nn.Sequential(*[decB3])
    self.decB4 = nn.Sequential(*decB4)

  def forward_a(self, x, z):
    out0 = self.dec_share(x)
    z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    x_and_z = torch.cat([out0, z_img], 1)
    out1 = self.decA1(x_and_z)
    z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
    x_and_z2 = torch.cat([out1, z_img2], 1)
    out2 = self.decA2(x_and_z2)
    z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
    x_and_z3 = torch.cat([out2, z_img3], 1)
    out3 = self.decA3(x_and_z3)
    z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
    x_and_z4 = torch.cat([out3, z_img4], 1)
    out4 = self.decA4(x_and_z4)
    return out4

  def forward_b(self, x, z):
    out0 = self.dec_share(x)
    z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    x_and_z = torch.cat([out0,  z_img], 1)
    out1 = self.decB1(x_and_z)
    z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out1.size(2), out1.size(3))
    x_and_z2 = torch.cat([out1, z_img2], 1)
    out2 = self.decB2(x_and_z2)
    z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out2.size(2), out2.size(3))
    x_and_z3 = torch.cat([out2, z_img3], 1)
    out3 = self.decB3(x_and_z3)
    z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), out3.size(2), out3.size(3))
    x_and_z4 = torch.cat([out3, z_img4], 1)
    out4 = self.decB4(x_and_z4)
    return out4