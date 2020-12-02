import torch.nn as nn
from .blocks import *

class E_content(nn.Module):
  def __init__(self, input_dim_a, input_dim_b):
    super(E_content, self).__init__()
    encA_c = []
    tch = 64
    encA_c += [LeakyReLUConv2d(input_dim_a, tch, kernel_size=7, stride=1, padding=3)]
    for i in range(1, 3):
      encA_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, 3):
      encA_c += [INSResBlock(tch, tch)]

    encB_c = []
    tch = 64
    encB_c += [LeakyReLUConv2d(input_dim_b, tch, kernel_size=7, stride=1, padding=3)]
    for i in range(1, 3):
      encB_c += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2
    for i in range(0, 3):
      encB_c += [INSResBlock(tch, tch)]

    enc_share = []
    for i in range(0, 1):
      enc_share += [INSResBlock(tch, tch)]
      enc_share += [GaussianNoiseLayer()]
      self.conv_share = nn.Sequential(*enc_share)

    self.convA = nn.Sequential(*encA_c)
    self.convB = nn.Sequential(*encB_c)

  def forward(self, xa, xb):
    outputA = self.convA(xa)
    outputB = self.convB(xb)
    outputA = self.conv_share(outputA)
    outputB = self.conv_share(outputB)
    return outputA, outputB

  def forward_a(self, xa):
    outputA = self.convA(xa)
    outputA = self.conv_share(outputA)
    return outputA

  def forward_b(self, xb):
    outputB = self.convB(xb)
    outputB = self.conv_share(outputB)
    return outputB

class E_attr(nn.Module):
  def __init__(self, input_dim_a, input_dim_b, output_nc=8):
    super(E_attr, self).__init__()
    dim = 64
    self.model_a = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_a, dim, 7, 1),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))
    self.model_b = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_dim_b, dim, 7, 1),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim*2, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*2, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim*4, dim*4, 4, 2),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(dim*4, output_nc, 1, 1, 0))
    return

  def forward(self, xa, xb):
    xa = self.model_a(xa)
    xb = self.model_b(xb)
    output_A = xa.view(xa.size(0), -1)
    output_B = xb.view(xb.size(0), -1)
    return output_A, output_B

  def forward_a(self, xa):
    xa = self.model_a(xa)
    output_A = xa.view(xa.size(0), -1)
    return output_A

  def forward_b(self, xb):
    xb = self.model_b(xb)
    output_B = xb.view(xb.size(0), -1)
    return output_B

class E_attr_concat(nn.Module):
  def __init__(self, input_dim_a, input_dim_b, output_nc=8, norm_layer=None, nl_layer=None):
    super(E_attr_concat, self).__init__()

    ndf = 64
    n_blocks=4
    max_ndf = 4

    conv_layers_A = [nn.ReflectionPad2d(1)]
    conv_layers_A += [nn.Conv2d(input_dim_a, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers_A += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers_A += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    self.fc_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.fcVar_A = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.conv_A = nn.Sequential(*conv_layers_A)

    conv_layers_B = [nn.ReflectionPad2d(1)]
    conv_layers_B += [nn.Conv2d(input_dim_b, ndf, kernel_size=4, stride=2, padding=0, bias=True)]
    for n in range(1, n_blocks):
      input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
      output_ndf = ndf * min(max_ndf, n+1)  # 2**n
      conv_layers_B += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]
    conv_layers_B += [nl_layer(), nn.AdaptiveAvgPool2d(1)] # AvgPool2d(13)
    self.fc_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.fcVar_B = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
    self.conv_B = nn.Sequential(*conv_layers_B)

  def forward(self, xa, xb):
    x_conv_A = self.conv_A(xa)
    conv_flat_A = x_conv_A.view(xa.size(0), -1)
    output_A = self.fc_A(conv_flat_A)
    outputVar_A = self.fcVar_A(conv_flat_A)
    x_conv_B = self.conv_B(xb)
    conv_flat_B = x_conv_B.view(xb.size(0), -1)
    output_B = self.fc_B(conv_flat_B)
    outputVar_B = self.fcVar_B(conv_flat_B)
    return output_A, outputVar_A, output_B, outputVar_B

  def forward_a(self, xa):
    x_conv_A = self.conv_A(xa)
    conv_flat_A = x_conv_A.view(xa.size(0), -1)
    output_A = self.fc_A(conv_flat_A)
    outputVar_A = self.fcVar_A(conv_flat_A)
    return output_A, outputVar_A

  def forward_b(self, xb):
    x_conv_B = self.conv_B(xb)
    conv_flat_B = x_conv_B.view(xb.size(0), -1)
    output_B = self.fc_B(conv_flat_B)
    outputVar_B = self.fcVar_B(conv_flat_B)
    return output_B, outputVar_B