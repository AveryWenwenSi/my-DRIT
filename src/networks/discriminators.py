import torch.nn as nn
from .blocks import *

class Dis_content(nn.Module):
  def __init__(self):
    super(Dis_content, self).__init__()
    model = []
    model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv2d(256, 256, kernel_size=7, stride=2, padding=1, norm='Instance')]
    model += [LeakyReLUConv2d(256, 256, kernel_size=4, stride=1, padding=0)]
    model += [nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    out = self.model(x)
    out = out.view(-1)
    outs = []
    outs.append(out)
    return outs

class MultiScaleDis(nn.Module):
  def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False):
    super(MultiScaleDis, self).__init__()
    ch = 64
    self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    self.Diss = nn.ModuleList()
    for _ in range(n_scale):
      self.Diss.append(self._make_net(ch, input_dim, n_layer, norm, sn))

  def _make_net(self, ch, input_dim, n_layer, norm, sn):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1, norm, sn)]
    tch = ch
    for _ in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
      tch *= 2
    if sn:
      model += [spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
    else:
      model += [nn.Conv2d(tch, 1, 1, 1, 0)]
    return nn.Sequential(*model)

  def forward(self, x):
    outs = []
    for Dis in self.Diss:
      outs.append(Dis(x))
      x = self.downsample(x)
    return outs

class Dis(nn.Module):
  def __init__(self, input_dim, norm='None', sn=False):
    super(Dis, self).__init__()
    ch = 64
    n_layer = 6
    self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

  def _make_net(self, ch, input_dim, n_layer, norm, sn):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] #16
    tch = ch
    for i in range(1, n_layer-1):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)] # 8
      tch *= 2
    model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)] # 2
    tch *= 2
    if sn:
      model += [spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]  # 1
    else:
      model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
    return nn.Sequential(*model)

  def cuda(self,gpu):
    self.model.cuda(gpu)

  def forward(self, x_A):
    out_A = self.model(x_A)
    out_A = out_A.view(-1)
    outs_A = []
    outs_A.append(out_A)
    return outs_A