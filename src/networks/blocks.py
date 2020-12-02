import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .funcs import *
from .SpectralNorm import spectral_norm

class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    return
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)

class BasicBlock(nn.Module):
  def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
    super(BasicBlock, self).__init__()
    layers = []
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += conv3x3(inplanes, inplanes)
    if norm_layer is not None:
      layers += [norm_layer(inplanes)]
    layers += [nl_layer()]
    layers += [convMeanpool(inplanes, outplanes)]
    self.conv = nn.Sequential(*layers)
    self.shortcut = meanpoolConv(inplanes, outplanes)
  def forward(self, x):
    out = self.conv(x) + self.shortcut(x)
    return out

class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    if sn:
      model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
    else:
      model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    if 'norm' == 'Instance':
      model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    #elif == 'Group'
  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += self.conv3x3(inplanes, planes, stride)
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += self.conv3x3(planes, planes)
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class MisINSResBlock(nn.Module):
  def conv3x3(self, dim_in, dim_out, stride=1):
    return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
  def conv1x1(self, dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
  def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
    super(MisINSResBlock, self).__init__()
    self.conv1 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.conv2 = nn.Sequential(
        self.conv3x3(dim, dim, stride),
        nn.InstanceNorm2d(dim))
    self.blk1 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    self.blk2 = nn.Sequential(
        self.conv1x1(dim + dim_extra, dim + dim_extra),
        nn.ReLU(inplace=False),
        self.conv1x1(dim + dim_extra, dim),
        nn.ReLU(inplace=False))
    model = []
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.conv1.apply(gaussian_weights_init)
    self.conv2.apply(gaussian_weights_init)
    self.blk1.apply(gaussian_weights_init)
    self.blk2.apply(gaussian_weights_init)
  def forward(self, x, z):
    residual = x
    z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
    o1 = self.conv1(x)
    o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
    o3 = self.conv2(o2)
    out = self.blk2(torch.cat([o3, z_expand], dim=1))
    out += residual
    return out

class GaussianNoiseLayer(nn.Module):
  def __init__(self,):
    super(GaussianNoiseLayer, self).__init__()
  def forward(self, x):
    if self.training == False:
      return x
    noise = Variable(torch.randn(x.size()))#.cuda(x.get_device()))
    return x + noise

class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [LayerNorm(n_out)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)