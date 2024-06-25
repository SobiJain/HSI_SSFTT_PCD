import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from dct import dct_2d, idct_2d

def make_group(num_filters):
    x = 2
    output_channels = []

    while num_filters >= x:
        output_channels.append(num_filters // x)
        num_filters = num_filters - (num_filters // x)
        x = x*2
    
    if num_filters > 0:
        output_channels.append(num_filters)
    
    return output_channels

def make_group1(num_filters):
    x = 2
    output_channels = []
    y=3
    while y > 0:
        output_channels.append(num_filters // x)
        num_filters = num_filters - (num_filters // x)
        y=y-1
    if num_filters > 0:
        output_channels.append(num_filters)
    
    return output_channels

class LogConv2D(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(LogConv2D, self).__init__()

    output_channels = make_group(output_channels)
    self.num_groups = len(output_channels)
    self.layers = nn.Sequential()

    for i in output_channels:
      self.layers.append(nn.Sequential(
              nn.Conv2d(in_channels=8*30, out_channels=i, kernel_size=(3, 3), padding=1),
              nn.BatchNorm2d(i),
              nn.ReLU(),
          ))

  def forward(self, x):

    #input_channel_groups = torch.chunk(x, self.num_groups, dim=0)
    output_channel_groups = []
    for i in self.layers:
      x1 = i(x)
      output_channel_groups.append(x1)

    output = torch.cat(output_channel_groups, dim=1)
    return output

    return x

class LogConv3D(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(LogConv3D, self).__init__()

    output_channels = make_group(output_channels)
    self.num_groups = len(output_channels)
    self.layers = nn.Sequential()

    for i in output_channels:
      self.layers.append(nn.Sequential(
              nn.Conv3d(in_channels=1, out_channels=i, kernel_size=(3, 3, 3), padding=1),
              nn.BatchNorm3d(i),
              nn.ReLU(),
          ))

  def forward(self, x):

    #input_channel_groups = torch.chunk(x, self.num_groups, dim=0)
    output_channel_groups = []
    for i in self.layers:
      x1 = i(x)
      output_channel_groups.append(x1)

    output = torch.cat(output_channel_groups, dim=1)
    return output


    return x

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

NUM_CLASS = 16

class SSFTTnet_DCT(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=4, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super(SSFTTnet_DCT, self).__init__()
        self.L = num_tokens
        self.cT = dim
        self.conv3d_features = nn.Sequential(
            # nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            LogConv3D(in_channels, 8),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            # nn.Conv2d(in_channels=8*28, out_channels=64, kernel_size=(3, 3)),
            LogConv2D(8*30, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq, Wk, Wv for each vector, hence *3
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        x = self.conv2d_features(x) 

        #DCT
        x = dct_2d(x)
        x = x[:,:,:7,:7]
        x = idct_2d(x)

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # Multi-head attentions

        # Pooling operations on q
        q1 = nn.AdaptiveAvgPool2d((n, h))(q)  # q1: (batch_size, heads, sequence_length, class)
        q2 = nn.AdaptiveMaxPool2d((h, n))(q)  # q2: (batch_size, heads, class, head_dim)

        # Apply softmax on q1
        q1 = q1.softmax(dim=-1)

        # Matrix multiply q2 with k transpose
        qk = torch.einsum('bhcd,bhjd->bhcj', q2, k) * self.scale

        # Apply softmax on qk
        qk = qk.softmax(dim=-1)

        # Multiply qk with v
        qkv = torch.einsum('bhcj,bhjd->bhcd', qk, v)

        # Multiply q1 with qkv
        out = torch.einsum('bhnd,bhcd->bhcn', q1, qkv)
        out = rearrange(out, 'b h n d -> b n (h d)')  # Concatenate heads

        out = self.nn1(out)
        out = self.do1(out)
        return out


if __name__ == '__main__':
    model = SSFTTnet_DCT()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 15, 15)
    y = model(input)
    print(y.size())
