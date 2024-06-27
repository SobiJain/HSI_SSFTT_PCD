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

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq, Wk, Wv for each vector, hence *3
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into q, k, v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # Multi-head attentions
        d = 8
        # print('q', q.shape)

        # Pooling operations on q
        q1 = nn.AdaptiveAvgPool2d((n, h))(q)  # q1: (batch_size, heads, sequence_length, class)
        q2 = nn.AdaptiveMaxPool2d((h, d))(q)  # q2: (batch_size, heads, class, head_dim)

        # Apply softmax on q1
        q1 = q1.softmax(dim=-1)
        # print('q1', q1.shape)

        # Matrix multiply q2 with k transpose
        # print('q2', q2.shape)
        k = rearrange(k, 'b n h w -> b n w h')  # Transpose
        # print('k', k.shape)
        qk = torch.einsum('bhcd,bhdj->bhcj', q2, k) * self.scale
        # print('qk', qk.shape)

        # Apply softmax on qk
        qk = qk.softmax(dim=-1)

        # Multiply qk with v
        qkv = torch.einsum('bhcj,bhjd->bhcd', qk, v)
        # print('qkv', qkv.shape)

        # Multiply q1 with qkv
        out = torch.einsum('bhnd,bhdc->bhnc', q1, qkv)
        # print('out', out.shape)
        out = rearrange(out, 'b h n d -> b n (h d)')  # Concatenate heads
        # print('out', out.shape)

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

NUM_CLASS = 3

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

        # Tokenization
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.kaiming_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.kaiming_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):

        x1 = x[:,:,:15,:,1:]
        x2 = x[:,:,15:23,:,:8]
        x3 = x[:,:,23:,1:,:]

        x1 = F.pad(input=x1, pad=(0, 1, 0, 0), mode='constant', value=0)
        x2 = F.pad(input=x2, pad=(1, 0, 0, 0), mode='constant', value=0)
        x3 = F.pad(input=x3, pad=(0, 0, 0, 1), mode='constant', value=0)

        x = torch.cat((x1,x2,x3), dim = 2)
        
        x = self.conv3d_features(x)
        x = rearrange(x, 'b c h w y -> b (c h) w y')
        
        x1 = x[:,:120,:,:8]
        x2 = x[:,120:180,:,1:]
        x3 = x[:,180:,:8,:]

        x1 = F.pad(input=x1, pad=(1, 0, 0, 0), mode='constant', value=0)
        x2 = F.pad(input=x2, pad=(0, 1, 0, 0), mode='constant', value=0)
        x3 = F.pad(input=x3, pad=(0, 0, 1, 0), mode='constant', value=0)

        x = torch.cat((x1,x2,x3), dim = 1)

        x = self.conv2d_features(x) 

        x = rearrange(x,'b c h w -> b (h w) c')
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask)  # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x


if __name__ == '__main__':
    model = SSFTTnet_DCT()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 9, 9)
    y = model(input)
    print(y.size())
