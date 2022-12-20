import math
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, last_relu=True, bias=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, 3, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, 3, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)
        self.last_relu = last_relu

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x) if self.last_relu else x


class DownSample(nn.Module):
    def __init__(self, channel, bias=True):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(channel, channel, 3, 2, 1, bias=bias)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channel, bias=True):
        super(UpSample, self).__init__()
        self.conv = nn.ConvTranspose2d(channel, channel, 3, 2, 1, output_padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class SkipBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(SkipBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.upsample = UpSample(out_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, skip):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat((x, skip), 1)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class MemModule(nn.Module):
    def __init__(self, num_protos=20, fea_dim=512, shrink_thres=0.0025, branch=False):
        super(MemModule, self).__init__()
        print(f'loading number of protos = {num_protos}')
        self.num_protos = num_protos
        self.fea_dim = fea_dim
        self.proto = nn.Parameter(torch.Tensor(self.num_protos, self.fea_dim))  # N * 512
        self.shrink_thres = shrink_thres
        self.initial_weight()
        self.branch = branch

    def initial_weight(self):
        stdv = 1. / math.sqrt(self.fea_dim)
        self.proto.data.uniform_(-stdv, stdv)

    @staticmethod
    def hard_shrink_relu(input, lambd=0.0, epsilon=1e-12):
        output = (torch.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
        return output

    def forward(self, X):
        if self.branch:
            # [B, X, fea_dim]
            batch = X.shape[0]
            X = X.contiguous().view(-1, self.fea_dim)
            similarity = F.linear(X, self.proto)  # [BX, fea_dim] * [fea_dim, N] = [BX, N]
            similarity = F.softmax(similarity, dim=1)  # [BX, N]
            if self.shrink_thres > 0:
                similarity = self.hard_shrink_relu(similarity, lambd=self.shrink_thres)
                similarity = F.normalize(similarity, p=1, dim=1)  # [BX, N]
            mem_trans = self.proto.permute(1, 0)  # [fea_dim, N]
            proto_based_fea = F.linear(similarity, mem_trans)  # [BX, N] *  [N, fea_dim] = [BX, fea_dim]
            proto_based_fea = proto_based_fea.view(batch, -1, self.fea_dim)  # [B, X, fea_dim]
            similarity = similarity.view(batch, -1, self.num_protos)  # [B, X, N]
            similarity = similarity.permute(0, 2, 1)  # [B, N, X]
            return {'output': proto_based_fea, 'similarity': similarity}
        else:
            batch, channel, H, W = X.shape  # [B, 512, H, W]
            X = X.permute(0, 2, 3, 1)  # [B, H, W, 512]
            X = X.contiguous().view(-1, channel)  # [BHW, 512]
            similarity = F.linear(X, self.proto)  # [BHW, 512] * [512, N] = [BHW, N]
            similarity = F.softmax(similarity, dim=1)  # [BHW, N]
            if self.shrink_thres > 0:
                similarity = self.hard_shrink_relu(similarity, lambd=self.shrink_thres)
                similarity = F.normalize(similarity, p=1, dim=1)  # [BHW, N]
            mem_trans = self.proto.permute(1, 0)  # [512, N]
            proto_based_fea = F.linear(similarity, mem_trans)  # [BHW, N] *  [N, 512] = [BHW, 512]
            proto_based_fea = proto_based_fea.view(batch, H, W, channel)  # [B, H, W, 512]
            proto_based_fea = proto_based_fea.permute(0, 3, 1, 2)  # [B, 512, H, W]
            similarity = similarity.view(batch, H, W, self.num_protos)  # [B, H, W, N]
            similarity = similarity.permute(0, 3, 1, 2)  # [B, N, H, W]
            return {'output': proto_based_fea, 'similarity': similarity}


class TemporalShift(nn.Module):
    def __init__(self, n_div=8, direction='left'):
        super(TemporalShift, self).__init__()
        self.fold_div = n_div
        self.direction = direction

        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, fold_div=self.fold_div, direction=self.direction)
        return x

    @staticmethod
    def shift(x, fold_div=8, direction='left'):
        #  [B, N, C, H, W]
        c = x.shape[2]
        fold = c // fold_div
        out = torch.zeros_like(x)
        if direction == 'left':
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, :, fold:] = x[:, :, fold:]  # not shift
        elif direction == 'right':
            out[:, 1:, :fold] = x[:, :-1, :fold]  # shift right
            out[:, :, fold:] = x[:, :, fold:]  # not shift
        else:  # shift left and right
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        print('using channel attention module!')
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        return x, y
