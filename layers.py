import torch
from torch import nn
from torch.nn import functional as F
import unfoldNd
from torch.nn.init import kaiming_normal_


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = BasicConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x


class ResConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ResConv3d, self).__init__()
        self.conv1 = BasicConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.res_conv = BasicConv3d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.res_conv = None

    def forward(self, x):
        residual = self.res_conv(x) if self.res_conv else x
        x = self.conv1(x)
        x += residual
        return x


class SkConv3d(nn.Module):
    def __init__(self, channel, M=2, reduction=4, L=4, G=4):
        super(SkConv3d, self).__init__()
        self.M = M
        self.channel = channel
        # 尺度不变
        self.conv = nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(self.M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(
                nn.Conv3d(channel, channel, 3, 1, padding=1 + i, dilation=1 + i, groups=G, bias=False),
                nn.BatchNorm3d(channel),
                nn.ReLU(inplace=True))
            )
        self.fbap = nn.AdaptiveAvgPool3d(1)  # 三维自适应pool到指定维度    这里指定为1，实现 三维GAP
        d = max(channel // reduction, L)  # 计算向量Z 的长度d   下限为L
        self.fc1 = nn.Conv3d(in_channels=channel, out_channels=d, kernel_size=(1, 1, 1), bias=False)
        self.fc2 = nn.Conv3d(in_channels=d, out_channels=channel * M, kernel_size=(1, 1, 1), bias=False)  # 升维
        self.softmax = nn.Softmax(dim=1)  # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1

    def forward(self, input):
        batch_size, channel, _, _, _ = input.shape
        # split阶段
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))

        # fusion阶段
        U = output[0] + output[1]  # 逐元素相加生成 混合特征U
        s = self.fbap(U)
        z = self.fc1(s)  # S->Z降维
        a_b = self.fc2(z)  # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b = a_b.reshape(batch_size, self.M, channel, 1, 1, 1)  # 调整形状，变为 两个全连接层的值
        a_b = self.softmax(a_b)  # 使得两个全连接层对应位置进行softmax

        # selection阶段
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b = list(map(lambda x: torch.squeeze(x, dim=1), a_b))  # 压缩第一维
        V = list(map(lambda x, y: x * y, output, a_b))  # 权重与对应 不同卷积核输出的U 逐元素相乘
        V = V[0] + V[1]  # 两个加权后的特征 逐元素相加
        return V


class IAS_Net(nn.Module):
    def __init__(self, features, M, **kwargs):
        super(IAS_Net, self).__init__()
        self.M = M
        self.features = features
        self.avg = nn.AdaptiveAvgPool3d((1,1,1))
        self.mlp = nn.Sequential(nn.Linear(features, features * 2),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(features * 2, features),
                                 nn.Sigmoid())
        self.conv_q = BasicConv3d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv_k = BasicConv3d(features, features, kernel_size=5, stride=1, padding=2)
        self.conv_v = BasicConv3d(features, features, kernel_size=1, stride=1, padding=0)

        self.norm = nn.InstanceNorm3d(features, affine=True)
        self.soft = nn.Softmax(dim=2)
        self.conv_1 = nn.Conv3d(2 * features, features, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        global ret, result
        channel_x = self.avg(x)
        channel_x = channel_x.squeeze(-1).squeeze(-1).squeeze(-1)
        channel_w = self.mlp(channel_x)
        batch_size = len(channel_w.tolist())
        lis = channel_w.tolist()
        for batch in range(batch_size):
            pro = lis[batch]
            for i in range(len(pro)):
                vec = pro[i] * x[batch, i, :, :, :]
                vec = torch.unsqueeze(vec, 0)
                if i == 0:
                    ret = vec
                else:
                    ret = torch.cat([ret, vec], dim=0)
            ret = torch.unsqueeze(ret, 0)
            if batch == 0:
                result = ret
            else:
                result = torch.cat([result, ret], dim=0)
        channel_out = result

        Q = self.conv_q(channel_out)
        K = self.conv_k(channel_out)
        V = self.conv_v(channel_out)
        fuse = torch.matmul(Q, K)
        b, c, size_x, size_y, size_z = fuse.size()
        S = fuse.reshape(b, c, size_x * size_y * size_z)
        T = self.soft(S)
        W = T.reshape(b, c, size_x, size_y, size_z)
        att_sp = W * V
        att_f = torch.cat([att_sp, x], dim=1)
        out = self.conv_1(att_f)
        return out


# The following is the V_NET
def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features=16):
        super(ContBatchNorm3d, self).__init__(num_features=16)
        self.num_features = num_features

    def forward(self, input):
        self._check_input_dim(input)

        return F.batch_norm(

            input, self.running_mean, self.running_var, self.weight, self.bias,

            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)
        self.bn1 = nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []

    for _ in range(depth):
        layers.append(LUConv(nchan, elu))

    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(2, outChans, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):

        x1 = x
        x = self.conv1(x)
        out = self.bn1(x)

        x16 = torch.cat((x1, x1, x1, x1, x1, x1, x1, x1), 1)

        out = self.relu1(torch.add(out, x16))

        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()

        outChans = 2 * inChans

        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm3d(outChans)

        self.do1 = passthrough

        self.relu1 = ELUCons(elu, outChans)

        self.relu2 = ELUCons(elu, outChans)

        if dropout:
            self.do1 = nn.Dropout3d()

        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)

        self.bn1 = nn.BatchNorm3d(outChans // 2)

        self.do1 = passthrough

        self.do2 = nn.Dropout3d()

        self.relu1 = ELUCons(elu, outChans // 2)

        self.relu2 = ELUCons(elu, outChans)

        if dropout:
            self.do1 = nn.Dropout3d()

        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):

        out = self.do1(x)

        skipxdo = self.do2(skipx)

        out = self.relu1(self.bn1(self.up_conv(out)))

        xcat = torch.cat((out, skipxdo), 1)

        out = self.ops(xcat)

        out = self.relu2(torch.add(out, xcat))

        return out


class OutputTransition(nn.Module):

    def __init__(self, inChans, elu, nll):

        super(OutputTransition, self).__init__()

        self.conv1 = nn.Conv3d(inChans, 1, kernel_size=5, padding=2)


        self.bn1 = nn.BatchNorm3d(1)

        self.conv2 = nn.Conv3d(1, 1, kernel_size=1)

        self.relu1 = ELUCons(elu, 1)

        if nll:

            self.softmax = F.log_softmax

        else:

            self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.relu1(self.bn1(self.conv1(x)))

        out = self.conv2(out)

        out = self.sigmoid(out)

        return out


