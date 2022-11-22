import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_
from layers import ResConv3d,  BasicConv3d, InputTransition, \
    DownTransition, UpTransition, OutputTransition, IAS_Net


class BaselineUNet(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(BaselineUNet, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left = BasicConv3d(in_channels, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_left = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/8
        self.block_4_1_left = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)  # 256, 1/16
        self.block_5_1_left = BasicConv3d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_2_left = BasicConv3d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_4_1_right = BasicConv3d((8 + 8) * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = BasicConv3d((4 + 4) * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = BasicConv3d((2 + 2) * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = BasicConv3d((1 + 1) * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(n_filters, self.n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: b 2 h w d
        ct = x[:, 0, :, :, :]
        ct = torch.unsqueeze(ct, dim=1)
        pt = x[:, 1, :, :, :]
        pt = torch.unsqueeze(pt, dim=1)
        ds0 = self.block_1_2_left(self.block_1_1_left(pt))
        ds1 = self.block_2_2_left(self.block_2_1_left(self.pool_1(ds0)))
        ds2 = self.block_3_2_left(self.block_3_1_left(self.pool_2(ds1)))
        ds3 = self.block_4_2_left(self.block_4_1_left(self.pool_3(ds2)))
        x = self.block_5_2_left(self.block_5_1_left(self.pool_4(ds3)))

        x = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(x), ds3], 1)))
        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), ds2], 1)))
        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), ds1], 1)))
        x = self.block_1_2_right(self.block_1_1_right(torch.cat([self.upconv_1(x), ds0], 1)))

        x = self.conv1x1(x)

        if self.n_cls == 1:
            return torch.sigmoid(x)
        else:
            return F.softmax(x, dim=1)



class VNet(nn.Module):
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()

        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

    def forward(self, x):

        out16 = self.in_tr(x)

        out32 = self.down_tr32(out16)

        out64 = self.down_tr64(out32)

        out128 = self.down_tr128(out64)

        out256 = self.down_tr256(out128)

        out = self.up_tr256(out256, out128)

        out = self.up_tr128(out, out64)

        out = self.up_tr64(out, out32)

        out = self.up_tr32(out, out16)

        out = self.out_tr(out)
        return out



class Proposed(nn.Module):
    def __init__(self, in_channels, n_cls, n_filters):
        super(Proposed, self).__init__()
        self.in_channels = in_channels
        self.n_cls = 1 if n_cls == 2 else n_cls
        self.n_filters = n_filters

        self.block_1_1_left_ct = ResConv3d(1, n_filters, kernel_size=5, stride=1, padding=2)
        self.block_1_2_left_ct = ResConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_3_left_ct = IAS_Net(n_filters, M=3)  
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.pool_1_ct = nn.MaxPool3d(kernel_size=2, stride=2)  # 64, 1/2
        self.block_2_1_left_ct = ResConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left_ct = ResConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_3_left_ct = IAS_Net(2 * n_filters, M=3)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.pool_2_ct = nn.MaxPool3d(kernel_size=2, stride=2)  # 128, 1/4
        self.block_3_1_left_ct = ResConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left_ct = ResConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_3_left_ct = IAS_Net(4 * n_filters, M=3)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.block_4_1_left_ct = ResConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left_ct = ResConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_3_left_ct = IAS_Net(8 * n_filters, M=3)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
    
        self.block_5_1_ct = ResConv3d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_2_ct = ResConv3d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_3_ct = IAS_Net(16 * n_filters, M=3)

        self.DeConv4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_4_1_right_ct = ResConv3d(24 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right_ct = IAS_Net(8 * n_filters, M=3)

        self.DeConv3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right_ct = ResConv3d(12 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right_ct = IAS_Net(4 * n_filters, M=3)

        self.DeConv2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right_ct = ResConv3d(6 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right_ct = IAS_Net(2 * n_filters, M=3)

        self.DeConv1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right_ct = ResConv3d(3 * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right_ct = IAS_Net(n_filters, M=3)

        # pet
        self.block_1_1_left_pet = ResConv3d(1, n_filters, kernel_size=5, stride=1, padding=2)  # 5x5 获得大的感受野
        self.block_1_2_left_pet = ResConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_3_left_pet = IAS_Net(n_filters, M=3)  # 默认为3x3卷积  kernel_size=3,stride=1,padding=1

        self.pool_1_pet = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_2_1_left_pet = ResConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_left_pet = ResConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_3_left_pet = IAS_Net(2 * n_filters, M=3)

        self.pool_2_pet = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_3_1_left_pet = ResConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_left_pet = ResConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_3_left_pet = IAS_Net(4 * n_filters, M=3)

        self.pool_3_pet = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_4_1_left_pet = ResConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_left_pet = ResConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_3_left_pet = IAS_Net(8 * n_filters, M=3)
       
        self.pool_4_pet = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block_5_1_pet = ResConv3d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_2_pet = ResConv3d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_5_3_pet = IAS_Net(16 * n_filters, M=3)

        self.DeConv4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_4_1_right_pet = ResConv3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right_pet = IAS_Net(8 * n_filters, M=3)

        self.DeConv3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right_pet = ResConv3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right_pet = IAS_Net(4 * n_filters, M=3)

        self.DeConv2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right_pet = ResConv3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right_pet = IAS_Net(2 * n_filters, M=3)

        self.DeConv1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right_pet = ResConv3d(2 * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right_pet = IAS_Net(n_filters, M=3)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        ct = x[:, 0, :, :, :].unsqueeze(1)
        pet = x[:, 1, :, :, :].unsqueeze(1)

        # pet
        pt0 = self.block_1_2_left_pet(self.block_1_1_left_pet(pet))  # batch_size*features* 144*144*144
        pt1 = self.block_2_2_left_pet(self.block_2_1_left_pet(self.pool_1(pt0)))  # batch_size*(2*features)* 72*72*72
        pt2 = self.block_3_2_left_pet(self.block_3_1_left_pet(self.pool_2(pt1)))  # batch_size*(4*features)* 36*36*36
        pt3 = self.block_4_2_left_pet(self.block_4_1_left_pet(self.pool_3(pt2)))  # batch_size*(8*features)* 18*18*18
        pt4, _, _ = self.block_5_3_pet(self.block_5_2_pet(self.block_5_1_pet(self.pool_4(pt3))))  # b*(16*features)*9*9*9

        pst4, _, _ = self.block_4_2_right_pet(self.block_4_1_right_pet(torch.cat([self.DeConv4(pt4), pt3], dim=1)))  # b*(8*features)*18*18*18
        pst3, _, _ = self.block_3_2_right_pet(self.block_3_1_right_pet(torch.cat([self.DeConv3(pst4), pt2], dim=1)))  # batch_size*(4*features)* 36*36*36
        pst2, _, _ = self.block_2_2_right_pet(self.block_2_1_right_pet(torch.cat([self.DeConv2(pst3), pt1], dim=1)))  # batch_size*(2*features)* 72*72*72
        pst1, C, S = self.block_1_2_right_pet(self.block_1_1_right_pet(torch.cat([self.DeConv1(pst2), pt0], dim=1)))  # batch_size*features* 144*144*144


        # ct
        ct0 = self.block_1_2_left_ct(self.block_1_1_left_ct(ct))  # batch_size*features* 144*144*144
        ct1 = self.block_2_2_left_ct(self.block_2_1_left_ct(self.pool_1(ct0)))  # batch_size*(2*features)* 72*72*72
        ct2 = self.block_3_2_left_ct(self.block_3_1_left_ct(self.pool_2(ct1)))  # batch_size*(4*features)* 36*36*36
        ct3 = self.block_4_2_left_ct(self.block_4_1_left_ct(self.pool_3(ct2)))  # batch_size*(8*features)* 18*18*18
        ct4, _, _ = self.block_5_3_ct(self.block_5_2_ct(self.block_5_1_ct(self.pool_4(ct3))))

        ds4, _, _ = self.block_4_2_right_ct(self.block_4_1_right_ct(torch.cat([self.DeConv4(ct4), ct3, pst4], dim=1)))  # b*(8*features)*18*18*18
        ds3, _, _ = self.block_3_2_right_ct(self.block_3_1_right_ct(torch.cat([self.DeConv3(ds4), ct2, pst3], dim=1)))  # batch_size*(4*features)* 36*36*36
        ds2, _, _ = self.block_2_2_right_ct(self.block_2_1_right_ct(torch.cat([self.DeConv2(ds3), ct1, pst2], dim=1)))  # batch_size*(2*features)* 72*72*72
        ds1, _, _ = self.block_1_2_right_ct(self.block_1_1_right_ct(torch.cat([self.DeConv1(ds2), ct0, pst1], dim=1)))  # batch_size*features* 144*144*144

        out = self.conv1x1(ds1)  # batch_size*1*144*144*144

        return torch.sigmoid(out)


