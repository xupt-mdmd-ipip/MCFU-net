from torch.nn import functional as F
from torch import nn
import torch


class MCFU_net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MCFU_net, self).__init__()

        self.conv1 = DoubleConv1(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv1(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv1(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv1(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = SSAP(256)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(256, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(128, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(64, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(32, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)
        self.CLFM3 = CLFM3(512, 64)
        self.CLFM2 = CLFM2(512, 128)
        self.CLFM1 = CLFM1(512, 256)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        c6 = self.conv6(up_6)
        up_7 = self.up7(self.CLFM1(c4, c5) + c6)
        c7 = self.conv7(up_7)
        up_8 = self.up8(self.CLFM2(c3, c4, c5) + c7)
        c8 = self.conv8(up_8)
        up_9 = self.up9(self.CLFM3(c2, c3, c4, c5) + c8)
        c9 = self.conv9(up_9)
        out = self.conv10(c9)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv1, self).__init__()
        self.conv = nn.Sequential(
            MBC(in_channels=in_ch, out_channels=out_ch),
            MBC(in_channels=out_ch, out_channels=out_ch)
        )

    def forward(self, input):
        return self.conv(input)


class MBC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MBC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))
        self.nonlinearity = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.nonlinearity(self.conv1(x) + self.conv2(x) + self.conv3(x))
        return x


class SSAP(nn.Module):
    def __init__(self, in_channels, r=4, L=32):
        super(SSAP, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=3 * in_channels // 2 + 3 * (in_channels // r) + in_channels,
                                            out_channels=2 * in_channels,
                                            kernel_size=1, bias=False),
                                  nn.BatchNorm2d(2 * in_channels),
                                  nn.ReLU(inplace=True))
        self.conv_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels,
                                  kernel_size=1, bias=False)
        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     nn.AdaptiveAvgPool2d((2, 2)),
                                     nn.AdaptiveAvgPool2d((4, 4)))
        self.after_pooling1 = nn.Sequential(nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels, in_channels // r, 1, bias=False))
        self.after_pooling2 = nn.Sequential(nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels, in_channels // r, 1, bias=False))
        self.after_pooling4 = nn.Sequential(nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels, in_channels // r, 1, bias=False))
        self.SK = SKConv(in_channels=in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        x_1 = self.conv_1x1(x)
        x_pooling1 = F.interpolate(self.after_pooling1(self.pooling[0](x)), size=(h, w), mode='bilinear',
                                   align_corners=False)
        x_pooling2 = F.interpolate(self.after_pooling2(self.pooling[1](x)), size=(h, w), mode='bilinear',
                                   align_corners=False)
        x_pooling4 = F.interpolate(self.after_pooling4(self.pooling[2](x)), size=(h, w), mode='bilinear',
                                   align_corners=False)
        x_aspp = torch.cat([x_1, self.SK(x), x_pooling4, x_pooling2, x_pooling1], dim=1)
        return self.conv(x_aspp) + x_1


class CLFM3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CLFM3, self).__init__()
        self.CLFU1 = CLFU(in_channels, in_channels//2)
        self.CLFU2 = CLFU(in_channels//2, in_channels//4)
        self.CLFU3 = CLFU(in_channels//4, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, *inputs):
        feat1 = self.CLFU1(inputs[-2], inputs[-1])
        feat1 = self.CLFU2(inputs[-3], feat1)
        feat1 = self.CLFU3(inputs[-4], feat1)
        feat2 = self.conv(feat1)
        return feat2


class CLFM2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CLFM2, self).__init__()
        self.CLFU1 = CLFU(in_channels, in_channels//2)
        self.CLFU2 = CLFU(in_channels//2, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, *inputs):
        feat1 = self.CLFU1(inputs[-2], inputs[-1])
        feat1 = self.CLFU2(inputs[-3], feat1)
        feat2 = self.conv(feat1)
        return feat2


class CLFM1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CLFM1, self).__init__()
        self.CLFU = CLFU(in_channels, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, *inputs):
        feat1 = self.CLFU(inputs[-2], inputs[-1])
        feat2 = self.conv(feat1)
        return feat2


class CLFU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CLFU, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.dilation1 = SeparableConv2d_fusion(in_channels, out_channels//2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dilation2 = SeparableConv2d_fusion(in_channels, out_channels//2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

    def forward(self, *inputs):
        feat1 = torch.cat([self.up(inputs[-1]), inputs[-2]], dim=1)
        feat2 = torch.cat([self.dilation1(feat1), self.dilation2(feat1)], dim=1)
        return feat2


class SKConv(nn.Module):
    def __init__(self, in_channels):
        super(SKConv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=1,
                                             kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channels // 2),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=2,
                                             kernel_size=3, padding=2, bias=False),
                                   nn.BatchNorm2d(in_channels // 2),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, dilation=4,
                                             kernel_size=3, padding=4, bias=False),
                                   nn.BatchNorm2d(in_channels // 2),
                                   nn.ReLU(inplace=True))

        self.conv1x1 = nn.Conv2d(in_channels=3 * in_channels // 2, out_channels=in_channels, dilation=1, kernel_size=1,
                                 padding=0)
        self.conv3x3_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 3, dilation=1, kernel_size=3,
                                   padding=1)
        self.conv3x3_2 = nn.Conv2d(in_channels=in_channels // 3, out_channels=3, dilation=1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branches_1 = self.conv1(x)
        branches_2 = self.conv2(x)
        branches_4 = self.conv4(x)

        feat = torch.cat([branches_1, branches_2, branches_4], dim=1)
        feat = self.relu(self.conv1x1(feat))
        feat = self.relu(self.conv3x3_1(feat))
        att = self.conv3x3_2(feat)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        att_4 = att[:, 2, :, :].unsqueeze(1)

        fusion_1_2 = att_1 * branches_1 + att_2 * branches_2 + att_4 * branches_4
        return fusion_1_2


class SeparableConv2d_fusion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d_fusion, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.bn1 = BatchNorm(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = BatchNorm(out_channels)
        self.nonlinearity = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.nonlinearity(x)
        return x


if __name__ == '__main__':
    inputs = torch.ones(1, 3, 256, 256)
    model = MCFU_net(3, 1)
    out = model(inputs)
    model.eval()
    print(out.shape)
