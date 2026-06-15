import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import deform_conv2d

class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=torch.math.sqrt(5))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.padding = padding

    def forward(self, x):
        offset = self.offset_conv(x)
        out = deform_conv2d(x, offset, self.weight, padding=self.padding)
        return self.relu(self.bn(out))

class SelfAttentionBottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, C, H, W = x.size()
        q = self.query_conv(x).view(b, -1, H*W).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, H*W)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        v = self.value_conv(x).view(b, -1, H*W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, C, H, W)
        return self.gamma * out + x

class DBHead(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.prob = nn.Sequential(
            nn.Conv2d(in_c, in_c//4, 3, padding=1, bias=False), nn.BatchNorm2d(in_c//4), nn.ReLU(True),
            nn.ConvTranspose2d(in_c//4, in_c//4, 2, 2), nn.BatchNorm2d(in_c//4), nn.ReLU(True),
            nn.ConvTranspose2d(in_c//4, 1, 2, 2), nn.Sigmoid())
        self.threshold = nn.Sequential(
            nn.Conv2d(in_c, in_c//4, 3, padding=1, bias=False), nn.BatchNorm2d(in_c//4), nn.ReLU(True),
            nn.ConvTranspose2d(in_c//4, in_c//4, 2, 2), nn.BatchNorm2d(in_c//4), nn.ReLU(True),
            nn.ConvTranspose2d(in_c//4, 1, 2, 2), nn.Sigmoid())

    def forward(self, x):
        prob = self.prob(x)
        thresh = self.threshold(x)
        binary = torch.reciprocal(1 + torch.exp(-50 * (prob - thresh)))
        return prob, thresh, binary

class SADBNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.attention = SelfAttentionBottleneck(512)
        self.in5 = nn.Conv2d(512, 256, 1)
        self.in4 = nn.Conv2d(256, 256, 1)
        self.in3 = nn.Conv2d(128, 256, 1)
        self.in2 = nn.Conv2d(64, 256, 1)
        self.out5 = DeformableConvBlock(256, 64)
        self.out4 = DeformableConvBlock(256, 64)
        self.out3 = DeformableConvBlock(256, 64)
        self.out2 = DeformableConvBlock(256, 64)
        self.head = DBHead(256)

    def forward(self, x):
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.attention(self.layer4(c4))
        in5 = self.in5(c5)
        in4 = self.in4(c4) + F.interpolate(in5, scale_factor=2, mode='nearest')
        in3 = self.in3(c3) + F.interpolate(in4, scale_factor=2, mode='nearest')
        in2 = self.in2(c2) + F.interpolate(in3, scale_factor=2, mode='nearest')
        p5 = F.interpolate(self.out5(in5), scale_factor=8, mode='nearest')
        p4 = F.interpolate(self.out4(in4), scale_factor=4, mode='nearest')
        p3 = F.interpolate(self.out3(in3), scale_factor=2, mode='nearest')
        p2 = self.out2(in2)
        fuse = torch.cat((p5, p4, p3, p2), dim=1)
        return self.head(fuse)
