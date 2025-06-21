import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ----------------------------
# EfficientNetB7 Classification Model
# ----------------------------
class RetinaClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5, l2_reg=1e-4):
        super().__init__()
        self.base = models.efficientnet_b7(weights="EfficientNet_B7_Weights.IMAGENET1K_V1")
        self.base.classifier = nn.Identity()  # Remove the default classifier
        self.dropout = nn.Dropout(dropout_rate)

        self.fc_retino = nn.Linear(2560, 5)  # 5 classes for retinopathy grade
        self.fc_edema = nn.Linear(2560, 3)   # 3 classes for macular edema

        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.base(x)
        x = self.dropout(x)
        retino_out = self.fc_retino(x)
        edema_out = self.fc_edema(x)
        return retino_out, edema_out

    def get_l2_loss(self):
        l2 = 0.
        for param in self.parameters():
            l2 += torch.norm(param, p=2)
        return self.l2_reg * l2


# ----------------------------
# UNet Segmentation Model
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes=6):  # 0 is background
        super().__init__()
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)
