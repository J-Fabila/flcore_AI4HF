import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DownscalingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownscalingBlock, self).__init__()
        mid_channels = out_channels
        self.conv_reduction = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
    
    def forward(self, x):
        return self.conv_reduction(x)

class UpscalingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpscalingBlock, self).__init__()
        mid_channels = in_channels // 2
        self.augmentation = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.convolution = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, res):
        x = self.augmentation(x)

        diffZ = res.size()[2] - x.size()[2]
        diffY = res.size()[3] - x.size()[3]
        diffX = res.size()[4] - x.size()[4]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2,
                       diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([res, x], dim=1)
        return self.convolution(x)

class InitialLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialLayer, self).__init__()
        self.initial = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.initial(x)

class FinalLayer(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(FinalLayer, self).__init__()
        self.final = nn.Conv3d(in_channels, n_classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.final(x)

class UNet(nn.Module):
    def __init__(self, params):
        super(UNet, self).__init__()
        pow_2 = [2**i for i in range(6, 15)]
        self.depth = params.UNet_depth
        bilinear = params.UNet_bilinear

        ################################################ Encoder
        if params.UNet_custom_shape == "None":
            blocks = [InitialLayer(1, 64)]
            for i in range(self.depth):
                if i + 1 == self.depth and bilinear:
                    blocks.append(DownscalingBlock(pow_2[i], pow_2[i]))
                else:
                    blocks.append(DownscalingBlock(pow_2[i], pow_2[i + 1]))
        else:
            blocks = []
            for layer in params.UNet_custom_shape:
                blocks.append(DownscalingBlock(layer[0], layer[1]))

        self.encoder = nn.Sequential(*blocks)

        ################################################ Decoder
        pow_2 = pow_2[:self.depth + 1]
        pow_2.reverse()
        blocks = []
        if params.UNet_custom_shape == "None":
            for i in range(self.depth):
                if bilinear and i + 1 != self.depth:
                    blocks.append(UpscalingBlock(pow_2[i], pow_2[i + 1] // 2))
                else:
                    blocks.append(UpscalingBlock(pow_2[i], pow_2[i + 1]))
            blocks.append(FinalLayer(64, params.n_classes))
        else:
            for layer in params.UNet_custom_shape:
                blocks.append(UpscalingBlock(layer[0], layer[1]))

        self.decoder = nn.Sequential(*blocks)

    def forward(self, x):
        residual = []
        for layer in self.encoder:
            out_temp = layer(x.float())
            residual.append(out_temp)
            x = out_temp
        residual.reverse()

        for i in range(len(self.decoder) - 1):
            x = self.decoder[i](x, residual[i + 1])

        return self.decoder[-1](x)
