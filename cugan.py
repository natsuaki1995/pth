import sys, torch, torch.nn as nn, torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, c, r=8, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c // r, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(c // r, c, 1, 1, 0, bias=bias)
    def forward(self, x):
        z = torch.mean(x, dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.conv2(F.relu(self.conv1(z))))

class UNetConv(nn.Module):
    def __init__(self, i, m, o, se):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(i, m, 3, 1, 0), nn.LeakyReLU(0.1, True),
                                  nn.Conv2d(m, o, 3, 1, 0), nn.LeakyReLU(0.1, True))
        self.seblock = SEBlock(o, r=8, bias=True) if se else None
    def forward(self, x):
        z = self.conv(x)
        return self.seblock(z) if self.seblock is not None else z

class UNet1(nn.Module):
    def __init__(self, i, o, deconv):
        super().__init__()
        self.conv1 = UNetConv(i, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv_bottom = nn.ConvTranspose2d(64, o, 4, 2, 3) if deconv else nn.Conv2d(64, o, 3, 1, 0)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.leaky_relu(self.conv1_down(x1), 0.1)
        x2 = F.leaky_relu(self.conv2_up(self.conv2(x2)), 0.1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = F.leaky_relu(self.conv3(x1 + x2), 0.1)
        return self.conv_bottom(x3)

class UNet2(nn.Module):
    def __init__(self, i, o, deconv):
        super().__init__()
        self.conv1 = UNetConv(i, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv_bottom = nn.ConvTranspose2d(64, o, 4, 2, 3) if deconv else nn.Conv2d(64, o, 3, 1, 0)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(F.leaky_relu(self.conv1_down(x1), 0.1))
        x3 = F.leaky_relu(self.conv2_down(x2), 0.1)
        x3 = F.leaky_relu(self.conv3_up(self.conv3(x3)), 0.1)
        x4 = self.conv4(F.pad(x2, (-4, -4, -4, -4)) + x3)
        x4 = F.leaky_relu(self.conv4_up(x4), 0.1)
        x5 = F.leaky_relu(self.conv5(F.pad(x1, (-16, -16, -16, -16)) + x4), 0.1)
        return self.conv_bottom(x5)

class UpCunet4x(nn.Module):
    def __init__(self, i=3, o=3):
        super().__init__()
        self.unet1 = UNet1(i, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps = nn.PixelShuffle(2)
        self.conv_final = nn.Conv2d(64, 12, 3, 1, padding=0, bias=True)
    def forward(self, x):
        _, _, h0, w0 = x.shape
        x00 = x
        ph, pw = ((h0 - 1) // 2 + 1) * 2, ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (19, 19 + pw - w0, 19, 19 + ph - h0), "reflect")
        x = self.unet1(x)
        x = self.unet2(x) + F.pad(x, (-20, -20, -20, -20))
        x = self.ps(F.pad(self.conv_final(x), (-1, -1, -1, -1)))
        x = x[:, :, : h0 * 4, : w0 * 4]
        return x + F.interpolate(x00, scale_factor=4, mode="nearest")

def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = UpCunet4x(); m.load_state_dict(sd, strict=True)
    return m.eval(), sd
