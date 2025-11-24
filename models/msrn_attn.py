# msrn_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- basic blocks ----
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch//reduction, ch),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel,padding=kernel//2)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        maxc, _ = torch.max(x, dim=1, keepdim=True)
        avgc = torch.mean(x, dim=1, keepdim=True)
        m = torch.cat([maxc, avgc], dim=1)
        return x * self.sig(self.conv(m))

# ---- shared encoder-decoder (used at each scale) ----
class ScaleBackbone(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_res=6):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # encoder
        self.enc1 = nn.Sequential(*[ResBlock(base_ch) for _ in range(n_res)])
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(*[ResBlock(base_ch*2) for _ in range(n_res)])
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(*[ResBlock(base_ch*4) for _ in range(n_res)])
        # attention
        self.se = SEBlock(base_ch*4)
        self.spatial_att = SpatialAttention()
        # decoder
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(*[ResBlock(base_ch*2) for _ in range(n_res)])
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(*[ResBlock(base_ch) for _ in range(n_res)])
        self.tail = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x, hidden=None):
        # x: input image at this scale
        feat0 = self.head(x)
        e1 = self.enc1(feat0)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        # incorporate hidden (from coarser scale) if present
        if hidden is not None:
            # upsample hidden and add to bottleneck
            h_up = F.interpolate(hidden, size=b.shape[-2:], mode='bilinear', align_corners=False)
            b = b + h_up
        # attention
        b = self.se(b)
        b = self.spatial_att(b)
        d2 = self.up2(b)
        d2 = d2 + e2  # skip
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = d1 + e1
        d1 = self.dec1(d1)
        out = torch.tanh(self.tail(d1))  # in [-1,1], scale to image space outside
        # return both output and a hidden feature to pass to finer scale
        hidden_for_next = b  # choose bottleneck as recurrent hidden
        return out, hidden_for_next

# ---- multi-scale recurrent wrapper ----
class MSRN_Atten(nn.Module):
    def __init__(self, scales=3, stages=2, base_ch=64):
        super().__init__()
        self.scales = scales
        self.stages = stages
        # shared backbone used at each scale (weight sharing)
        self.backbone = ScaleBackbone(in_ch=3, base_ch=base_ch, n_res=4)

    def forward_single_scale(self, x_scale, hidden):
        # run backbone --> return output and hidden
        out, hidden_next = self.backbone(x_scale, hidden)
        return out, hidden_next

    def forward(self, x):
        # x: Bx3xHxW (full resolution)
        pyramid = [F.interpolate(x, scale_factor=1/(2**i), mode='bilinear', align_corners=False)
                   for i in reversed(range(self.scales))]  # coarse->fine
        hidden = None
        outputs = [None]*self.scales
        # coarse-to-fine (scale recurrent)
        for s_idx, x_s in enumerate(pyramid):
            out_s, hidden = self.forward_single_scale(x_s, hidden)
            outputs[s_idx] = out_s
            # upsample hidden for next scale implicitly happens in backbone
        # outputs list is [coarse,...,fine] -> return fine result (last)
        return outputs[-1]

# quick sanity check
if __name__ == "__main__":
    model = MSRN_Atten(scales=3, stages=2, base_ch=48)
    inp = torch.randn(1,3,256,256)
    out = model(inp)
    print("out", out.shape)  # expect [1,3,256,256]
