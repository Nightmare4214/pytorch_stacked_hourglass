import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual
from task.loss import HeatmapLoss


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim=256, oup_dim=16, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()

        self.nstack = nstack
        self.pre = nn.Sequential(  # ([B, 3, 256, 256])
            Conv(3, 64, 7, 2, bn=True, relu=True),  # ([B, 64, 128, 128])
            Residual(64, 128),  # ([B, 128, 128, 128])
            Pool(2, 2),  # ([B, 128, 64, 64])
            Residual(128, 128),  # ([B, 128, 64, 64])
            Residual(128, inp_dim)  # ([B, 256, 64, 64])
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack - 1)])
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        x = imgs.permute(0, 3, 1, 2)  # x of size 1,3,inpdim,inpdim
        x = self.pre(x)  # ([B, 256, 64, 64])
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)  # ([B, 256, 64, 64])
            feature = self.features[i](hg)  # ([B, 256, 64, 64])
            preds = self.outs[i](feature)  # ([B, 16, 64, 64])
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)  # ([B, nstack, 16, 64, 64])

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:, i], heatmaps))  # ([16, ])
        combined_loss = torch.stack(combined_loss, dim=1)  # ([16, nstack])
        return combined_loss
