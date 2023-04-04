###SEnet codee


from torch import nn
import numpy as np
import torch
import pdb
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction,
                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,
                  kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                  bias=False) ,
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c,1,1)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SELayernog(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayernog, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction,
                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,
                  kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                  bias=False) ,
        )
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.fc(x)
        y=y.view(b, c, h, w)
        return y

class MS_CAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MS_CAM, self).__init__()
        self.GSE=SELayer(channel=channel, reduction=reduction)
        self.SE=SELayernog(channel=channel, reduction=reduction)
        self.sig=nn.Sigmoid()
    def forward(self,x):
        xg=self.GSE(x)
        xn=self.SE(x)
        xgp=xg.repeat(1,1,xn.size()[2],xn.size()[3])
        y=xgp+xn
        ys=self.sig(y)
        return x * ys.expand_as(x)

class BBFM(nn.Module):
    """BBFM in pytorch mode
    Parameters
    Forked network, two-way integration. cat first,add last
    ----------
    channel: int
        Number of channel
    reduction: int
        the numer underline
    """
    def __init__(self, channel, reduction=16):
        super(BBFM, self).__init__()
        self.msc = MS_CAM(channel=channel*2, reduction=reduction)
        self.msa = MS_CAM(channel=channel, reduction=reduction)
    def forward(self,x,y):
        za = torch.add(x, y)
        zc = torch.cat((x, y), dim=1)
        moutc = self.msc(zc)
        mouta = self.msa(za)
        xout = torch.cat((x, (1-x)),dim=1)*moutc
        yout = mouta*(y)
        return xout, yout
