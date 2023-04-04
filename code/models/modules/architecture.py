import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import block as B
from ..attention.fusion import BBFM
import numpy as np

# Generator
def define_G(opt, device=None):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    netG = OIGAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                          nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'],
                          norm_type=opt_net['norm_type'],
                          act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    if gpu_ids:
        if device is not None:
            netG = nn.DataParallel(netG.to(device))
        else:
            netG = nn.DataParallel(netG)
    return netG


class guide_filter(nn.Module):
    def __init__(self):
        super(guide_filter, self).__init__()
        kernel_m = np.ones([3,3])*(1/9)
        kernel_m = (torch.from_numpy(kernel_m).to(torch.float)).unsqueeze(0).unsqueeze(0)
        self.weight_m = nn.Parameter(data=kernel_m, requires_grad=False)
        self.eps=0.01
        self.pad=1

    def pad2(self,data):
        ReplicationPad = nn.ReplicationPad2d(padding=(self.pad, self.pad, self.pad, self.pad))
        out = ReplicationPad(data)
        return out

    def forward(self, I,p):
        mean_I=F.conv2d(self.pad2(I), self.weight_m)
        mean_p=F.conv2d(self.pad2(p), self.weight_m)
        mean_II = F.conv2d(self.pad2(I*I), self.weight_m)
        mean_Ip = F.conv2d(self.pad2(I*p), self.weight_m)
        var_I = mean_II - mean_I * mean_I
        cov_Ip = mean_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = F.conv2d(self.pad2(a), self.weight_m)
        mean_b = F.conv2d(self.pad2(b), self.weight_m)
        q = mean_a * I + mean_b
        return q


class Get_focus_nopadding(nn.Module):
    def __init__(self):
        super(Get_focus_nopadding, self).__init__()
        kernel_m = np.ones([3,3])*(1/9)
        kernel_m = (torch.from_numpy(kernel_m).to(torch.float)).unsqueeze(0).unsqueeze(0)
        self.weight_m = nn.Parameter(data=kernel_m, requires_grad=False)
        self.gf=guide_filter()
        self.pad=1

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:,i]
            xs=x_i.unsqueeze(1)
            ReplicationPad = nn.ReplicationPad2d(padding=(self.pad,self.pad, self.pad, self.pad))
            xss = ReplicationPad(xs)
            x_i_m = F.conv2d(xss, self.weight_m)
            rfm=torch.abs(xs-x_i_m)
            afm=self.gf(xs,rfm)
            afm=afm/torch.max(afm)
            x_list.append(afm)
        x = torch.cat(x_list, dim=1)
        return x


class OIGAN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
                 act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(OIGAN, self).__init__()

        n_upscale = int(math.log(upscale, 2))

        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]

        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        self.HR_conv0_new = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1_new = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)), \
                                  *upsampler, self.HR_conv0_new)

        self.get_dof_nopadding = Get_focus_nopadding()

        self.b_fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)

        self.b_concat_1 = B.conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_1 = B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA')

        self.b_concat_2 = B.conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_2 = B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA')

        self.b_concat_3 = B.conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_3 = B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA')

        self.b_concat_4 = B.conv_block(2 * nf, nf, kernel_size=3, norm_type=None, act_type=None)
        self.b_block_4 = B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA')

        self.b_LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            b_upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            b_upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        b_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        b_HR_conv1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        self.b_module = B.sequential(*b_upsampler, b_HR_conv0, b_HR_conv1)

        self.conv_w = B.conv_block(nf, out_nc, kernel_size=1, norm_type=None, act_type=None)

        self.f_concat = B.conv_block(nf * 2, nf, kernel_size=3, norm_type=None, act_type=None)

        self.f_block = B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                              norm_type=norm_type, act_type=act_type, mode='CNA')

        self.f_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.f_HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        #fusion
        self.fu1 = BBFM(channel=64)
        self.fu2 = BBFM(channel=64)
        self.fu3 = BBFM(channel=64)
        self.fu4 = BBFM(channel=64)
        self.fu5 = BBFM(channel=64)

    def forward(self, x):

        x_dof = self.get_dof_nopadding(x)
        x = self.model[0](x)

        x, block_list = self.model[1](x)

        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<ONE
        x_ori = x
        for i in range(5):
            x = block_list[i](x)
        x_fea1 = x

        x_b_fea = self.b_fea_conv(x_dof)
        x_cat_1,x = self.fu1(x_b_fea, x_fea1)   #fusion1

        x_cat_1 = self.b_block_1(x_cat_1)
        x_cat_1 = self.b_concat_1(x_cat_1)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<TWO
        for i in range(5):
            x = block_list[i + 5](x)
        x_fea2 = x

        x_cat_2,x = self.fu2(x_cat_1, x_fea2)   # fusion2

        x_cat_2 = self.b_block_2(x_cat_2)
        x_cat_2 = self.b_concat_2(x_cat_2)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<THREE
        for i in range(5):
            x = block_list[i + 10](x)
        x_fea3 = x

        x_cat_3,x = self.fu3(x_cat_2, x_fea3)   # fusion3

        x_cat_3 = self.b_block_3(x_cat_3)
        x_cat_3 = self.b_concat_3(x_cat_3)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<FOUR
        for i in range(5):
            x = block_list[i + 15](x)
        x_fea4 = x

        x_cat_4,x = self.fu4(x_cat_3,x_fea4)   # fusion4

        x_cat_4 = self.b_block_4(x_cat_4)
        x_cat_4 = self.b_concat_4(x_cat_4)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<short cut
        x = block_list[20:](x)
        x = x_ori + x
        x = self.model[2:](x)   # upsample
        x = self.HR_conv1_new(x)

        x_cat_4 = self.b_LR_conv(x_cat_4)
        # short cut<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        x_cat_4 = x_cat_4 + x_b_fea
        x_branch = self.b_module(x_cat_4)   # upsample

        x_branch_d = x_branch
        x_f_cat,xd = self.fu5(x,x_branch_d) # last fusion
        x_out_branch = self.conv_w(xd)
        ########

        x_f_cat = self.f_block(x_f_cat)
        x_out = self.f_concat(x_f_cat)
        x_out = self.f_HR_conv0(x_out)
        x_out = self.f_HR_conv1(x_out)

        #########
        return x_out_branch, x_out, x_dof

