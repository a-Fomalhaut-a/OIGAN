import torch
import torch.nn as nn
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import logging
import os
import models.modules.architecture as arch

logger = logging.getLogger('base')

class guide_filter(nn.Module):
    def __init__(self):
        super(guide_filter, self).__init__()
        kernel_m = np.ones([3, 3]) * (1 / 9)
        kernel_m = (torch.from_numpy(kernel_m).to(torch.float)).unsqueeze(0).unsqueeze(0)
        #self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.device = torch.device('cpu')
        self.weight_m = nn.Parameter(data=kernel_m, requires_grad=False).to(self.device)
        self.eps = 0.01
        self.pad = 1

    def pad2(self, data):
        ReplicationPad = nn.ReplicationPad2d(padding=(self.pad, self.pad, self.pad, self.pad))
        out = ReplicationPad(data)
        return out

    def forward(self, I, p):
        mean_I = F.conv2d(self.pad2(I), self.weight_m)
        mean_p = F.conv2d(self.pad2(p), self.weight_m)
        mean_II = F.conv2d(self.pad2(I * I), self.weight_m)
        mean_Ip = F.conv2d(self.pad2(I * p), self.weight_m)
        # 方差
        var_I = mean_II - mean_I * mean_I  # 方差公式
        # 协方差
        cov_Ip = mean_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        # 对a、b进行均值平滑
        mean_a = F.conv2d(self.pad2(a), self.weight_m)
        mean_b = F.conv2d(self.pad2(b), self.weight_m)
        q = mean_a * I + mean_b
        return q
class Get_focus_nopadding(nn.Module):
    def __init__(self,opt):
        super(Get_focus_nopadding, self).__init__()
        kernel_m = np.ones([3, 3]) * (1 / 9)
        kernel_m = (torch.from_numpy(kernel_m).to(torch.float)).unsqueeze(0).unsqueeze(0)
        #self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.device = torch.device('cpu')
        self.weight_m = nn.Parameter(data=kernel_m, requires_grad=False).to(self.device)
        self.gf = guide_filter()
        self.pad = 1

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            xs = x_i.unsqueeze(1)
            ReplicationPad = nn.ReplicationPad2d(padding=(self.pad, self.pad, self.pad, self.pad))
            xss = ReplicationPad(xs)
            x_i_m = F.conv2d(xss, self.weight_m)
            rfm = torch.abs(xs - x_i_m)
            afm = self.gf(xs, rfm)
            afm = afm / torch.max(afm)
            x_list.append(afm)
        x = torch.cat(x_list, dim=1)
        return x

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        #self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.device = torch.device('cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_step):
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        pretrained_dict = torch.load(load_path)
        model_dict = network.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

class TESTModel(BaseModel):
    def __init__(self, opt):
        super(TESTModel, self).__init__(opt)
        self.netG = arch.define_G(opt).to(self.device)
        if opt["pretrain"] == 1:
            self.load()
        self.get_foc_nopadding = Get_focus_nopadding(opt)

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)
            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H_branch, self.fake_H, self.dof_LR = self.netG(self.var_L)
        self.netG.zero_grad()

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['SR_branch'] = self.fake_H_branch.detach()[0].float().cpu()
        out_dict['LR_dof'] = self.dof_LR.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            try:
                self.load_network(load_path_G, self.netG)
            except:
                print('-----------some wrong with mode')
                self.load_network(load_path_G, self.netG)



