
import os
import sys
import logging
from logging import handlers
import time
import argparse
import numpy as np
from collections import OrderedDict
import options.options as option
import utils.util as util
from dataset.util import bgr2ycbcr
from dataset import create_dataset, create_dataloader
from models import create_model
from tqdm import tqdm
from score import *
import torch
import pandas as pd
import metrics.LPIPS.dist_model as dm
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

import metrics.LPIPS.dist_model as dm
def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
def count_lpips(img_target, img_gt):
    # LPIPS
    model_LPIPS = dm.DistModel()
    model_LPIPS.initialize(model='net-lin', net='alex', use_gpu=False)
    with torch.no_grad():
        dist = model_LPIPS.forward(im2tensor(img_target), im2tensor(img_gt))
    return dist

GK = Gkernel()

if __name__ == '__main__':
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str,default="options/test/test_spsr_test.json",help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    # config loggers. Before it, the log will not work
    log = Logger(opt['path']['log']+'/test.log', level='info')
    log.logger.info(option.dict2str(opt))
    # Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
    # Test begin
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
        allzb = []
        pathofmodel = opt['path']['pretrain_dir']
        list = os.listdir(pathofmodel)
        log.logger.info("----------------------------------------")
        log.logger.info("test in " + pathofmodel)
        for cont in tqdm(list):
            if cont.split('_')[-1] == 'G.pth':
                opt['path']['pretrain_now'] = cont
                datasave_dir = opt['path']['results_root'] + "/" + opt['path']['pretrain_now'].split('.')[0]
                if not os.path.exists(datasave_dir):
                    os.makedirs(datasave_dir)
                ssim, psnr, lpips = [], [], []
                picname = []
                opt['path']['pretrain_model_G'] = opt['path']['pretrain_dir'] + opt['path']['pretrain_now']
                log.logger.info("model now: " + opt['path']['pretrain_model_G'])
                model = create_model(opt)
                log.logger.info("need_HR: {}".format(need_HR))
                test_bar = tqdm(test_loader)
                log.logger.info("----------------------------------------")
                for data in (test_bar):
                    model.feed_data(data, need_HR=need_HR)
                    img_path = data['LR_path'][0]
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    ssimt, psnrt, lpipst = 0, 0, 0
                    model.test()  # test
                    visuals = model.get_current_visuals(need_HR=need_HR)
                    sr_img = util.tensor2img(visuals['SR'])
                    if need_HR:
                        gt_img = util.tensor2img(visuals['HR'])
                        if "ssim" in opt["zb_order"]:
                            ssimt = GK.calculate_ssim(sr_img, gt_img)
                            ssim.append(ssimt)
                        if "psnr" in opt["zb_order"]:
                            psnrt = GK.calculate_psnr(sr_img, gt_img)
                            psnr.append(psnrt)
                        if "lpips" in opt["zb_order"]:
                            lpipst = np.array(
                                count_lpips(sr_img, gt_img).detach().cpu()).squeeze().squeeze().squeeze().squeeze()
                            lpips.append(lpipst)
                        test_bar.desc = "ssim:{:.3f} psnr:{:.3f} lpips:{:.3f}".format(
                            ssimt, psnrt, lpipst)
                    # save images
                    picname.append(img_name)
                    if opt['save_pic']:
                        savedirname = 'pic'
                        if opt['path']['savedirname']:
                            savedirname = opt['path']['savedirname']
                        if not os.path.exists(datasave_dir + '/' + savedirname):
                            os.makedirs(datasave_dir + '/' + savedirname)
                        save_img_path = os.path.join(datasave_dir + '/' + savedirname, img_name + '.png')
                        util.save_img(sr_img, save_img_path)
                if need_HR:
                    if opt['save_pic_excel']:
                        order = opt['zb_order']
                        sheet = []
                        sheet.append(picname)
                        colname = []
                        colname.append('model_name')
                        if "ssim" in opt["zb_order"]:
                            sheet.append(ssim)
                            colname.append('ssim')
                        if "psnr" in opt["zb_order"]:
                            sheet.append(psnr)
                            colname.append('psnr')
                        if "lpips" in opt["zb_order"]:
                            sheet.append(lpips)
                            colname.append('lpips')
                        df = pd.DataFrame(sheet).T
                        df.to_excel(datasave_dir + '/pic_metric.xlsx', index=False, header=colname)
                    # comput mean
                    ssimmean, psnrmean, lpipsmean = 0, 0, 0
                    if "ssim" in opt["zb_order"]:
                        ssimmean = np.mean(ssim)
                    if "psnr" in opt["zb_order"]:
                        psnrmean = np.mean(psnr)
                    if "lpips" in opt["zb_order"]:
                        lpipsmean = np.mean(lpips)
                    log.logger.info('ssim {} and lpips {} with psnr {}'.format(ssimmean, lpipsmean, psnrmean))
                    zbnow = [cont, ssimmean, psnrmean, lpipsmean]
                    allzb.append(zbnow)
        if need_HR:
            df = pd.DataFrame(allzb)
            df.to_excel(opt['path']['results_root'] + '/model_metric_mean.xlsx', index=False
                        , header=["name", "ssim", "psnr", "lpips"])
        log.logger.info('End of testing.')
