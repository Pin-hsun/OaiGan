from __future__ import print_function
import argparse
import os
from utils.data_utils import imagesc
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from utils.make_config import load_config
from dotenv import load_dotenv

load_dotenv('.env')


def norm_01(x):
    """
    normalize to 0 - 1
    """
    x = x - x.min()
    x = x / x.max()
    return x


def get_model(epochs, name, dir_checkpoints, device):
    model_path = dir_checkpoints + "{}/{}_model_epoch_{}.pth".format(opt.prj, name, epochs)
    net = torch.load(model_path).to(device)
    return net


def overlap_red(x0, y0, channel):
    y = 1 * y0
    x = 1 * x0
    c = 0
    x[0, y == c] = 0.0 * x[0, y == c]
    c = 2
    x[1, y == c] = 0.0 * x[1, y == c]
    c = 1
    x[2, y == c] = 0.0 * x[2, y == c]
    return x


def overlap_red2(x0, y0, channel):
    y = 1 * y0
    x = 1 * x0
    c = 0
    x[0, y == c] = 1
    x[1, y == c] = 0.0 * x[1, y == c]
    x[2, y == c] = 0.0 * x[1, y == c]
    return x


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='pain', help='name of training dataset')
parser.add_argument('--testset', default=None, help='name of testing dataset')
parser.add_argument('--prj', type=str, default='AttGAN0', help='name of the project')
parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
parser.add_argument('--crop', type=int, default=0)
parser.add_argument('--nepochs', nargs='+', default=[260, 280, 20], help='which checkpoints to be interfered with')
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--port', type=str, default='dummy')
parser.add_argument('--cycle', action='store_true', dest='cycle', default=False)
parser.add_argument('--t2d', action='store_true', dest='t2d', default=True)

opt = parser.parse_args()
opt.prj = opt.dataset + '_' + opt.prj
opt.nepochs = range(int(opt.nepochs[0]), int(opt.nepochs[1]), int(opt.nepochs[2]))
print(opt)


class Pix2PixModel:
    def __init__(self, opt):
        self.opt = opt
        self.dir_checkpoints = os.environ.get('CHECKPOINTS')
        from dataloader.data import DatasetFromFolder as Dataset

        if opt.testset:
            self.test_set = Dataset(os.environ.get('DATASET') + opt.testset + '/test/', opt, mode='test')
        else:
            self.test_set = Dataset(os.environ.get('DATASET') + opt.dataset + '/test/', opt, mode='test')

        if not os.path.exists(os.path.join("result", opt.prj)):
            os.makedirs(os.path.join("result", opt.prj))

        self.seg_model = torch.load(os.environ.get('model_seg')).cuda()
        self.netg_t2d = torch.load(os.environ.get('model_t2d')).cuda()

        self.device = torch.device("cuda:0")

        self.irange = [15, 19, 34, 79, 95, 109, 172, 173, 208, 249, 266]  # pain
        # irange = [102, 109, 128, 190, 193, 235, 252, 276, 318, 335]

    def get_one_output(self, i):
        opt = self.opt
        x = self.test_set.__getitem__(i)
        img = x[0]
        mask = x[1]

        # model
        input = img.unsqueeze(0).to(self.device)
        if not opt.cycle:
            net_g = get_model(epoch, 'netG', self.dir_checkpoints, self.device)
        else:
            net_g = get_model(epoch, 'netG_ab', self.dir_checkpoints, self.device)
        out = net_g(input)
        return img, mask, out



if __name__ == '__main__':
    for epoch in opt.nepochs:
        a_all = []
        b_all = []
        o_all = []

        aseg_all = []
        bseg_all = []
        oseg_all = []

        # for i in [102, 109, 128, 190, 193, 235, 252, 276, 318, 335]:
        for i in irange:  #:
            # test
            x = test_set.__getitem__(i)
            img = x[0]
            mask = x[1]

            # model
            input = img.unsqueeze(0).to(device)
            if not opt.cycle:
                net_g = get_model(epoch, name='netG')
            else:
                net_g = get_model(epoch, name='netG_ab')
            out = net_g(input)

            if opt.crop:
                img = img[:, opt.crop:-opt.crop, opt.crop:-opt.crop]
                mask = mask[:, opt.crop:-opt.crop, opt.crop:-opt.crop]
                out = out[:, :, opt.crop:-opt.crop, opt.crop:-opt.crop]

            # T2D
            if opt.t2d:
                a_t2d = netg_t2d(img.cuda().unsqueeze(0)).detach().cpu()
                out_t2d = netg_t2d(out.cuda()).detach().cpu()
            else:
                a_t2d = img.unsqueeze(0).detach().cpu()
                out_t2d = out.detach().cpu()

            # normalize to 0-1
            img = norm_01(img)
            mask = norm_01(mask)
            out = norm_01(out)
            # a_t2d = norm_01(a_t2d)
            # out_t2d = norm_01(out_t2d)
            a_all.append(img.unsqueeze(0))
            b_all.append(mask.unsqueeze(0))
            o_all.append(out.detach().cpu())

            # segmentation
            aseg = seg_model(a_t2d.cuda())[0].detach().cpu().numpy()
            aseg = np.argmax(aseg, 1)[0, ::]
            aseg_all.append(aseg)

            bseg = seg_model(mask.unsqueeze(0).cuda())[0].detach().cpu().numpy()
            bseg = np.argmax(bseg, 1)[0, ::]
            bseg_all.append(bseg)

            oseg = seg_model(out_t2d.cuda())[0].detach().cpu().numpy()
            oseg = np.argmax(oseg, 1)[0, ::]
            oseg_all.append(oseg)

        a_all = make_grid(torch.cat(a_all, 0), nrow=len(irange), padding=0)
        b_all = make_grid(torch.cat(b_all, 0), nrow=len(irange), padding=0)
        o_all = make_grid(torch.cat(o_all, 0), nrow=len(irange), padding=0)

        aover = overlap_red(a_all, np.concatenate(aseg_all, 1), channel=[1, 2])
        bover = overlap_red(b_all, np.concatenate(bseg_all, 1), channel=[1, 2])
        oover = overlap_red(o_all, np.concatenate(oseg_all, 1), channel=[1, 2])

        # difference a-o
        diff = a_all - o_all
        diff[diff <= 0] = 0
        oseg_all = np.concatenate(oseg_all, 1)
        oseg_all = (oseg_all == 1) / 1

        diff = overlap_red(diff, oseg_all, channel=[1])

        # diff[:, oseg_all == 0] = 1

        # diff = aover

        # if not opt.t2d:
        imagesc(torch.cat([a_all, aover, b_all, bover, o_all, oover, diff], 1), show=False,
                save=os.path.join("result", opt.prj, str(epoch) + '.jpg'))
        # else:
        #    imagesc(torch.cat([a_all, b_all, o_all, diff], 1), show=False,
        #            save=os.path.join("result", opt.prj, str(epoch) + '.jpg'))

# USAGE
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset pain --nepochs 0 601 20 --prj cycle_eff_check --direction a_b --cycle
# CUDA_VISIBLE_DEVICES=1 python test.py --dataset pain --nepochs 0 601 20 --prj wseg1000 --direction a_b
# CUDA_VISIBLE_DEVICES=3 python test.py --dataset TSE_DESS --nepochs 0 601 20 --prj up256patchgan --direction a_b
# CUDA_VISIBLE_DEVICES=2 python test.py --dataset TSE_DESS --nepochs 0 601 20 --prj TrySeg100cartilage --direction a_b