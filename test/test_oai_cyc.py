import torch
import tifffile as tiff
from utils.data_utils import imagesc
import os, glob

model = torch.load('/media/ExtHDD01/logs/womac4/cyc/0/checkpoints/net_gYX_model_epoch_200.pth').cuda()

l = sorted(glob.glob('/home/ghc/Dataset/paired_images/womac4/full/aup/*'))

#[2943, 6283, 6962, 8226]

x = tiff.imread(l[6283])

imagesc(x)

x = x / x.max()
x = (x - 0.5) * 2
x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().cuda()

o = model(x)['out0'].cpu().detach()

imagesc(o[0,0,:,:].numpy())