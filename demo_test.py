
import argparse, os
import torch
from torch.autograd import Variable
import time
import numpy as np
import scipy.io as io
import hdf5storage


def saveCube(path, cube, bands=np.linspace(400,700,num=31), norm_factor=None):
    hdf5storage.write({u'cube': cube,
                       u'bands': bands,
                       u'norm_factor': norm_factor}, '.',
                       path, matlab_compatible=True)

name='model_80.pth'
data = io.loadmat('NTIRE2022_Test.mat')
rgb = data['data']
band_num=31
# model_input
parser = argparse.ArgumentParser(description="PyTorch sen4to6 Demo")
parser.add_argument("--cuda", action="store_true", default=True, help="use cuda?")
parser.add_argument("--scale", default=2, type=int, help="scale factor")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--model", default=name, type=str, help="model path")
opt = parser.parse_args()
cuda = opt.cuda
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

for i in range(50):
    img_rgb = rgb[:,:,:,i]
    #image_transpose
    img_input = img_rgb.astype(float)
    input=np.transpose(img_input, [2,0,1])
    with torch.no_grad():
        input = Variable(torch.from_numpy(input).float()).view(1, -1, input.shape[1], input.shape[2])
        # model_forward
        if cuda:
            model = model.cuda()
            input = input.cuda()
        else:
            model = model.cpu()
        start_time = time.time()
        out = model(input)
        out_temp = out.cpu()
        out = out_temp[0,:,:,:].permute(1,2,0).numpy().astype(np.float32)
        # save to .mat
        elapsed_time = time.time() - start_time
        print("[{}]: It takes {}s for processing".format(i+1,elapsed_time))
        saveCube('%s%04d%s' % ('output/ARAD_1K_',i + 951, '.mat'), out)


