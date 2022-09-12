import sys
import os
import glob
import torch
import torchvision.transforms as torch_transforms
import cv2
import time
import random
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import load_model
from utils.utils_files import ultrasound_img_load

def pre_process_uls_contours(frames_folder: str, basic_transform: str, batch_size: int, img_paths: List, input_img_size: int = 224) -> torch.Tensor:
    imgs = []
    for im_indx in range(batch_size):
        fname = os.path.join(frames_folder, img_paths[random.randint(0, len(img_paths) - 1)])
        img, _ = ultrasound_img_load(fname)
        img = cv2.resize(img, (input_img_size, input_img_size))
        img = basic_transform(img).reshape(3, input_img_size, input_img_size)
        imgs.append(img)
    imgs = torch.stack(imgs)

    return imgs

def pre_process_uls_contours_v2(batch_size: int, img_paths: List, input_img_size: int) -> torch.Tensor:
    imgs = []
    for im_indx in range(batch_size):
        # Read single frame:
        img = cv2.imread(img_paths[im_indx])

        # Crop according to us scan region: 708 x 708 size
        ignore_margin = int(0.5 * (max(img.shape[0], img.shape[1]) - min(img.shape[0], img.shape[1])))  # 154  # 154*2+708=1016
        img = img[:, ignore_margin: -ignore_margin, :]

        # Reshape to input size:
        img = cv2.resize(img, (input_img_size, input_img_size))

        # normalize:
        img = img / 255
        img = (img - 0.5) / 0.5
        # Transform to torch tensor -- replace with open-vino tensor.
        img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()

        # append frame to batch
        imgs.append(img)

    imgs = torch.stack(imgs)    # replace this with the tesnor type that fits open-vino

    return imgs


if __name__ == '__main__':

    # PARAMS:
    frames_folder = '/mntssd/guy/pleura/frames/'
    frames_folder = '/shared-data5/guy/data/USMultiView/2ch_sir/frames/H8JGA61O.C1/'
    model_names = ['CNNGCNV3', 'CNNGCNV3', 'CNNGCNV3', 'UNet', 'EchoNet']  #'CNNGCNV3Q' # 'CNNGCNV2' # 'VAEGCN', 'CNNGCN' 'UNet'
    #model_names = ['CNNGCN', 'CNNGCN', 'CNNGCN', 'UNet', 'EchoNet']
    backbones = ['mobilenet2', 'resnet18', 'resnet50']#'mobilenet2', 'mobilenet2_quantize']#'resnet18', 'resnet18_quantize', 'mobilenet2', 'mobilenet2_quantize']#, 'shufflenetV2', 'resnet18']#, 'mobilenet2', 'resnet18']#, 'squeezenet1', 'mnasnet1_0', 'resnet50']#, 'mobilenet2', 'resnet18', 'mobilenet3small', 'mobilenet3large']    #['densenet201', 'resnet18','mobilenet']#, 'alexnet', 'resnext101']
    is_gpu = False#False #True #
    check_with_preprocess = False  # False #True #False
    num_kpts = 40 #9
    kpt_channels = 2    # kpts dim
    input_img_size = 112 #224 #128 #256    #128 #224
    batch_size = 16#32 #32  # better to do 32

    for ii, model_name in enumerate(model_names):

        if model_name == 'UNet':
            m = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, pretrained=True)
        elif model_name == 'EchoNet':
            # Model taken from the echonet dynamic repo
            m = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True, aux_loss=False)
            m.classifier[-1] = torch.nn.Conv2d(m.classifier[-1].in_channels, 1, kernel_size=m.classifier[
                -1].kernel_size)  # change number of outputs to 1
        else:
            backbone = backbones[ii]
            m = load_model(model_name=model_name, num_kpts=num_kpts, backbone=backbone, is_gpu=is_gpu)

        if is_gpu:
            m = m.cuda()

        m.eval()
        basic_transform = torch_transforms.Compose([
            torch_transforms.ToTensor(),
            torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        num_loops = 100 #100

        img_paths = glob.glob(os.path.join(frames_folder, "*.png"))

        consts_imgs = torch.rand(size=(batch_size, 3, input_img_size, input_img_size))
        if is_gpu:
            consts_imgs = consts_imgs.cuda()

        # warm up:
        for t in range(100000):
            pass

        start_time = time.time()
        for t in range(num_loops):
            if check_with_preprocess:
                imgs = pre_process_uls_contours(frames_folder, basic_transform, batch_size, img_paths, input_img_size)
                if is_gpu:
                    imgs = imgs.cuda()
            else:
                imgs = consts_imgs

            if model_name == "VAEGCN":
                o = m.forward_kpts(imgs)
            else:
                o = m(imgs)
        end_time = time.time()

        num_forward_runs = num_loops * batch_size
        time_per_frame_SECONDS = ((end_time - start_time) / num_forward_runs)
        time_per_frame_MILISECONDS = time_per_frame_SECONDS * 1000 #10

        model_total_params = sum(p.numel() for p in m.parameters())

        gpu_text = 'CPU'
        if is_gpu:
            gpu_text = 'GPU'
        print("Benchmarking {} model, with {} backbone. Inference time on {} for each frame (size {}x3x{}x{}) is {:.6f} sec".
              format(model_name, backbone, gpu_text, batch_size, input_img_size, input_img_size, time_per_frame_SECONDS))
        # print("Benchmarking {} model, with {} backbone. Inference time on {} for each frame is {:.6f} msec".
        #       format(model_name, backbone, gpu_text, time_per_frame_MILISECONDS))
        print("Number of parameters in the model is {:.2f}M.".format(model_total_params/1000000))


    # Run:
    # $taskset -c 0-7 python test_runtime.py --option

    # Xeon, on my station.
    # Pytorch adjusted to Intel
    # Intel AI analytics toolkit