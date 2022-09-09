import os
import numpy as np
from utils.utils_plot import plot_inference_movie
import torch
from typing import List, Dict, Tuple

from engine.checkpoints import load_trained_model
from utils.utils_data import load_sequence_as_npy, load_image_as_npy, transform_image_sequence_to_tensor
from config.defaults import cfg_costum_setup, default_argument_parser,overwrite_eval_cfg


########################################### model ##############################################
def load_model_from_weights(weight_file:str = None) -> torch.nn.Module:
    model, _, _ = load_trained_model(weight_file, load_dataset_from_checkpoint=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

def run_inference(model:torch.nn.Module, device:torch.device, output_directory:str, data:np.ndarray = None, name:str ='sample') -> Dict:

    data = [data]
    if model.output_type == 'seq2ef':
        outputs = seq2ef(model, data, device)
        outputs['ef_pred'] = outputs['ef_pred'].cpu().detach().numpy()[0]

    elif model.output_type == 'img2kpts':
        outputs = img2kpts(model, data, device)
        outputs['kpts_pred'] = outputs['kpts_pred'].cpu().detach().numpy()
        outputs['imgs'] = outputs['imgs'].cpu().detach().numpy()

    elif model.output_type == 'seq2ef&kpts':
        outputs = seq2ef_kpts(model,data,device)
        outputs['kpts_pred'] = outputs['kpts_pred'].cpu().detach().numpy()[0]
        outputs['ef_pred'] = outputs['ef_pred'].cpu().detach().numpy()[0]

    elif model.output_type == 'seq2ef&kpts&sd':
        outputs = seq2ef_kpts_sd(model, data, device)
        outputs['kpts_pred'] = outputs['kpts_pred'].cpu().detach().numpy()[0]
        outputs['ef_pred'] = outputs['ef_pred'].cpu().detach().numpy()[0]
        outputs['sd_pred'] = outputs['sd_pred'].cpu().detach().numpy()[0]
    else:
        raise NotImplementedError("Forward method to model type {} is not supported..".format(model.output_type))

    anim = plot_inference_movie(outputs['imgs'],outputs['kpts_pred'],input_size=512,metric_name='Name',value = name)
    output_filname = name+".gif"
    out_directory = output_directory
    gifname = os.path.join(out_directory, output_filname)
    anim.save(gifname, writer='imagemagick', fps=10)
    return outputs


def seq2ef(model: torch.nn, data: List, device: torch.device) -> Dict:

    imgs = data[0].to(device)
    ef_pred = torch.squeeze(model(imgs), 1)
    outputs = {"ef_pred": ef_pred, "imgs": imgs}

    return outputs

def img2kpts(model: torch.nn, data: List, device: torch.device) -> Dict:

    imgs = data[0].to(device)
    model.to(device)
    kpts_pred = model(imgs)
    outputs = {"kpts_pred": kpts_pred, "imgs": imgs}

    return outputs

def seq2ef_kpts(model: torch.nn, data: List, device: torch.device) -> Dict:

    imgs= data[0].to(device)
    ef_pred, kpts_pred = model(imgs)
    ef_pred = torch.squeeze(ef_pred, 1)
    batch_size = kpts_pred.shape[0]
    kpts_pred = torch.reshape(kpts_pred, (batch_size, 40, 2, 2))
    outputs = {"kpts_pred": kpts_pred, "ef_pred": ef_pred, "imgs": imgs}

    return outputs

def seq2ef_kpts_sd(model: torch.nn, data: List, criterion: torch.nn, device: torch.device) -> Dict:

    imgs = data[0].to(device)
    ef_pred, kpts_pred, sd_pred = model(imgs)
    ef_pred = torch.squeeze(ef_pred, 1)
    batch_size = kpts_pred.shape[0]
    kpts_pred = torch.reshape(kpts_pred, (batch_size, 40, 2, 2))
    outputs = {"kpts_pred": kpts_pred, "ef_pred": ef_pred, "sd_pred": sd_pred, "imgs": imgs}

    return outputs

########################################### load ##############################################

def get_filenames_from_folder(image_folder:str) -> List:
    image_list = []
    for (dirpath,dirnames,filenames) in os.walk(image_folder):
        image_list =[dirpath+name for name in filenames]
    return image_list

if __name__ == '__main__':
    args = default_argument_parser()
    cfg_eval = cfg_costum_setup(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    weights = cfg_eval.INF.WEIGHTS
    mode = cfg_eval.INF.MODE
    output_directory =cfg_eval.INF.OUTPUT
    input = cfg_eval.INF.INPUT

    model = load_model_from_weights(weights)

    if 'single' in mode:
        file = input
        if not os.path.exists(file) & os.path.isfile(file):
            raise FileNotFoundError(file)

        if mode == 'single_image':
            image = load_image_as_npy(file)
            tensor = transform_image_sequence_to_tensor(image,device)

        elif mode == 'single_sequence':
            sequence = load_sequence_as_npy(file)
            tensor = transform_image_sequence_to_tensor(sequence,device)
        else:
            raise NotImplementedError("Mode {} is not supported..".format(mode))

        run_inference(model,device,output_directory,tensor, file.split('/')[-1][:-4])

    elif 'folder' in mode:
        if not os.path.isdir(input):
            raise 'Path is not a directory'

        image_files = get_filenames_from_folder(input)
        for file in image_files:
            if not os.path.exists(file):
                raise FileNotFoundError(file)
            if mode == 'folder_image':
                image = load_image_as_npy(file)
                tensor = transform_image_sequence_to_tensor(image,device)

            elif mode == 'folder_sequence':
                sequence = load_sequence_as_npy(file)
                if len(sequence[0]) > 100:
                    sequence = sequence[:100]
                tensor = transform_image_sequence_to_tensor(sequence,device)
            else:
                raise NotImplementedError("Mode {} is not supported..".format(mode))

            print(file.split('/')[-1][:-4])
            run_inference(model,device,output_directory,tensor, file.split('/')[-1][:-4])

    else:
        raise NotImplementedError("Mode {} is not supported..".format(mode))
