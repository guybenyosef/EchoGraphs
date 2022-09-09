import sys
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple

from engine.forwards import run_forward
from data.USKpts import USKpts
from utils.utils_files import to_numpy, AverageMeter

########################################################
########################################################
# Run single epoch for Train/Validate/Eval
########################################################
########################################################
def train_vanilla(epoch: int,
                  loader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  model: torch.nn.Module,
                  device: torch.device,
                  criterion: torch.nn.Module,
                  prossesID: int = None
                  ) -> Tuple[int, list]:
    """vanilla training"""

    model.train()
    prefix = 'Training'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    losses = {"main": AverageMeter(), "ef": AverageMeter(), "kpts": AverageMeter()}

    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, data in enumerate(loader, 0):

            # ================= Extract Data ==================
            filenames = data[2]

            # =================== forward =====================
            batch_loss, batch_output = run_forward(model, data, criterion, device)

            # =================== backward ====================
            if optimizer is not None:
                optimizer.zero_grad()
                batch_loss["loss"].backward()
                optimizer.step()

            pbar.update()
            for dat_name, dat in batch_output.items():
                to_numpy(dat)

            # accumulate losses:
            losses["main"].update(batch_loss["reported_loss"], len(filenames))
            if "reported_ef_loss" in batch_loss:
                losses["ef"].update(batch_loss["reported_ef_loss"], len(filenames))
            if "reported_kpts_loss" in batch_loss:
                losses["kpts"].update(batch_loss["reported_kpts_loss"], len(filenames))

    return losses


def validate(mode: str,
             epoch: int,
             loader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             device: torch.device,
             criterion: torch.nn.Module,
             prossesID: int = None
             ) -> Tuple[int, list]:
    """validation"""

    model.eval()
    if mode == 'validation':
        prefix = 'Validating'
    elif mode == 'test':
        prefix = 'Testing'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    inputs, outputs = dict(), dict()
    losses = {"main": AverageMeter(), "ef": AverageMeter(), "sd": AverageMeter(), "kpts": AverageMeter()}

    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, data in enumerate(loader, 0):

            # ================= Extract Data ==================
            filenames = data[2]
            
            # =================== forward =====================
            batch_loss, batch_output = run_forward(model, data, criterion, device)

            pbar.update()

            for dat_name, dat in batch_output.items():
                numpy_dat_on_cpu = to_numpy(dat)
                batch_output[dat_name] = numpy_dat_on_cpu

            # accumulate losses:
            losses["main"].update(batch_loss["reported_loss"], len(filenames))
            for loss_name in ["ef", "sd", "kpts"]:
                if "reported_{}_loss".format(loss_name) in batch_loss:
                    losses[loss_name].update(batch_loss["reported_{}_loss".format(loss_name)], len(filenames))
            # if "reported_ef_loss" in batch_loss:
            #     losses["ef"].update(batch_loss["reported_ef_loss"], len(filenames))
            # if "reported_sd_loss" in batch_loss:
            #     losses["sd"].update(batch_loss["reported_sd_loss"], len(filenames))
            # if "reported_kpts_loss" in batch_loss:
            #     losses["kpts"].update(batch_loss["reported_kpts_loss"], len(filenames))

            # accumulate outputs nad inputs:
            for ii, filename in enumerate(filenames):
                inputs[filename] = {"keypoints": batch_output["kpts_gt"][ii] if "kpts_gt" in batch_output else None,
                                    "ef": batch_output["ef_gt"][ii] if "ef_gt" in batch_output else None,
                                    "sd": batch_output["sd_gt"][ii] if "sd_gt" in batch_output else None
                                    }
                outputs[filename] = {"keypoints_prediction": batch_output["kpts_pred"][ii] if "kpts_pred" in batch_output else None,
                                     "ef_prediction": batch_output["ef_pred"][ii] if "ef_pred" in batch_output else None,
                                     "sd_prediction": batch_output["sd_pred"][ii] if "sd_pred" in batch_output else None
                                     }

    return losses, outputs, inputs

########################################
########################################
# Sample dataset
########################################
########################################
def sample_dataset(trainset: USKpts, valset: USKpts, testset: USKpts, overfit: bool, batch_size: int = 8, num_workers: int = 8):
    if overfit:  # sample identical very few examples for both train ans val sets:
        num_samples_for_overfit = 10
        annotated = np.random.choice(np.arange(trainset.num_of_annotated_files), num_samples_for_overfit)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(annotated),
                                                  shuffle=False, pin_memory=True)
        valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(annotated),
                                                 shuffle=False, pin_memory=True)
        print("DATA: Sampling identical sets of {} ANNOTATED examples for train and val sets.. ".format(num_samples_for_overfit))

    else:
        # --- Train: ---
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  num_workers=num_workers)
        # --- Val: ---
        if valset is not None:
            valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                     shuffle=False, pin_memory=True,
                                                     num_workers=num_workers)

    # --- Test: ---
    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True,
                                                num_workers=num_workers)
    else:
        testloader = []

    return trainloader,valloader, testloader



