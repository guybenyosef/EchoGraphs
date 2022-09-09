import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
import argparse
from typing import List


def load_optimizer(method_name: str,
                   parameters: List[nn.Module],
                   learningrate: float,
                   momentum: float = 0.99,
                   weight_decay: float = 0  # 1e-5
) -> torch.optim.Optimizer:

    if method_name == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=learningrate, momentum=momentum, weight_decay=weight_decay)
    elif method_name == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=learningrate, betas=(0.9, 0.999), weight_decay=weight_decay)
    else:
        raise NotImplementedError("Optimizer method is currently not implemented..")

    print("Using {} optimizer, lr={}..".format(method_name, learningrate))
    return optimizer

# TODO:
def load_scheduler(
        args: argparse.Namespace,
        optimizer: torch.optim.Optimizer, batches: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULERS = {
        'step': StepLR(optimizer, args.lr_stepsize, args.gamma),
        'multi_step': MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma),
        'cosine': CosineAnnealingLR(optimizer, batches * args.epochs, eta_min=1e-6),
        None: None
    }
    return SCHEDULERS[args.scheduler]