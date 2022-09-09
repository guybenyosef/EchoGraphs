import os
import torch
from models import load_model
from datasets import load_dataset, datas
import argparse
from typing import List, Dict, Optional, Union, Tuple
from fvcore.common.config import CfgNode
########################################
########################################
# Load/Save models
########################################
########################################
def load_trained_model(weights_filename: str = None, load_dataset_from_checkpoint: bool = True) -> Tuple[torch.nn.Module, CfgNode, datas]:
    model = None
    cfg = None
    ds = None
    if weights_filename is not None and os.path.exists(weights_filename):
        print("loading file %s.." % weights_filename)
        checkpoint = torch.load(weights_filename)
        print('epoch is %d' % checkpoint['epoch'])
        if 'cfg' in checkpoint:
            print(checkpoint['cfg'])
            cfg = checkpoint['cfg']
            if load_dataset_from_checkpoint == True:
                ds = load_dataset(cfg.TRAIN.DATASET, input_transform=None, input_size=cfg.TRAIN.INPUT_SIZE, num_frames=cfg.NUM_FRAMES)
            model = load_model(cfg)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'current_train_loss' in checkpoint:
                print('current_train_loss = %.6f' % checkpoint['current_train_loss'])
            if 'current_val_loss' in checkpoint:
                print('current_val_loss = %.6f' % checkpoint['current_val_loss'])
            if 'best_val_metric' in checkpoint:
                best_metric = checkpoint['best_val_metric']
                if type(best_metric) is dict:
                    for key, val in best_metric.items():
                        print("best_{}_metric = {:.6f}".format(key, val))
                else:
                    print('best_val_metric = %.6f' % best_metric)
            if 'hostname' in checkpoint:
                print('Host name: %s' % checkpoint['hostname'])
            if 'weights_saved_to' in checkpoint:
                print('Saved to: %s' % checkpoint['weights_saved_to'])
    else:
        raise ValueError('path names or checkpoints do not exist..')

    return model, cfg, ds


def save_model(filename: str, epoch: int, model: torch.nn.Module, cfg: CfgNode,
               current_train_loss: float, current_val_loss: float, best_val_metric: float, hostname: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'cfg': cfg,
        'epoch': epoch,
        'best_val_metric': best_val_metric,
        'current_train_loss': current_train_loss,
        'current_val_loss': current_val_loss,
        'hostname': hostname,
        'weights_saved_to': filename,
    }, filename)
    print("model saved to {}".format(filename))
