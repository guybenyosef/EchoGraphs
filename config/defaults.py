"""Configs."""
from fvcore.common.config import CfgNode
from config import custom_config
import os
import sys
import argparse
import datetime
from typing import Dict
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# Dataset [echonet40, echonet_cycle]
_C.TRAIN.DATASET = "echonet40"

# Number of keypoints [40]
_C.TRAIN.NUM_KPTS = 40

# Input size at training
_C.TRAIN.INPUT_SIZE = 112

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 16

# Number of epochs.
_C.TRAIN.EPOCHS = 100000

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_INTERVAL = 1

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

_C.TRAIN.OVERFIT = False

_C.TRAIN.WEIGHTS = None


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False

# option to overfit (debug mode)
_C.DATA_LOADER.OVERFIT = False

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model name [CNNGCN, CNNGCNV3,CNN_GCN_EFV2,EFNet,EFKptsNetSD]
_C.MODEL.NAME = "CNNGCN"

# Model backbone [resnet50, mobilenet2, r3d_18]
_C.MODEL.BACKBONE = "mobilenet2"

# Loss function [L2, WeightedCrossEnt, CrossEnt, L2Seq]
_C.MODEL.LOSS_FUNC = ["L2"]


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# data root (e.g., for RAF-DB)
_C.DATA.PATH_TO_DATA_DIR = ""

# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()

# Augmentation probability (Portion of augmented images during training: [0,1])
_C.AUG.PROB = 0.90

# Augmentation type [strongkeep,strong_echo_cycle]
_C.AUG.METHOD = "strongkeep"

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()


# optimizer. [Adam, Adam_rul]
_C.SOLVER.OPTIMIZER = 'Adam'

# Base learning rate.
_C.SOLVER.BASE_LR = 0.0001

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.EVAL = CfgNode()

# Input size at training
_C.EVAL.INPUT_SIZE = 112
# weight file for the model, must be specified to evaluate a trained model
_C.EVAL.WEIGHTS = ''
# Batch size during evaluation
_C.EVAL.BATCH_SIZE = 4
# Number of data loader workers per training process.
_C.EVAL.NUM_WORKERS = 8
# how many example images to plot
_C.EVAL.EXAMPLES_TO_PLOT = 10
# mode to execute the evaluation ['normal', 'sliding window']
_C.EVAL.MODE ='normal'
# dataset to be used for evaluation in normal(!) mode [echonet40, echonet_cycle]
_C.EVAL.DATASET = "echonet40"

# ---------------------------------------------------------------------------- #
# Inference options
# ---------------------------------------------------------------------------- #
_C.INF = CfgNode()
# specifies whether you want to do inference on entire folder or single images
# ["folder_sequence","single_image","single_sequence","folder_image"]
_C.INF.MODE = 'folder_sequence'
# weight file for the model, must be specified to run inference with a trained model
_C.INF.WEIGHTS = ''
# input file or folder for the images
_C.INF.INPUT = ''
# output folder to save the output images
_C.INF.OUTPUT = ''

# ---------------------------------------------------------------------------- #
# MISC options
# ---------------------------------------------------------------------------- #
# Seed
_C.SEED = 1234

# Number of frames (1 for single, >1 for multiple) [1,16]
_C.NUM_FRAMES = 2

_C.KPTS_EXTRACTOR_WEIGHTS = None

# Add custom config with default values.
custom_config.add_custom_config(_C)

def convert_to_dict(cfg_node, key_list):
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

def create_tensorboard_run_dict(cfg: CfgNode, params_included=["TRAIN", "DATA", "DATA_LOADER", "MODEL", "AUG", "SOLVER"]) -> Dict:
    run_dict = dict()  # fixme later: assume nesting of depth 2
    for k, v in cfg.items():
        if v is not None:
            if type(v) == CfgNode:
                if k in params_included:
                    for kk, vv in v.items():
                        if not (type(vv) == CfgNode):
                            if isinstance(vv, list):
                                value = str(vv)
                            else: 
                                value = vv
                            run_dict["{}.{}".format(k, kk)] = value
            else:
                run_dict[k] = v

    return run_dict

def assert_and_infer_cfg(cfg):
    # check for specific requirements
    # TRANSCRIPT assertions.
    #assert cfg.AUG.METHOD in ["basic", "bla1", "bla2", "aug_rul"]

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()


def cfg_costum_setup(args):

    cfg = get_cfg()
    cfg.defrost()

    if os.path.exists(args.config_file):
        cfg.merge_from_file(args.config_file)
    else:
        print("No yaml file loaded. Use default cfg file configuration.")
    if hasattr(args, 'opts'):
        if len(args.opts) > 0:
            cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg

def overwrite_eval_cfg(cfg_train:CfgNode,cfg_eval:CfgNode):
    """
    only overwrites EVAL parameters in the config to ensure that model parameters are compatible with the weight file
    PLEASE be aware that all other cfg parameters are overwritten.
    """
    cfg_overwrite = cfg_train.clone()
    cfg_overwrite.EVAL.merge_from_other_cfg(cfg_eval.EVAL)
    return cfg_overwrite

def default_argument_parser(epilog=None):
    """
    Create a parser with a single argument for the config file but also the possibility to add
    additional arguments for overwriting the existing cfg by the command line
    Args:
    config_file (str): path to the config file.
    opts (argument): provide additional options from the command line, it
        overwrites the config loaded from file at a later stage.
    """
    parser = argparse.ArgumentParser(
        description="Provide XXX pipeline."
    )
    parser.add_argument(
        "--config_file",
        dest="config_file",
        help="Path to the config file",
        default="files/configs/Train_single_frame.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def get_run_id() -> str:
    ''' provides ID for experiment based on time-stamp or index '''
    ct = datetime.datetime.now()
    run_id = "{}{:02}{:02}{:02}{:02}".format(ct.month, ct.day, ct.hour, ct.minute, ct.second)
    #run_id = "{}{}{}{}{}{}".format(ct.year - 2000, ct.month, ct.day, ct.hour, ct.minute, ct.second)
    return run_id

if __name__ == '__main__':
    cfg = get_cfg()
    print("hi")
