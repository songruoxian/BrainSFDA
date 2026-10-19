import itertools
import os
import argparse
import random
import logging
from functools import partial
import numpy as np
import openpyxl
import torch
import torch.nn as nn
import yaml
from torch import optim as optim
from tensorboardX import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--target", type=str, default='NYU')
    parser.add_argument("--source", type=str, default='UM')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--source_training_rate", type=float, default=0.7)
    parser.add_argument("--lamda_con", type=float, default=1.0)
    parser.add_argument("--lamda_dis", type=float, default=1.0)
    parser.add_argument("--kernel_number", type=int, default=8)
    parser.add_argument("--meta_length", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--cc200_nodes", type=int, default=168)
    parser.add_argument("--hcp360_nodes", type=int, default=253)
    parser.add_argument("--schaefer400_nodes", type=int, default=186)
    parser.add_argument("--schaefer400_nodes_gq", type=int, default=278)
    parser.add_argument("--original_nodes", type=int, default=116)
    parser.add_argument("--original_nodes_cc200", type=int, default=200)
    parser.add_argument("--original_nodes_hcp360", type=int, default=360)
    parser.add_argument("--original_nodes_schaefer400", type=int, default=400)
    
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


import torch.optim as optim


def create_optimizer_fix(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer
