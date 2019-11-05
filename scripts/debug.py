from comet_ml import Experiment
# comet_exp = Experiment()
from utils import (
    get_all_data_loaders,
    prepare_sub_folder,
    write_html,
    write_loss,
    get_config,
    write_2images,
    Timer,
    get_synthetic_data_loader,
    get_data_loader_mask_and_im,
    get_fid_data_loader,
    get_data_loader_mask_and_im_HD
    )
from inception_utils import prepare_inception_metrics, load_inception_net
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
# import tensorboardX
import shutil
config = get_config('configs/SyntheticAblationSNGAN.yaml')
# ----------------------------------------------------------
# ----- For development purpose only we set is_HD to 1 -----
# ----------------------------------------------------------
config['is_HD'] =1

max_iter = config["max_iter"]
display_size = config["display_size"]

# Setup model and data loader
trainer = MUNIT_Trainer(config)

# Instantiate dataloader for G2
train_G2 = True
train_loader_a_w_mask = get_data_loader_mask_and_im(
    config["data_list_train_a"],
    config["data_list_train_a_seg"],
    config["batch_size"],
    True,
    new_size=config["new_size"],
    new_size_HD=config["new_size_HD"],
    height=config["crop_image_height"],
    width=config["crop_image_width"],
    num_workers=config["num_workers"],
    crop=True,
)

train_loader_b_w_mask = get_data_loader_mask_and_im(
    config["data_list_train_b"],
    config["data_list_train_b_seg"],
    config["batch_size"],
    True,
    new_size=config["new_size"],
    new_size_HD=config["new_size_HD"],
    height=config["crop_image_height"],
    width=config["crop_image_width"],
    num_workers=config["num_workers"],
    crop=True,
)