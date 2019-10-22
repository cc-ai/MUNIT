"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
# Package to log experiment results online see: https://www.comet.ml/
from comet_ml import Experiment
comet_exp = Experiment()

# Utils function and dataloader
from utils import (
    get_all_data_loaders,
    prepare_sub_folder,
    write_html,
    write_loss,
    get_config,
    get_scheduler,
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
import tensorboardX
import shutil

# Parse the different arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/edges2handbags_folder.yaml",
    help="Path to the config file.",
)
parser.add_argument("--output_path", type=str, default=".", help="outputs path")
parser.add_argument("--ckpt_path", type=str, default=".", help="ckpt_path")
parser.add_argument("--ckpt_path_HD", type=str, default=".", help="ckpt_path_HD")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--trainer", type=str, default="MUNIT", help="MUNIT|UNIT")
parser.add_argument("--git_hash", type=str, default="no-git-hash", help="output of git log --pretty=format:'%h' -n 1")

opts = parser.parse_args()

if comet_exp is not None:
    comet_exp.log_asset(file_data=opts.config, file_name="config.yaml")
    comet_exp.log_parameter("git_hash", opts.git_hash)
    
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config["max_iter"]
display_size = config["display_size"]
config["vgg_model_path"] = opts.output_path

# Setup model and data loader
if opts.trainer == "MUNIT":
    trainer = MUNIT_Trainer(config)
elif opts.trainer == "UNIT":
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")

# Send model to cuda
trainer.cuda()

# Define natural images dataloader for domain A and B
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(
    config
)

# If semantic consistency is used, define a dataloader with binary masks
if config["semantic_w"] > 0:
    train_loader_a_w_mask = get_data_loader_mask_and_im(
        config["data_list_train_a"],
        config["data_list_train_a_seg"],
        config["batch_size"],
        True,
        new_size=config["new_size"],
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
        height=config["crop_image_height"],
        width=config["crop_image_width"],
        num_workers=config["num_workers"],
        crop=True,
    )

# If Synthetic images are used, define a dataloader with paired images
if config["synthetic_frequency"] > 0:
    synthetic_loader = get_synthetic_data_loader(
        config["data_list_train_a_synth"],
        config["data_list_train_b_synth"],
        config["data_list_train_b_seg_synth"],
        config["batch_size"],
        True,
        new_size=config["new_size"],
        height=config["crop_image_height"],
        width=config["crop_image_width"],
        num_workers=config["num_workers"],
        crop=True,
    )

# If FID is to be evaluated to monitor the training, define FID dataloader
if config["eval_fid"] > 0:
    fid_loader   = get_fid_data_loader(
        config["data_list_fid_a"],
        config["data_list_fid_b"],
        config["batch_size_fid"],
        train=False,
        new_size = config["new_size"],
        num_workers=config["num_workers"]
    )
    get_inception_metrics = prepare_inception_metrics(inception_moment=config["inception_moment_path"],parallel=False)

# Define the dataloader that will be used to display intermediate results (train and test on both domain)
train_display_images_a = torch.stack(
    [train_loader_a.dataset[i] for i in range(display_size)]
).cuda()
train_display_images_b = torch.stack(
    [train_loader_b.dataset[i] for i in range(display_size)]
).cuda()
test_display_images_a = torch.stack(
    [test_loader_a.dataset[i] for i in range(display_size)]
).cuda()
test_display_images_b = torch.stack(
    [test_loader_b.dataset[i] for i in range(display_size)]
).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(
    os.path.join(opts.output_path + "/logs", model_name)
)
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

print('Checkpoint directory :',checkpoint_directory) 

# Copy config file to output folder
shutil.copy(
    opts.config, os.path.join(output_directory, "config.yaml")
)  

# Start training
iterations = (
    trainer.resume(opts.ckpt_path, hyperparameters=config) if opts.resume else 0
)

# train_G1 is a boolean set to true when training MUNIT only
train_G1 = False

while train_G1:
    for it, ((images_a, mask_a), (images_b, mask_b)) in enumerate(
        zip(train_loader_a_w_mask, train_loader_b_w_mask)
    ):
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        mask_a, mask_b = mask_a.cuda().detach(), mask_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config, comet_exp)

            if (iterations + 1)% config["ratio_disc_gen"] == 0:
                trainer.gen_update(
                    images_a, images_b, config, mask_a, mask_b, comet_exp
                )
            if config["domain_adv_w"] > 0:
                trainer.domain_classifier_update(
                    images_a, images_b, config, comet_exp
                )
            torch.cuda.synchronize()
            
        # If Synthetic images are used
        if config["synthetic_frequency"] > 0:
            if iterations % config["synthetic_frequency"] == 0:
                images_a, images_b, mask_b = next(iter(synthetic_loader))
                mask_a                     = mask_b
                images_a, images_b         = images_a.cuda().detach(), images_b.cuda().detach()
                mask_a, mask_b             = mask_a.cuda().detach(), mask_b.cuda().detach()

                with Timer("Elapsed time in update: %f"):
                    # Main training code
                    trainer.dis_update(images_a, images_b, config, comet_exp)
                    
                    # Call gen_update with synth set to true (different loss) to take into account the pairs
                    trainer.gen_update(
                        images_a, images_b, config, mask_a, mask_b, comet_exp = comet_exp, synth = True
                    )

        # Write images to comet
        if (iterations + 1) % config["image_save_iter"] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(
                    test_display_images_a, test_display_images_b
                )
                train_image_outputs = trainer.sample(
                    train_display_images_a, train_display_images_b
                )
            write_2images(
                test_image_outputs,
                display_size,
                image_directory,
                "test_%08d" % (iterations + 1),
                comet_exp,
            )
            write_2images(
                train_image_outputs,
                display_size,
                image_directory,
                "train_%08d" % (iterations + 1),
                comet_exp,
            )
            
            # Compute FID
            FID = get_inception_metrics(trainer, fid_loader,prints=True, use_torch=False)
            if comet_exp is not None:
                comet_exp.log_metric("FID", FID)
            print('FID =',FID)
            # HTML
            # write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config["image_display_iter"] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(
                    train_display_images_a, train_display_images_b
                )
            write_2images(
                image_outputs,
                display_size,
                image_directory,
                "train_current",
                comet_exp,
            )

        # Save network weights
        if (iterations + 1) % config["snapshot_save_iter"] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit("Finish training")

# In the case we want to train MUNIT's pix2pixHD-like extension
            
# Instantiate dataloader for G2
train_G2 = True

# Use HD dataloader
train_loader_a_w_mask = get_data_loader_mask_and_im_HD(
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

train_loader_b_w_mask = get_data_loader_mask_and_im_HD(
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

# Define the lists of images to display
list_image_HD_a = []
list_image_HD_b = []
list_image_a    = []
list_image_b    = []

for i in range(display_size):

    image_HD_a, mask_HD_a, images_a, mask_a = train_loader_a_w_mask.dataset[i]
    list_image_HD_a.append(image_HD_a)
    list_image_a.append(images_a)

    image_HD_b, mask_HD_b, images_b, mask_b = train_loader_b_w_mask.dataset[i]
    list_image_HD_b.append(image_HD_b)
    list_image_b.append(images_b)

train_display_images_a_HD = torch.stack(list_image_HD_a).cuda()
train_display_images_a    = torch.stack(list_image_a).cuda()
train_display_images_b    = torch.stack(list_image_b).cuda()
train_display_images_b_HD = torch.stack(list_image_HD_b).cuda()

# Load ckpt for the extension only if resume HD is provided 
iteration_G2 = 0
print("opts.ckpt_path_HD" , opts.ckpt_path_HD)
if opts.ckpt_path_HD != ".":
    iteration_G2 = (
    trainer.resume(opts.ckpt_path_HD, hyperparameters=config, HD_ckpt=True) if opts.resume else 0
)

# Train G2 only
while train_G2:
    for it, ((images_HD_a, mask_HD_a, images_a, mask_a), (images_HD_b, mask_HD_b, images_b, mask_b)) in enumerate(
        zip(train_loader_a_w_mask, train_loader_b_w_mask)
    ):
        trainer.update_learning_rate_HD()
        
        # warmup is boolean value 
        # When warming up we push the upsampler towards learning a pixelwise upsampling operation
        warmup = iteration_G2 < 5000 # We could set an hyperparameter here
        
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        mask_a, mask_b     = mask_a.cuda().detach(), mask_b.cuda().detach()

        images_HD_a, images_HD_b = images_HD_a.cuda().detach(), images_HD_b.cuda().detach()
        mask_HD_a, mask_HD_b     = mask_HD_a.cuda().detach(), mask_HD_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_HD_update(images_a, images_HD_a, images_b, images_HD_b, config, comet_exp,
                                  lamda_dis = 1.0-torch.exp(torch.tensor(-0.00001*iteration_G2, device = 'cuda')))
             
            if (iteration_G2 + 1)% config["ratio_disc_gen"] ==0:
                trainer.gen_HD_update(
                    images_a, images_HD_a, images_b, images_HD_b, 
                    config, mask_a, mask_HD_a, mask_b, mask_HD_b, 
                    comet_exp,
                    warmup = warmup, 
                    lambda_dis =  1.0-torch.exp(torch.tensor(-0.00001*iteration_G2, device = 'cuda'))
                )
            torch.cuda.synchronize()

        if (iteration_G2 + 1) % config["image_display_iter"] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample_HD(
                                    train_display_images_a, train_display_images_a_HD, 
                                    train_display_images_b, train_display_images_b_HD
                                )
            write_2images(
                image_outputs,
                display_size,
                image_directory,
                "train_current",
                comet_exp,
            )

        # Save network weights
        if (iteration_G2 + 1) % config["snapshot_save_iter"] == 0:
            print('saved weights')
            
            trainer.save(checkpoint_directory, 
                         iterations = iterations, 
                         iterations_HD = iteration_G2,
                         save_HD =True)

        iteration_G2 += 1
        if iteration_G2 >= max_iter:
            sys.exit("Finish training")


# Train MUNIT and G2 at the same time
train_global = False

# Define the scheduler 
trainer.dis_scheduler = get_scheduler(trainer.dis_opt_global, config, iterations)
trainer.gen_scheduler = get_scheduler(trainer.gen_opt_global, config, iterations)

# Initialize number of it
iteration_global = 0
while train_global:
    for it, ((images_HD_a, mask_HD_a, images_a, mask_a), (images_HD_b, mask_HD_b, images_b, mask_b)) in enumerate(
        zip(train_loader_a_w_mask, train_loader_b_w_mask)
    ):
        trainer.update_learning_rate_global()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        mask_a, mask_b     = mask_a.cuda().detach(), mask_b.cuda().detach()

        images_HD_a, images_HD_b = images_HD_a.cuda().detach(), images_HD_b.cuda().detach()
        mask_HD_a, mask_HD_b     = mask_HD_a.cuda().detach(), mask_HD_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_global_update(images_a, images_HD_a, images_b, images_HD_b, config, comet_exp)
            
            if (iteration_global + 1)% config["ratio_disc_gen_global"] == 0:
                trainer.gen_global_update(
                    images_a, images_HD_a, images_b, images_HD_b, 
                    config, mask_a, mask_HD_a, mask_b, mask_HD_b, 
                    comet_exp
                )
            torch.cuda.synchronize()
            
            if (iteration_global + 1) % config["image_display_iter"] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample_HD(
                                        train_display_images_a, train_display_images_a_HD, 
                                        train_display_images_b, train_display_images_b_HD
                                    )
                write_2images(
                    image_outputs,
                    display_size,
                    image_directory,
                    "train_current",
                    comet_exp,
                )

        # Save network weights
        if (iteration_global + 1) % config["snapshot_save_iter"] == 0:
            print('saved weights')
            
            trainer.save(checkpoint_directory, 
                         iterations = iteration_global, 
                         iterations_HD = iteration_global,
                         save_HD =False)
            
            trainer.save(checkpoint_directory, 
                         iterations = iteration_global, 
                         iterations_HD = iteration_global,
                         save_HD =True)

        iteration_global += 1
        if iteration_global >= max_iter:
            sys.exit("Finish training")
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                           
                                            