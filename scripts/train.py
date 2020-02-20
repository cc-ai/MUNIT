"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from comet_ml import Experiment

comet_exp = Experiment(workspace="sunandr", project_name="testing-munit")

from utils import (
    get_all_data_loaders,
    prepare_sub_folder,
    write_loss,
    get_config,
    flatten_opts,
    write_2images,
    Timer,
    get_synthetic_data_loader,
    get_data_loader_mask_and_im,
    get_fid_data_loader,
)
from inception_utils import prepare_inception_metrics, load_inception_net
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
from pathlib import Path

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import os
import sys
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="configs/config256.yaml", help="Path to the config file.",
)
parser.add_argument("--output_path", type=str, default=".", help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--trainer", type=str, default="MUNIT", help="MUNIT|UNIT")
parser.add_argument(
    "--git_hash",
    type=str,
    default="no-git-hash",
    help="output of git log --pretty=format:'%h' -n 1",
)
opts = parser.parse_args()


cudnn.benchmark = True
# Load experiment setting
config = get_config(opts.config)
max_iter = config["max_iter"]
display_size = config["display_size"]
config["vgg_model_path"] = opts.output_path

if comet_exp is not None:
    comet_exp.log_asset(file_data=opts.config, file_name=Path(opts.config))
    comet_exp.log_parameter("git_hash", opts.git_hash)
    comet_exp.log_parameters(flatten_opts(config))
# Setup model and data loader
if opts.trainer == "MUNIT":
    trainer = MUNIT_Trainer(config)
elif opts.trainer == "UNIT":
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()


train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)


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

synthetic_loader = get_synthetic_data_loader(
    config["data_list_train_a_synth"],
    config["data_list_train_b_synth"],
    config["data_list_train_b_seg_synth"],
    config["seg_list_a"],
    config["seg_list_b"],
    config["batch_size"],
    True,
    new_size=config["new_size"],
    height=config["crop_image_height"],
    width=config["crop_image_width"],
    num_workers=config["num_workers"],
    crop=True,
)


if config["eval_fid"] > 0:
    fid_loader = get_fid_data_loader(
        config["data_list_fid_a"],
        config["data_list_fid_b"],
        config["batch_size_fid"],
        train=False,
        new_size=config["new_size"],
        num_workers=config["num_workers"],
    )
    get_inception_metrics = prepare_inception_metrics(
        inception_moment=config["inception_moment_path"], parallel=False
    )

train_display_images_a = torch.stack(
    [train_loader_a.dataset[i] for i in range(display_size)]
).cuda()
train_display_images_b = torch.stack(
    [train_loader_b.dataset[i] for i in range(display_size)]
).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(
    opts.config, os.path.join(output_directory, "config.yaml")
)  # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0


if config["semantic_w"] != 0:
    while True:
        for (
            it,
            ((images_a, mask_a), (images_b, mask_b), (images_as, images_bs, mask_s, sem_a, sem_b),),
        ) in enumerate(zip(train_loader_a_w_mask, train_loader_b_w_mask, synthetic_loader)):
            with Timer("Elapsed time in update s: %f"):
                trainer.update_learning_rate()
                images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
                mask_a, mask_b = mask_a.cuda().detach(), mask_b.cuda().detach()
                images_as, images_bs, mask_s = (
                    images_as.cuda().detach(),
                    images_bs.cuda().detach(),
                    mask_s.cuda().detach(),
                )

                # Main training code
                trainer.dis_update(images_a, images_b, config, comet_exp)
                # Gen update
                if (iterations + 1) % config["ratio_disc_gen"] == 0:
                    trainer.gen_update(images_a, images_b, config, mask_a, mask_b, comet_exp)
                # Domain classifier update
                if config["domain_adv_w"] > 0:
                    trainer.domain_classifier_update(images_a, images_b, config, comet_exp)
                # Domain classifier s,r update
                if (
                    trainer.use_classifier_sr
                    and (iterations + 1) % config["adaptation"]["classif_frequency"] == 0
                ):
                    print(iterations + 1)
                    trainer.domain_classifier_sr_update(
                        images_a,
                        images_b,
                        False,
                        config["adaptation"]["dfeat_lambda"],
                        iterations + 1,
                        comet_exp,
                    )

                # Output domain classifier s,r update
                if (
                    trainer.use_output_classifier_sr
                    and (iterations + 1) % config["adaptation"]["output_classif_freq"] == 0
                ):
                    trainer.output_domain_classifier_sr_update(
                        images_a, images_as, images_b, images_bs, config, iterations + 1, comet_exp,
                    )

                torch.cuda.synchronize()

                # If the number of iteration match the synthetic frequency
                # We sample one example of the synthetic paired dataset
                if config["synthetic_frequency"] > 0:
                    if iterations % config["synthetic_frequency"] == 0:
                        #                   images_as, images_bs = images_a.cuda().detach(), images_b.cuda().detach()
                        #                   mask_s = mask_s.cuda().detach()

                        # Main training code
                        trainer.dis_update(images_as, images_bs, config, comet_exp)
                        # Same mask because we know the area where we want to flood
                        if config["synthetic_seg_gt"] == 0:
                            trainer.gen_update(
                                images_as,
                                images_bs,
                                config,
                                mask_s,
                                mask_s,
                                comet_exp,
                                True,
                                None,
                                None,
                            )
                        else:
                            trainer.gen_update(
                                images_as,
                                images_bs,
                                config,
                                mask_s,
                                mask_s,
                                comet_exp,
                                True,
                                sem_a,
                                sem_b,
                            )
                        if (
                            trainer.use_classifier_sr
                            and (iterations + 1) % config["adaptation"]["classif_frequency"] == 0
                        ):
                            trainer.domain_classifier_sr_update(
                                images_as,
                                images_bs,
                                True,
                                config["adaptation"]["dfeat_lambda"],
                                iterations + 1,
                                comet_exp,
                            )
                    if trainer.train_seg:
                        trainer.segmentation_head_update(
                            images_as,
                            images_bs,
                            sem_a,
                            sem_b,
                            config["adaptation"]["sem_seg_lambda"],
                            comet_exp,
                        )

                # Write images
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

                if (iterations + 1) % config["image_display_iter"] == 0:
                    with torch.no_grad():
                        image_outputs = trainer.sample(
                            train_display_images_a, train_display_images_b
                        )
                    write_2images(
                        image_outputs, display_size, image_directory, "train_current", comet_exp,
                    )

                # Save network weights
                if (iterations + 1) % config["snapshot_save_iter"] == 0:

                    trainer.save(checkpoint_directory, iterations)

                iterations += 1
                if iterations >= max_iter:
                    sys.exit("Finish training")

