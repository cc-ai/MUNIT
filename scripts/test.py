"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import tqdm as tq
import glob

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="network configuration file")
parser.add_argument("--input", type=str, help="directory of input images")
parser.add_argument("--output_folder", type=str, help="output image directory")
parser.add_argument("--checkpoint", type=str, help="checkpoint of generator")
parser.add_argument("--style", type=str, default="", help="style image path")
parser.add_argument("--seed", type=int, default=10, help="random seed")
parser.add_argument("--save_input", action="store_true",)
parser.add_argument("--output_path",type=str,default=".", help="path for logs, checkpoints, and VGG model weight",)
opts = parser.parse_args()

# Set the seed value
torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

# Create output folder if it does not exist
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
config["vgg_model_path"] = opts.output_path

# Set Style dimension
style_dim = config["gen"]["style_dim"]
trainer = MUNIT_Trainer(config)

# Load the model (here we currently only load the latest model architecture: one single style)
try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen.load_state_dict(state_dict["2"])
except:
    sys.exit("Cannot load the checkpoints")

# Send the trainer to cuda
trainer.cuda()
trainer.eval()

# Set param new_size
new_size = config["new_size"]

# Define the list of non-flooded images
list_non_flooded = glob.glob(opts.input+'*')

# Assert there are some elements inside
if len(list_non_flooded) ==0:
    sys.exit('Image list is empty. Please ensure opts.input ends with a /')

# Inference
with torch.no_grad():
    # Define the transform to infer with the generator
    transform = transforms.Compose(
        [
            transforms.Resize(new_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # Load and Transform the Style Image
    style_image = (
        Variable(transform(Image.open(opts.style).convert("RGB")).unsqueeze(0).cuda())
    )
    # Extract the style from the Style Image
    _, s_b = trainer.gen.encode(style_image, 2)
    
    for j in tq.tqdm(range(len(list_non_flooded))):
        
        # Define image path
        path_xa = list_non_flooded[j]
        
        # Load and transform the non_flooded image
        x_a = Variable(
            transform(Image.open(path_xa).convert("RGB")).unsqueeze(0).cuda()
        )
        if opts.save_input:
            inputs = (x_a + 1) / 2.0
            path = os.path.join(opts.output_folder, "input{:03d}.jpg".format(j))
            vutils.save_image(inputs.data, path, padding=0, normalize=True)
            
        # Extract content and style
        c_a, _ = trainer.gen.encode(x_a, 1)
        
        # Perform cross domain translation
        x_ab = trainer.gen.decode(c_a, s_b, 2)

        # Denormalize .Normalize(0.5,0.5,0.5)...
        outputs = (x_ab + 1) / 2.0

        # Define output path
        path = os.path.join(opts.output_folder, "output{:03d}.jpg".format(j))

        # Save image 
        vutils.save_image(outputs.data, path, padding=0, normalize=True)