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
from torchvision.transforms import functional as F

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="network configuration file")
parser.add_argument("--input", type=str, help="directory of input images")
parser.add_argument("--output_folder", type=str, help="output image directory")
parser.add_argument("--checkpoint", type=str, help="checkpoint of generator")
parser.add_argument("--style", type=str, default="", help="style image path")
parser.add_argument("--seed", type=int, default=10, help="random seed")
parser.add_argument("--HD_checkpoint", type=str, default=".", help="HD_checkpoint")

parser.add_argument(
    "--synchronized",
    action="store_true",
    help="whether use synchronized style code or not",
)
parser.add_argument(
    "--output_only",
    action="store_true",
    help="whether use synchronized style code or not",
)
parser.add_argument(
    "--output_path",
    type=str,
    default=".",
    help="path for logs, checkpoints, and VGG model weight",
)
opts = parser.parse_args()

# Set the seed value
# torch.manual_seed(opts.seed)
# torch.cuda.manual_seed(opts.seed)

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
    # overwrite entries in the existing state dict
    gen_dict = trainer.gen.state_dict()

    if trainer.gen_state == 1:
        gen_dict.update(state_dict["2"]) 
        trainer.gen.load_state_dict(gen_dict)
        print('Loaded gen ckpt')
except:
    sys.exit("Cannot load the checkpoints")
    
if opts.HD_checkpoint != ".":
    try:
        state_dict = torch.load(opts.HD_checkpoint)
        trainer.gen.localUp.load_state_dict(state_dict["localUp"])
        trainer.gen.localDown.load_state_dict(state_dict["localDown"]) 
        print('Loaded HD ckpt')
    except:
        sys.exit("Cannot load the gen HD checkpoints")



# Send the trainer to cuda
trainer.cuda()
trainer.eval()

# Set param new_size
new_size = config["new_size"]
new_size_HD = config["new_size_HD"]

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
                transforms.RandomCrop((new_size,new_size)),
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

    for jjj in tq.tqdm(range(len(list_non_flooded))):
        
        # Define image path
        path_xa = list_non_flooded[jjj]
        
        #####################################################
        image = Image.open(path_xa).convert("RGB")
        # Resizing to the closest multiple of 4 (pix2pix)
        new_size_hd =512
        ow, oh = image.size
        aspect_ratio = ow/oh
        if aspect_ratio <1:
            ow, oh = new_size_hd,int(round(new_size_hd*oh/ow))
        else:
            ow, oh = int(round(new_size_hd*ow/oh)),new_size_hd
            
        base = 4
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        # print('debugging mask transform 2 size',mask.size)
        # Define Resize HD (multiple of 4) used by the downsampler
        resize_HD = transforms.Resize((h,w))
        # Resize for HD
        image_HD = resize_HD(image)
        
        i, j, h, w = transforms.RandomCrop.get_params(
            image_HD, output_size=(new_size_hd, new_size_hd)
        )
        image_HD = F.crop(image_HD, i, j, h, w)
        
        # Define Resize for G1
        resize    = transforms.Resize((new_size,new_size))
        
        # Resize for G1
        image = resize(image_HD)
       
        to_tensor = transforms.ToTensor()
        image_HD  = to_tensor(image_HD)
        image     = to_tensor(image)

        # print('debugging mask transform 5 size',mask.size)
        # Normalize
        normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        x_a_HD = normalizer(image_HD)
        x_a    = normalizer(image)
        
        x_a = x_a.unsqueeze(0).cuda()
        x_a_HD = x_a_HD.unsqueeze(0).cuda()
        #####################################################
        
        # Downsample
        Downsampled_x_a = trainer.gen.localDown(x_a_HD)
        
        # Extract content and style
        c_a, _ = trainer.gen.encode(x_a, 1)

        # Perform cross domain translation
        x_ab, embedding_x_ab = trainer.gen.decode(c_a, s_b, 2, return_content=True)
        
        x_ab_HD = trainer.gen.localUp(embedding_x_ab + Downsampled_x_a)
        
        # Denormalize .Normalize(0.5,0.5,0.5)...
        outputs = (x_ab_HD + 1) / 2.0

        # Define output path
        path = os.path.join(opts.output_folder, "output{:03d}.jpg".format(jjj))

        # Save image 
        vutils.save_image(outputs.data, path, padding=0, normalize=True)