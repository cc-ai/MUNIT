[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# Visualizing Climate Change - MUNIT

### License

This repo contains the code adapted from [MUNIT](https://github.com/NVlabs/MUNIT) for the needs of the [VICC project](https://github.com/cc-ai/kdb). 
It keeps the same CC BY-NC-SA 4.0 license

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

### Dependency


pytorch, yaml, comet_ml to visualize the results


The code base was developed using Anaconda with the following packages.
```
conda install pytorch=0.4.1 torchvision cuda90 -c pytorch
conda install -y -c anaconda pip
conda install -y -c anaconda pyyaml
pip install -r ./requirements.txt
```

### Testing the demo

First, download the [pretrained models](https://drive.google.com/open?id=1cSxke52PcYV00mzKjai3JSIAXN82w2O6) and put them in `models` folder.

Run the following command to translate a folder of non flooded image to it's flooded version:
``` 
   python test.py --config ../configs/config.yaml --checkpoint models/gen_00370000.pt --input ../input_folder/ --output_folder output_folder/ --style ../Style_Image/style_image.jpg
```

It is possible to control the style of output using an example style_image. The results are stored in `outputs` folder.

# For Developers

### Training

1. Generate the dataset you want to use. The data we use is not currently available for public re-use. Please contact us if you want more informations.

2. Setup the yaml file. Check out `configs/config_HD.yaml`. Currently we only support list-based dataset organization, check out the parameter: data_list. In the `yaml` config file, for each domain and for `train` and `test`, specify the path to the folder containing the images and a `txt` file listing the images in the folder to be considered for training/testing. E.g.  
    Example trainA.txt.

    ```
    ./images_A/dyzWTle6XGPRvdZiFqwMLQ.jpg
    ./images_A/sGXE_tiBOjqtidWK_VgUOA.jpg
    ./images_A/gxv9RvgHZi_XF11EbJ4Opw.jpg
    ./images_A/QV_vTfZC2JSs69XzsbmqXA.jpg
    ```

3. Start training: in 256x256

    ```
    python train.py --config configs/config_256.yaml
    ```

##### Continue training

Use `--resume` with the same `output_path` AND for now, for some reason, you need to use the **same name** for the config file.

#### How to use comet_ml ?

Create a file at `MUNIT/scripts/.comet.config` (which is ignored by git):

```
[comet]
api_key=YOUR-API-KEY
workspace=YOUR-WORKSPACE
project_name=THE-PROJECT
```

For more information, [see docs](https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables)

## Experiments run

https://docs.google.com/spreadsheets/d/1Csdi2B-LJPChLwO1ng4i2sjPgQxrNRlempa05o3a7og/edit?usp=sharing

### How to reproduce an experiment ?
- Open the link to the comet-ML experiment
- Go to Hyper Parameters to find the git_hash
- Go to assets and download the config.yaml
- Roll back to github version corresponding to the git_hash [useful docs](https://githowto.com/getting_old_versions)

If one wants to use the checkpoints, they can be found in the output path written in hyperparameters. For some reasons they might have been moved to a commonly used folder: ccai/checkpoints/archive_XX 

## Branches

- Master is the main branch, code that is meant to be deployed on the https://climatechangeai.org
- feature/cocoStuff_merged_logits is a branch where we merged several classes of cocostuff so that we have a semantic segmentation consistency that is able to detect water properly. See utils.py and the assignment_dir function. 
- feature/highres is an attempt of using pix2pix upsampling in our settings with some major modifications.
- feature/SpadeResBlock replaces the ResBlocks with SpadeResBlocks. The AdaIN coefficients are not computed from a style image but from the semantic segmentation of the image to be translated. 
