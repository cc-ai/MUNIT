"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from comet_ml import Experiment
from networks import AdaINGen, AdaINGen_double, MsImageDis, VAEGen,LocalUpsampler,AdaINGen_double_HD,MsImageDisExtended,MsImageDisDeeper
from utils import (
    weights_init,
    get_model_list,
    vgg_preprocess,
    load_vgg16,
    get_scheduler,
    load_flood_classifier,
    transform_torchVar,
    seg_transform,
    load_segmentation_model,
    decode_segmap,
    domainClassifier,
)
from torch.autograd import Variable
from torchvision import transforms
import torch
import torch.nn as nn
import os
from PIL import Image


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters["lr"]
        self.gen_state = hyperparameters["gen_state"]
        self.guided = hyperparameters["guided"]
        self.newsize = hyperparameters["crop_image_height"]
        self.semantic_w = hyperparameters["semantic_w"] > 0
        self.recon_mask = hyperparameters["recon_mask"] == 1
        self.train_G2_only = hyperparameters["train_G2_only"] == 1
        self.train_global = hyperparameters["train_global"] == 1
        self.dann_scheduler = None
        self.dis_HD_scheduler = None
        self.gen_HD_scheduler = None
        self.gen_global_scheduler = None
        self.dis_global_scheduler = None
        
        if "domain_adv_w" in hyperparameters.keys():
            self.domain_classif = hyperparameters["domain_adv_w"] > 0
        else:
            self.domain_classif = False

        if (self.gen_state == 1 and self.train_G2_only == 1) or (self.gen_state == 1 and self.train_global):
            self.gen = AdaINGen_double_HD(
                hyperparameters["input_dim_a"], hyperparameters["gen"]
            )
        else:
            print("self.gen_state unknown value: or not HD", self.gen_state)

        self.dis_a = MsImageDisDeeper(
            hyperparameters["input_dim_a"], hyperparameters["dis"]
        )  # discriminator for domain a
        
        self.dis_b = MsImageDisDeeper(
            hyperparameters["input_dim_b"], hyperparameters["dis"]
        )  # discriminator for domain b
        
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters["gen"]["style_dim"]

        # fix the noise used in sampling
        display_size = int(hyperparameters["display_size"])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        
        dis_HD_params = list(self.dis_a.D0.parameters()) + \
                        list(self.dis_b.D0.parameters())
        
                        # list(self.dis_a.D0.parameters()) + \
                        # list(self.dis_b.D0.parameters())

        if self.gen_state == 1:
            G1_param =  list(self.gen.enc_style.parameters())    + \
                        list(self.gen.enc1_content.parameters()) + \
                        list(self.gen.enc2_content.parameters()) + \
                        list(self.gen.dec1.parameters()) + \
                        list(self.gen.dec2.parameters()) + \
                        list(self.gen.mlp1.parameters()) + \
                        list(self.gen.mlp2.parameters())
        else:
            print("self.gen_state unknown value:", self.gen_state)
        
        # Network weight initialization
        self.apply(weights_init(hyperparameters["init"]))
        self.dis_a.apply(weights_init("gaussian"))
        self.dis_b.apply(weights_init("gaussian"))

        #         if not self.train_global:
        #             self.dis_opt = torch.optim.Adam(
        #                 [p for p in dis_params if p.requires_grad],
        #                 lr=lr,
        #                 betas=(beta1, beta2),
        #                 weight_decay=hyperparameters["weight_decay"],
        #             )
        #             self.gen_opt = torch.optim.Adam(
        #                 [p for p in G1_param if p.requires_grad],
        #                 lr=lr,
        #                 betas=(beta1, beta2),
        #                 weight_decay=hyperparameters["weight_decay"],
        #             )
        #             self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        #             self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        
        # Load HD Discriminator if needed
        if self.train_global or self.train_G2_only:
            # Define parameter of the local upsampler and local downsampler
            G2_params = list(self.gen.localUp.parameters()) +  list(self.gen.localDown.parameters())
            
            #             # HD DISCRIMINATOR 
            #             self.dis_a_HD = MsImageDis(hyperparameters["input_dim_a"], hyperparameters["dis"])  
            #             self.dis_b_HD = MsImageDis(hyperparameters["input_dim_b"], hyperparameters["dis"])

            #             # List parameters
            #             dis_HD_params = list(self.dis_a_HD.parameters()) + list(self.dis_b_HD.parameters())

            #             # Network weight initialization
            #             self.dis_a_HD.apply(weights_init("gaussian"))
            #             self.dis_b_HD.apply(weights_init("gaussian"))
            
            if not self.train_global:
                # Optimizers
                self.gen_opt_HD = torch.optim.Adam(
                    [p for p in G2_params if p.requires_grad],
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=hyperparameters["weight_decay"],
                )

                # Optimizers
                self.dis_HD_opt = torch.optim.Adam(
                    [p for p in dis_HD_params if p.requires_grad],
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=hyperparameters["weight_decay"],
                )
                self.dis_HD_scheduler = get_scheduler(self.dis_HD_opt, hyperparameters)
                self.gen_HD_scheduler = get_scheduler(self.gen_opt_HD, hyperparameters)
            else:
                # Define the list of generator parameters
                G_global_param = G1_param + G2_params
                
                # Define the list of discriminator parameters
                D_global_param = dis_HD_params # + dis_HD_params
                
                # Global optimizer for the generators
                self.gen_opt_global = torch.optim.Adam(
                    [p for p in G_global_param if p.requires_grad],
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=hyperparameters["weight_decay"],
                )
                
                # Global optimizer for the discriminators
                self.dis_opt_global = torch.optim.Adam(
                    [p for p in D_global_param if p.requires_grad],
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=hyperparameters["weight_decay"],
                )
                
                # Define the global scheduler
                self.gen_global_scheduler = get_scheduler(self.gen_opt_global, hyperparameters)
                self.dis_global_scheduler = get_scheduler(self.dis_opt_global, hyperparameters)
                
                
        # Load VGG model if needed
        if "vgg_w" in hyperparameters.keys() and hyperparameters["vgg_w"] > 0:
            self.vgg = load_vgg16(hyperparameters["vgg_model_path"] + "/models")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        # Load semantic segmentation model if needed
        if "semantic_w" in hyperparameters.keys() and hyperparameters["semantic_w"] > 0:
            self.segmentation_model = load_segmentation_model(
                hyperparameters["semantic_ckpt_path"]
            )
            self.segmentation_model.eval()
            for param in self.segmentation_model.parameters():
                param.requires_grad = False

        # Load domain classifier if needed
        if (
            "domain_adv_w" in hyperparameters.keys()
            and hyperparameters["domain_adv_w"] > 0
        ):
            self.domain_classifier = domainClassifier(256)
            dann_params = list(self.domain_classifier.parameters())
            self.dann_opt = torch.optim.Adam(
                [p for p in dann_params if p.requires_grad],
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=hyperparameters["weight_decay"],
            )
            self.domain_classifier.apply(weights_init("gaussian"))
            self.dann_scheduler = get_scheduler(self.dann_opt, hyperparameters)

    def recon_criterion(self, input, target):
        """
        Compute pixelwise L1 loss between two images input and target
        
        Arguments:
            input {torch.Tensor} -- Image tensor
            target {torch.Tensor} -- Image tensor
        
        Returns:
            torch.Float -- pixelwise L1 loss
        """
        return torch.mean(torch.abs(input - target))

    def recon_criterion_mask(self, input, target, mask):
        """
        Compute a weaker version of the recon_criterion between two images input and target 
        where the L1 is only computed on the unmasked region
        
        Arguments:
            input {torch.Tensor} -- Image (original image such as x_a)
            target {torch.Tensor} -- Image (after cycle-translation image x_aba)
            mask {} -- binary Mask of size HxW (input.shape ~ CxHxW)
        
        Returns:
            torch.Float -- L1 loss over input.(1-mask) and target.(1-mask)
        """
        return torch.mean(torch.abs(torch.mul((input - target), 1 - mask)))

    def forward(self, x_a, x_b):
        """
        Perform the translation from domain A (resp B) to domain B (resp A): x_a to x_ab (resp: x_b to x_ba).
        
        Arguments:
            x_a {torch.Tensor} -- Image from domain A after transform in tensor format
            x_b {torch.Tensor} -- Image from domain B after transform in tensor format
        
        Returns:
            torch.Tensor, torch.Tensor -- Translated version of x_a in domain B, Translated version of x_b in domain A
        """
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        if self.gen_state == 0:
            c_a, s_a_fake = self.gen_a.encode(x_a)
            c_b, s_b_fake = self.gen_b.encode(x_b)
            x_ba = self.gen_a.decode(c_b, s_a)
            x_ab = self.gen_b.decode(c_a, s_b)
        elif self.gen_state == 1:
            c_a, s_a_fake = self.gen.encode(x_a, 1)
            c_b, s_b_fake = self.gen.encode(x_b, 2)
            x_ba = self.gen.decode(c_b, s_a, 1)
            x_ab = self.gen.decode(c_a, s_b, 2)
        else:
            print("self.gen_state unknown value:", self.gen_state)
        self.train()
        return x_ab, x_ba

    def gen_update(
        self, x_a, x_b, hyperparameters, mask_a=None, mask_b=None, comet_exp=None, synth=False
    ):
        """
        Update the generator parameters

        Arguments:
            x_a {torch.Tensor} -- Image from domain A after transform in tensor format
            x_b {torch.Tensor} -- Image from domain B after transform in tensor format
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters 

        Keyword Arguments:
            mask_a {torch.Tensor} -- binary mask (0,1) corresponding to the ground in x_a (default: {None})
            mask_b {torch.Tensor} -- binary mask (0,1) corresponding to the water in x_b (default: {None})
            comet_exp {cometExperience} -- CometML object use to log all the loss and images (default: {None})
            synth {boolean}  -- binary True or False stating if we have a synthetic pair or not 

        Returns:
            [type] -- [description]
        """
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        if self.gen_state == 0:
            # encode
            c_a, s_a_prime = self.gen_a.encode(x_a)
            c_b, s_b_prime = self.gen_b.encode(x_b)
            # decode (within domain)
            x_a_recon = self.gen_a.decode(c_a, s_a_prime)
            x_b_recon = self.gen_b.decode(c_b, s_b_prime)
            # decode (cross domain)
            if self.guided == 0:
                x_ba = self.gen_a.decode(c_b, s_a)
                x_ab = self.gen_b.decode(c_a, s_b)
            elif self.guided == 1:
                x_ba = self.gen_a.decode(c_b, s_a_prime)
                x_ab = self.gen_b.decode(c_a, s_b_prime)
            else:
                print("self.guided unknown value:", self.guided)
            # encode again
            c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
            c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
            # decode again (if needed)
            x_aba = (
                self.gen_a.decode(c_a_recon, s_a_prime)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
            x_bab = (
                self.gen_b.decode(c_b_recon, s_b_prime)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
        elif self.gen_state == 1:
            # encode
            c_a, s_a_prime = self.gen.encode(x_a, 1)
            c_b, s_b_prime = self.gen.encode(x_b, 2)
            # decode (within domain)
            x_a_recon = self.gen.decode(c_a, s_a_prime, 1)
            x_b_recon = self.gen.decode(c_b, s_b_prime, 2)
            # decode (cross domain)
            if self.guided == 0:
                x_ba = self.gen.decode(c_b, s_a, 1)
                x_ab = self.gen.decode(c_a, s_b, 2)
            elif self.guided == 1:
                x_ba = self.gen.decode(c_b, s_a_prime, 1)
                x_ab = self.gen.decode(c_a, s_b_prime, 2)
            else:
                print("self.guided unknown value:", self.guided)

            # encode again
            c_b_recon, s_a_recon = self.gen.encode(x_ba, 1)
            c_a_recon, s_b_recon = self.gen.encode(x_ab, 2)
            # decode again (if needed)
            x_aba = (
                self.gen.decode(c_a_recon, s_a_prime, 1)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
            x_bab = (
                self.gen.decode(c_b_recon, s_b_prime, 2)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
        else:
            print("self.gen_state unknown value:", self.gen_state)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        
        # Synthetic reconstruction loss
        self.loss_gen_recon_synth = self.recon_criterion_mask(x_ab, x_b,mask_b) +\
                                    self.recon_criterion_mask(x_ba, x_a,mask_a)  if synth else 0
        
        if self.recon_mask:
            self.loss_gen_cycrecon_x_a = (
                self.recon_criterion_mask(x_aba, x_a, mask_a)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )
            self.loss_gen_cycrecon_x_b = (
                self.recon_criterion_mask(x_bab, x_b, mask_b)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )
        else:
            self.loss_gen_cycrecon_x_a = (
                self.recon_criterion(x_aba, x_a)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )
            self.loss_gen_cycrecon_x_b = (
                self.recon_criterion(x_bab, x_b)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = (
            self.compute_vgg_loss(self.vgg, x_ba, x_b)
            if hyperparameters["vgg_w"] > 0
            else 0
        )
        self.loss_gen_vgg_b = (
            self.compute_vgg_loss(self.vgg, x_ab, x_a)
            if hyperparameters["vgg_w"] > 0
            else 0
        )

        # semantic-segmentation loss
        self.loss_sem_seg = (
            self.compute_semantic_seg_loss(x_a.squeeze(), x_ab.squeeze(), mask_a)
            + self.compute_semantic_seg_loss(x_b.squeeze(), x_ba.squeeze(), mask_b)
            if hyperparameters["semantic_w"] > 0
            else 0
        )
        # Domain adversarial loss (c_a and c_b are swapped because we want the feature to be less informative
        # minmax (accuracy but max min loss)
        self.domain_adv_loss = (
            self.compute_domain_adv_loss(c_a,c_b, compute_accuracy=False, minimize=False)
            if hyperparameters["domain_adv_w"] > 0
            else 0
        )

        # total loss
        self.loss_gen_total = (
            hyperparameters["gan_w"] * self.loss_gen_adv_a
            + hyperparameters["gan_w"] * self.loss_gen_adv_b
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_a
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_a
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_a
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_b
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_b
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_b
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_a
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_b
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_a
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
            + hyperparameters["semantic_w"] * self.loss_sem_seg
            + hyperparameters["domain_adv_w"] * self.domain_adv_loss
            + hyperparameters["recon_synth_w"] * self.loss_gen_recon_synth
        )

        if comet_exp is not None:
            comet_exp.log_metric("loss_gen_adv_a", self.loss_gen_adv_a)
            comet_exp.log_metric("loss_gen_adv_b", self.loss_gen_adv_b)
            comet_exp.log_metric("loss_gen_recon_x_a", self.loss_gen_recon_x_a)
            comet_exp.log_metric("loss_gen_recon_s_a", self.loss_gen_recon_s_a)
            comet_exp.log_metric("loss_gen_recon_c_a", self.loss_gen_recon_c_a)
            comet_exp.log_metric("loss_gen_recon_x_b", self.loss_gen_recon_x_b)
            comet_exp.log_metric("loss_gen_recon_s_b", self.loss_gen_recon_s_b)
            comet_exp.log_metric("loss_gen_recon_c_b", self.loss_gen_recon_c_b)
            comet_exp.log_metric("loss_gen_cycrecon_x_a", self.loss_gen_cycrecon_x_a)
            comet_exp.log_metric("loss_gen_cycrecon_x_b", self.loss_gen_cycrecon_x_b)
            comet_exp.log_metric("loss_gen_total", self.loss_gen_total)
            if hyperparameters["vgg_w"] > 0:
                comet_exp.log_metric("loss_gen_vgg_a", self.loss_gen_vgg_a)
                comet_exp.log_metric("loss_gen_vgg_b", self.loss_gen_vgg_b)
            if hyperparameters["semantic_w"] > 0:
                comet_exp.log_metric("loss_sem_seg", self.loss_sem_seg)
            if hyperparameters["domain_adv_w"] > 0:
                comet_exp.log_metric("domain_adv_loss_gen", self.domain_adv_loss)
            if synth:
                comet_exp.log_metric("loss_gen_recon_synth", self.loss_gen_recon_synth)

        self.loss_gen_total.backward()
        self.gen_opt.step()


    def gen_HD_update(
        self, x_a_HD, x_b_HD, hyperparameters, mask_a, mask_a_HD, mask_b, mask_b_HD, comet_exp=None, synth=False, warmup = False, lambda_dis = 0.0
    ):
        """Update HD generator (Upsampler & Downsampler)
        
        Arguments:
            x_a {image}    -- image low res
            x_a_HD {image} -- image high res
            x_b {image}    -- image low res
            x_b_HD {image} -- image high res
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters 

            mask_a {torch.Tensor} -- binary mask (0,1) corresponding to the ground in x_a (default: {None})
            mask_a_HD -- mask_a for HD image
            mask_b {torch.Tensor} -- binary mask (0,1) corresponding to the water in x_b (default: {None})
            mask_b_HD -- mask_b for HD image
            comet_exp {cometExperience} -- CometML object use to log all the loss and images (default: {None})
            synth {boolean}  -- binary True or False stating if we have a synthetic pair or not 
        
        Keyword Arguments:
            comet_exp {[type]} -- [description] (default: {None})
            synth {bool} -- [description] (default: {False})
        """
        # Regular downsampling
        x_a = self.dis_a.downsample(x_a_HD) 
        x_b = self.dis_b.downsample(x_b_HD)
        
        self.gen_opt_HD.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        if self.gen_state == 1:
            # encode
            c_a, s_a_prime = self.gen.encode(x_a, 1)
            c_b, s_b_prime = self.gen.encode(x_b, 2)

            # decode (within domain)
            x_a_recon,embedding_x_a = self.gen.decode(c_a, s_a_prime, 1, return_content=True)
            x_b_recon,embedding_x_b = self.gen.decode(c_b, s_b_prime, 2, return_content=True)

            # Downsample
            Downsampled_x_a = self.gen.localDown(x_a_HD)
            Downsampled_x_b = self.gen.localDown(x_b_HD)

            # Upsample
            x_a_recon_HD = self.gen.localUp(embedding_x_a + Downsampled_x_a)
            x_b_recon_HD = self.gen.localUp(embedding_x_b + Downsampled_x_b)

            # decode (cross domain)
            if self.guided == 1:
                x_ba, embedding_x_ba = self.gen.decode(c_b, s_a_prime, 1, return_content=True)
                x_ab, embedding_x_ab = self.gen.decode(c_a, s_b_prime, 2, return_content=True)
            else:
                print("self.guided unknown value:", self.guided)
            # Upsample
            x_ab_HD = self.gen.localUp(embedding_x_ab + Downsampled_x_a)
            x_ba_HD = self.gen.localUp(embedding_x_ba + Downsampled_x_b)

            # # encode again
            # c_b_recon, s_a_recon = self.gen.encode(x_ba, 1)
            # c_a_recon, s_b_recon = self.gen.encode(x_ab, 2)
            # # decode again (if needed)
            # x_aba = (
            #     self.gen.decode(c_a_recon, s_a_prime, 1)
            #     if hyperparameters["recon_x_cyc_w"] > 0
            #     else None
            # )
            # x_bab = (
            #     self.gen.decode(c_b_recon, s_b_prime, 2)
            #     if hyperparameters["recon_x_cyc_w"] > 0
            #     else None
            # )
        else:
            print("self.gen_state unknown value:", self.gen_state)
        
        # Warm-up reconstruction loss
        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon_HD, x_a_HD)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon_HD, x_b_HD)
        if warmup:
            m_up_2 = nn.UpsamplingNearest2d(scale_factor=2)
            self.loss_gen_recon_x_ab = self.recon_criterion(x_ab_HD, m_up_2(x_ab))
            self.loss_gen_recon_x_ba = self.recon_criterion(x_ba_HD, m_up_2(x_ba))
        else:
            self.loss_gen_recon_x_ba = 0
            self.loss_gen_recon_x_ab = 0

            
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba, x_ba_HD, lambda_D = lambda_dis)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab, x_ab_HD, lambda_D = lambda_dis)

        # # # semantic-segmentation loss
        # self.loss_sem_seg = (
        #     self.compute_semantic_seg_loss(x_a.squeeze(), x_ab.squeeze(), mask_a)
        #     + self.compute_semantic_seg_loss(x_b.squeeze(), x_ba.squeeze(), mask_b)
        #     if hyperparameters["semantic_w"] > 0
        #     else 0
        # )

        # total loss
        self.loss_gen_total = (
            hyperparameters["gan_w_HD"]   * self.loss_gen_adv_a
            + hyperparameters["gan_w_HD"] * self.loss_gen_adv_b
            + hyperparameters["recon_x_w_HD"] * self.loss_gen_recon_x_a
            + hyperparameters["recon_x_w_HD"] * self.loss_gen_recon_x_b
            + hyperparameters["recon_x_w_HD"] * self.loss_gen_recon_x_ab 
            + hyperparameters["recon_x_w_HD"] * self.loss_gen_recon_x_ba
        )
        #            + hyperparameters["semantic_w"] * self.loss_sem_seg

        if comet_exp is not None:
            comet_exp.log_metric("loss_gen_adv_a", self.loss_gen_adv_a)
            comet_exp.log_metric("loss_gen_adv_b", self.loss_gen_adv_b)
            comet_exp.log_metric("loss_gen_recon_x_a_HD", self.loss_gen_recon_x_a)
            comet_exp.log_metric("loss_gen_recon_x_b_HD", self.loss_gen_recon_x_b)
            comet_exp.log_metric("loss_gen_total", self.loss_gen_total)
            if warmup:
                comet_exp.log_metric("loss_gen_recon_x_ab", self.loss_gen_recon_x_ab)
                comet_exp.log_metric("loss_gen_recon_x_ba", self.loss_gen_recon_x_ba)

        self.loss_gen_total.backward()
        self.gen_opt_HD.step()

    def gen_global_update(
        self, x_a, x_a_HD, x_b, x_b_HD, hyperparameters, mask_a, mask_a_HD, mask_b, mask_b_HD, comet_exp=None, synth=False
    ):
        self.gen_opt_global.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        
        if self.gen_state == 1:
            # encode
            c_a, s_a_prime = self.gen.encode(x_a, 1)
            c_b, s_b_prime = self.gen.encode(x_b, 2)

            # decode (within domain)
            x_a_recon,embedding_x_a = self.gen.decode(c_a, s_a_prime, 1, return_content=True)
            x_b_recon,embedding_x_b = self.gen.decode(c_b, s_b_prime, 2, return_content=True)

            # Downsample
            Downsampled_x_a = self.gen.localDown(x_a_HD)
            Downsampled_x_b = self.gen.localDown(x_b_HD)

            # Upsample
            x_a_recon_HD = self.gen.localUp(embedding_x_a+Downsampled_x_a)
            x_b_recon_HD = self.gen.localUp(embedding_x_b+Downsampled_x_b)

            # decode (cross domain)
            if self.guided == 1:
                x_ba, embedding_x_ba = self.gen.decode(c_b, s_a_prime, 1, return_content=True)
                x_ab, embedding_x_ab = self.gen.decode(c_a, s_b_prime, 2, return_content=True)
            else:
                print("self.guided unknown value:", self.guided)
                
            # Upsample
            x_ab_HD = self.gen.localUp(embedding_x_ab+Downsampled_x_a)
            x_ba_HD = self.gen.localUp(embedding_x_ba+Downsampled_x_b)

            # encode again
            c_b_recon, s_a_recon = self.gen.encode(x_ba, 1)
            c_a_recon, s_b_recon = self.gen.encode(x_ab, 2)
            # decode again (if needed)
            x_aba = (
                self.gen.decode(c_a_recon, s_a_prime, 1)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
            x_bab = (
                self.gen.decode(c_b_recon, s_b_prime, 2)
                if hyperparameters["recon_x_cyc_w"] > 0
                else None
            )
        else:
            print("self.gen_state unknown value:", self.gen_state)
        
        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        # Synthetic reconstruction loss
        self.loss_gen_recon_synth = self.recon_criterion_mask(x_ab, x_b,mask_b) +\
                                    self.recon_criterion_mask(x_ba, x_a,mask_a)  if synth else 0

        if self.recon_mask:
            self.loss_gen_cycrecon_x_a = (
                self.recon_criterion_mask(x_aba, x_a, mask_a)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )
            self.loss_gen_cycrecon_x_b = (
                self.recon_criterion_mask(x_bab, x_b, mask_b)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )
        else:
            self.loss_gen_cycrecon_x_a = (
                self.recon_criterion(x_aba, x_a)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )
            self.loss_gen_cycrecon_x_b = (
                self.recon_criterion(x_bab, x_b)
                if hyperparameters["recon_x_cyc_w"] > 0
                else 0
            )

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(nn.Upsample(scale_factor=2, mode='nearest')(x_ba))
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(nn.Upsample(scale_factor=2, mode='nearest')(x_ab))
        
        # GAN loss HD
        self.loss_gen_adv_a_HD = self.dis_a.calc_gen_loss(x_ba_HD)
        self.loss_gen_adv_b_HD = self.dis_b.calc_gen_loss(x_ab_HD)
        
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = (
            self.compute_vgg_loss(self.vgg, x_ba, x_b)
            if hyperparameters["vgg_w"] > 0
            else 0
        )
        self.loss_gen_vgg_b = (
            self.compute_vgg_loss(self.vgg, x_ab, x_a)
            if hyperparameters["vgg_w"] > 0
            else 0
        )

        # semantic-segmentation loss
        self.loss_sem_seg = (
            self.compute_semantic_seg_loss(x_a.squeeze(), x_ab.squeeze(), mask_a)
            + self.compute_semantic_seg_loss(x_b.squeeze(), x_ba.squeeze(), mask_b)
            if hyperparameters["semantic_w"] > 0
            else 0
        )
        # Domain adversarial loss (c_a and c_b are swapped because we want the feature to be less informative
        # minmax (accuracy but max min loss)
        self.domain_adv_loss = (
            self.compute_domain_adv_loss(c_a,c_b, compute_accuracy=False, minimize=False)
            if hyperparameters["domain_adv_w"] > 0
            else 0
        )
        
        # reconstruction loss HD
        self.loss_gen_recon_x_a_HD = self.recon_criterion(x_a_recon_HD, x_a_HD)
        self.loss_gen_recon_x_b_HD = self.recon_criterion(x_b_recon_HD, x_b_HD)

        # total loss
        self.loss_gen_total = (
            hyperparameters["gan_w"] * self.loss_gen_adv_a
            + hyperparameters["gan_w"] * self.loss_gen_adv_b
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_a
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_a
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_a
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_b
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_b
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_b
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_a
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_b
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_a
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
            + hyperparameters["semantic_w"] * self.loss_sem_seg
            + hyperparameters["domain_adv_w"] * self.domain_adv_loss
            + hyperparameters["recon_synth_w"] * self.loss_gen_recon_synth
            + hyperparameters["gan_w_HD"] * self.loss_gen_adv_a_HD
            + hyperparameters["gan_w_HD"] * self.loss_gen_adv_b_HD
            + hyperparameters["recon_x_w_HD"] * self.loss_gen_recon_x_a_HD
            + hyperparameters["recon_x_w_HD"] * self.loss_gen_recon_x_b_HD
        )

        if comet_exp is not None:
            comet_exp.log_metric("loss_gen_adv_a", self.loss_gen_adv_a)
            comet_exp.log_metric("loss_gen_adv_b", self.loss_gen_adv_b)
            comet_exp.log_metric("loss_gen_recon_x_a", self.loss_gen_recon_x_a)
            comet_exp.log_metric("loss_gen_recon_s_a", self.loss_gen_recon_s_a)
            comet_exp.log_metric("loss_gen_recon_c_a", self.loss_gen_recon_c_a)
            comet_exp.log_metric("loss_gen_recon_x_b", self.loss_gen_recon_x_b)
            comet_exp.log_metric("loss_gen_recon_s_b", self.loss_gen_recon_s_b)
            comet_exp.log_metric("loss_gen_recon_c_b", self.loss_gen_recon_c_b)
            comet_exp.log_metric("loss_gen_cycrecon_x_a", self.loss_gen_cycrecon_x_a)
            comet_exp.log_metric("loss_gen_cycrecon_x_b", self.loss_gen_cycrecon_x_b)
            comet_exp.log_metric("loss_gen_total", self.loss_gen_total)
            if hyperparameters["vgg_w"] > 0:
                comet_exp.log_metric("loss_gen_vgg_a", self.loss_gen_vgg_a)
                comet_exp.log_metric("loss_gen_vgg_b", self.loss_gen_vgg_b)
            if hyperparameters["semantic_w"] > 0:
                comet_exp.log_metric("loss_sem_seg", self.loss_sem_seg)
            if hyperparameters["domain_adv_w"] > 0:
                comet_exp.log_metric("domain_adv_loss_gen", self.domain_adv_loss)
            if synth:
                comet_exp.log_metric("loss_gen_recon_synth", self.loss_gen_recon_synth)
            comet_exp.log_metric("loss_gen_adv_a_HD", self.loss_gen_adv_a_HD)
            comet_exp.log_metric("loss_gen_adv_b_HD", self.loss_gen_adv_b_HD)
            comet_exp.log_metric("loss_gen_recon_x_a_HD", self.loss_gen_recon_x_a_HD)
            comet_exp.log_metric("loss_gen_recon_x_b_HD", self.loss_gen_recon_x_b_HD)
            comet_exp.log_metric("loss_gen_total", self.loss_gen_total)

        self.loss_gen_total.backward()
        self.gen_opt_global.step()
        
    def compute_vgg_loss(self, vgg, img, target):
        """ 
        Compute the domain-invariant perceptual loss
        
        Arguments:
            vgg {model} -- popular Convolutional Network for Classification and Detection
            img {torch.Tensor} -- image before translation
            target {torch.Tensor} -- image after translation
        
        Returns:
            torch.Float -- domain invariant perceptual loss
        """
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean(
            (self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2
        )

    def compute_domain_adv_loss(self, c_a, c_b, compute_accuracy=False,minimize=True):
        """ 
        Compute a domain adversarial loss on the embedding of the classifier:
        we are trying to learn an anonymized representation of the content. 
        
        Arguments:
            c_a {torch.tensor} -- content extracted from an image of domain A with encoder A
            c_b {torch.tensor} -- content extracted from an image of domain B with encoder B
        
        Keyword Arguments:
            compute_accuracy {bool} -- either return only the loss or loss and softmax probs
            (default: {False})
            minimize {bool} -- optimize classification accuracy(True) or anonymized the representation(False)
        
        Returns:
            torch.Float -- loss (optionnal softmax P(classifier(c_a)=a) and P(classifier(c_b)=b)) 
        """
        # Infer domain classifier on content extracted from an image of domainA
        output_a = self.domain_classifier(c_a)

        # Infer domain classifier on content extracted from an image of domainB
        output_b = self.domain_classifier(c_b)

        # Concatenate the output in a single vector
        output = torch.cat((output_a,output_b))       
        
    
        if minimize:
            target = torch.tensor([1.,0.,0.,1.],device='cuda') 
        else:
            target = torch.tensor([0.5,0.5,0.5,0.5],device='cuda')
        # mean square error loss
        loss = torch.nn.MSELoss()(output,target)
        if compute_accuracy:
            return loss, output_a[0], output_b[1]
        else:
            return loss

    def compute_semantic_seg_loss(self, img1, img2, mask):
        """ 
        Compute semantic segmentation loss between two images on the unmasked region
        
        Arguments:
            img1 {torch.Tensor} -- Image from domain A after transform in tensor format
            img2 {torch.Tensor} -- Image transformed 
            mask {torch.Tensor} -- Binary mask where we force the loss to be zero
        
        Returns:
            torch.float -- Cross entropy loss on the unmasked region
        """
        # denorm
        img1_denorm = (img1 + 1) / 2.0
        img2_denorm = (img2 + 1) / 2.0
        # norm for semantic seg network
        input_transformed1 = seg_transform()(img1_denorm).unsqueeze(0)
        input_transformed2 = seg_transform()(img2_denorm).unsqueeze(0)
        # compute labels from original image and logits from translated version
        target = (
            self.segmentation_model(input_transformed1).squeeze().max(0)[1].unsqueeze(0)
        )
        output = self.segmentation_model(input_transformed2)
        # Resize mask to the size of the image
        mask1 = torch.nn.functional.interpolate(mask, size=(self.newsize, self.newsize))
        mask1_tensor = torch.tensor(mask1, dtype=torch.long).cuda()
        # we want the masked region to be labeled as unknown (19 is not an existing label)
        target_with_mask = torch.mul(1 - mask1_tensor, target) + mask1_tensor * 19
        mask2 = torch.nn.functional.interpolate(mask, size=(self.newsize, self.newsize))
        mask_tensor = torch.tensor(mask2, dtype=torch.float).cuda()
        output_with_mask = torch.mul(1 - mask_tensor, output)
        # cat the mask as to the logits (loss=0 over the masked region)
        output_with_mask_cat = torch.cat(
            (output_with_mask.squeeze(), mask_tensor.squeeze().unsqueeze(0))
        )
        loss = nn.CrossEntropyLoss()(
            output_with_mask_cat.unsqueeze(0), target_with_mask.squeeze().unsqueeze(0)
        )
        return loss

    def sample(self, x_a, x_b):
        """ 
        Infer the model on a batch of image
        
        Arguments:
            x_a {torch.Tensor} -- batch of image from domain A
            x_b {[type]} -- batch of image from domain B
        
        Returns:
            A list of torch images -- columnwise :x_a, autoencode(x_a), x_ab_1, x_ab_2
            Or if self.semantic_w is true: x_a, autoencode(x_a), Semantic segmentation x_a, 
            x_ab_1,semantic segmentation x_ab_1, x_ab_2
        """
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []

        if self.gen_state == 0:
            for i in range(x_a.size(0)):
                c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
                c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
                x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
                x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
                if self.guided == 0:
                    x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
                    x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
                    x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
                    x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
                elif self.guided == 1:
                    x_ba1.append(
                        self.gen_a.decode(c_b, s_a_fake)
                    )  # s_a1[i].unsqueeze(0)))
                    x_ba2.append(
                        self.gen_a.decode(c_b, s_a_fake)
                    )  # s_a2[i].unsqueeze(0)))
                    x_ab1.append(
                        self.gen_b.decode(c_a, s_b_fake)
                    )  # s_b1[i].unsqueeze(0)))
                    x_ab2.append(
                        self.gen_b.decode(c_a, s_b_fake)
                    )  # s_b2[i].unsqueeze(0)))
                else:
                    print("self.guided unknown value:", self.guided)

        elif self.gen_state == 1:
            for i in range(x_a.size(0)):
                c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0), 1)
                c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0), 2)
                x_a_recon.append(self.gen.decode(c_a, s_a_fake, 1))
                x_b_recon.append(self.gen.decode(c_b, s_b_fake, 2))
                if self.guided == 0:
                    x_ba1.append(self.gen.decode(c_b, s_a1[i].unsqueeze(0), 1))
                    x_ba2.append(self.gen.decode(c_b, s_a2[i].unsqueeze(0), 1))
                    x_ab1.append(self.gen.decode(c_a, s_b1[i].unsqueeze(0), 2))
                    x_ab2.append(self.gen.decode(c_a, s_b2[i].unsqueeze(0), 2))
                elif self.guided == 1:
                    x_ba1.append(
                        self.gen.decode(c_b, s_a_fake, 1)
                    )  # s_a1[i].unsqueeze(0)))
                    x_ba2.append(
                        self.gen.decode(c_b, s_a_fake, 1)
                    )  # s_a2[i].unsqueeze(0)))
                    x_ab1.append(
                        self.gen.decode(c_a, s_b_fake, 2)
                    )  # s_b1[i].unsqueeze(0)))
                    x_ab2.append(
                        self.gen.decode(c_a, s_b_fake, 2)
                    )  # s_b2[i].unsqueeze(0)))
                else:
                    print("self.guided unknown value:", self.guided)

        else:
            print("self.gen_state unknown value:", self.gen_state)

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)

        if self.semantic_w:
            rgb_a_list, rgb_b_list, rgb_ab_list, rgb_ba_list = [], [], [], []

            for i in range(x_a.size(0)):

                # Inference semantic segmentation on original images
                im_a = (x_a[i].squeeze() + 1) / 2.0
                im_b = (x_b[i].squeeze() + 1) / 2.0

                input_transformed_a = seg_transform()(im_a).unsqueeze(0)
                input_transformed_b = seg_transform()(im_b).unsqueeze(0)
                output_a = (
                    self.segmentation_model(input_transformed_a).squeeze().max(0)[1]
                )
                output_b = (
                    self.segmentation_model(input_transformed_b).squeeze().max(0)[1]
                )

                rgb_a = decode_segmap(output_a.cpu().numpy())
                rgb_b = decode_segmap(output_b.cpu().numpy())
                rgb_a = Image.fromarray(rgb_a).resize((x_a.size(3), x_a.size(3)))
                rgb_b = Image.fromarray(rgb_b).resize((x_a.size(3), x_a.size(3)))

                rgb_a_list.append(transforms.ToTensor()(rgb_a).unsqueeze(0))
                rgb_b_list.append(transforms.ToTensor()(rgb_b).unsqueeze(0))

                # Inference semantic segmentation on fake images
                image_ab = (x_ab1[i].squeeze() + 1) / 2.0
                image_ba = (x_ba1[i].squeeze() + 1) / 2.0

                input_transformed_ab = seg_transform()(image_ab).unsqueeze(0).to("cuda")
                input_transformed_ba = seg_transform()(image_ba).unsqueeze(0).to("cuda")

                output_ab = (
                    self.segmentation_model(input_transformed_ab).squeeze().max(0)[1]
                )
                output_ba = (
                    self.segmentation_model(input_transformed_ba).squeeze().max(0)[1]
                )

                rgb_ab = decode_segmap(output_ab.cpu().numpy())
                rgb_ba = decode_segmap(output_ba.cpu().numpy())

                rgb_ab = Image.fromarray(rgb_ab).resize((x_a.size(3), x_a.size(3)))
                rgb_ba = Image.fromarray(rgb_ba).resize((x_a.size(3), x_a.size(3)))

                rgb_ab_list.append(transforms.ToTensor()(rgb_ab).unsqueeze(0))
                rgb_ba_list.append(transforms.ToTensor()(rgb_ba).unsqueeze(0))

            rgb1_a, rgb1_b, rgb1_ab, rgb1_ba = (
                torch.cat(rgb_a_list).cuda(),
                torch.cat(rgb_b_list).cuda(),
                torch.cat(rgb_ab_list).cuda(),
                torch.cat(rgb_ba_list).cuda(),
            )

        self.train()
        if self.semantic_w:
            self.segmentation_model.eval()
            return (
                x_a,
                x_a_recon,
                rgb1_a,
                x_ab1,
                rgb1_ab,
                x_ab2,
                x_b,
                x_b_recon,
                rgb1_b,
                x_ba1,
                rgb1_ba,
                x_ba2,
            )
        else:
            return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def sample_HD(self, x_a_HD, x_b_HD,lambda_dis = 0):
        """ 
        Infer the model on a batch of image
        
        Arguments:
            x_a {torch.Tensor} -- batch of image from domain A
            x_b {[type]} -- batch of image from domain B
        
        Returns:
            A list of torch images -- columnwise :x_a, autoencode(x_a), x_ab_1, x_ab_2
            Or if self.semantic_w is true: x_a, autoencode(x_a), Semantic segmentation x_a, 
            x_ab_1,semantic segmentation x_ab_1, x_ab_2
        """
        
        x_a = self.dis_a.downsample(x_a_HD) # Regular downsampling function
        x_b = self.dis_b.downsample(x_b_HD) # Regular downsampling function
        
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon_HD, x_b_recon_HD, x_ab_HD_, x_ba_HD_, x_ab_, x_ba_ = [], [], [], [], [], []

        if self.gen_state == 1:
            for i in range(x_a.size(0)):
                # encode
                c_a, s_a_prime = self.gen.encode(x_a[i].unsqueeze(0), 1)
                c_b, s_b_prime = self.gen.encode(x_b[i].unsqueeze(0), 2)

                # decode (within domain)
                x_a_recon,embedding_x_a = self.gen.decode(c_a, s_a_prime, 1, return_content=True)
                x_b_recon,embedding_x_b = self.gen.decode(c_b, s_b_prime, 2, return_content=True)

                # Downsample
                Downsampled_x_a = self.gen.localDown(x_a_HD[i].unsqueeze(0))
                Downsampled_x_b = self.gen.localDown(x_b_HD[i].unsqueeze(0))

                # Upsample
                x_a_recon_HD.append(self.gen.localUp(embedding_x_a+Downsampled_x_a))
                x_b_recon_HD.append(self.gen.localUp(embedding_x_b+Downsampled_x_b))
                
                # decode (cross domain)
                if self.guided == 1:
                    x_ba, embedding_x_ba = self.gen.decode(c_b, s_a_prime, 1, return_content=True)
                    x_ab, embedding_x_ab = self.gen.decode(c_a, s_b_prime, 2, return_content=True)
                else:
                    print("self.guided unknown value:", self.guided)
                 
                x_ab_.append(nn.Upsample(scale_factor=2, mode='nearest')(x_ab))
                x_ba_.append(nn.Upsample(scale_factor=2, mode='nearest')(x_ba))
                
                x_ab_HD =  self.gen.localUp(embedding_x_ab + Downsampled_x_a)
                x_ba_HD =  self.gen.localUp(embedding_x_ba + Downsampled_x_b)
                
                transition_ab = lambda_dis*x_ab_HD + (1-lambda_dis)*self.dis_a.upsample(x_ab)
                transition_ba = lambda_dis*x_ba_HD + (1-lambda_dis)*self.dis_a.upsample(x_ba)
                # Upsample
                x_ab_HD_.append(transition_ab)
                x_ba_HD_.append(transition_ba)

        else:
            print("self.gen_state unknown value:", self.gen_state)
            
        self.train()
        if self.semantic_w:
            self.segmentation_model.eval()
        
        x_a_recon_HD, x_b_recon_HD = torch.cat(x_a_recon_HD), torch.cat(x_b_recon_HD)
        x_ba_HD = torch.cat(x_ba_HD_)
        x_ab_HD = torch.cat(x_ab_HD_)
        x_ab_   = torch.cat(x_ab_)
        x_ba_   = torch.cat(x_ba_)

        
        return x_a_HD, x_a_recon_HD, x_ab_, x_ab_HD, x_b_HD, x_b_recon_HD, x_ba_, x_ba_HD

    def sample_fid(self, x_a, x_b):
        """ 
        Infer the model on a batch of image
        
        Arguments:
            x_a {torch.Tensor} -- batch of image from domain A
            x_b {[type]} -- batch of image from domain B
        
        Returns:
            A list of torch images -- columnwise :x_a, autoencode(x_a), x_ab_1, x_ab_2
            Or if self.semantic_w is true: x_a, autoencode(x_a), Semantic segmentation x_a, 
            x_ab_1,semantic segmentation x_ab_1, x_ab_2
        """
        self.eval()
        x_ab1= []
        
        if self.gen_state == 0:
            for i in range(x_a.size(0)):
                c_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
                _, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))

                if self.guided == 1:
                    x_ab1.append(
                        self.gen_b.decode(c_a, s_b_fake)
                    ) 

                else:
                    print("self.guided unknown value:", self.guided)

        elif self.gen_state == 1:
            for i in range(x_a.size(0)):
                c_a, _ = self.gen.encode(x_a[i].unsqueeze(0), 1)
                _, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0), 2)
                if self.guided == 1:
                    x_ab1.append(
                        self.gen.decode(c_a, s_b_fake, 2)
                    )
                else:
                    print("self.guided unknown value:", self.guided)

        else:
            print("self.gen_state unknown value:", self.gen_state)
            
        x_ab1 = torch.cat(x_ab1)
        self.train()
        if self.semantic_w:
            self.segmentation_model.eval()
            
        return x_ab1

        
    def dis_update(self, x_a, x_b, hyperparameters, comet_exp=None):
        """
        Update the weights of the discriminator
        
        Arguments:
            x_a {torch.Tensor} -- Image from domain A after transform in tensor format
            x_b {torch.Tensor} -- Image from domain B after transform in tensor format
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters 
        
        Keyword Arguments:
            comet_exp {cometExperience} -- CometML object use to log all the loss and images (default: {None})        
        """
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        if self.gen_state == 0:
            # encode
            c_a, s_a_prime = self.gen_a.encode(x_a)
            c_b, s_b_prime = self.gen_b.encode(x_b)
            # decode (cross domain)
            if self.guided == 0:
                x_ba = self.gen_a.decode(c_b, s_a)
                x_ab = self.gen_b.decode(c_a, s_b)
            elif self.guided == 1:
                x_ba = self.gen_a.decode(c_b, s_a_prime)
                x_ab = self.gen_b.decode(c_a, s_b_prime)
            else:
                print("self.guided unknown value:", self.guided)
        elif self.gen_state == 1:
            # encode
            c_a, s_a_prime = self.gen.encode(x_a, 1)
            c_b, s_b_prime = self.gen.encode(x_b, 2)
            # decode (cross domain)
            if self.guided == 0:
                x_ba = self.gen.decode(c_b, s_a, 1)
                x_ab = self.gen.decode(c_a, s_b, 2)
            elif self.guided == 1:
                x_ba = self.gen.decode(c_b, s_a_prime, 1)
                x_ab = self.gen.decode(c_a, s_b_prime, 2)
            else:
                print("self.guided unknown value:", self.guided)
        else:
            print("self.gen_state unknown value:", self.gen_state)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

        if comet_exp is not None:
            comet_exp.log_metric("loss_dis_b", self.loss_dis_b)
            comet_exp.log_metric("loss_dis_a", self.loss_dis_a)

        self.loss_dis_total = (
            hyperparameters["gan_w"] * self.loss_dis_a
            + hyperparameters["gan_w"] * self.loss_dis_b
        )
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def dis_HD_update(self, x_a_HD, x_b_HD, hyperparameters, comet_exp=None, lamda_dis=0.0):
        """
        Update the weights of the discriminator
        
        Arguments:
            x_a {torch.Tensor} -- Image from domain A after transform in tensor format
            x_b {torch.Tensor} -- Image from domain B after transform in tensor format
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters 
        
        Keyword Arguments:
            comet_exp {cometExperience} -- CometML object use to log all the loss and images (default: {None})        
        """
        x_a = self.dis_a.downsample(x_a_HD) # Regular downsampling function
        x_b = self.dis_b.downsample(x_b_HD) # Regular downsampling function
        
        self.dis_HD_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        if self.gen_state == 1:
            # encode
            c_a, s_a_prime = self.gen.encode(x_a, 1)
            c_b, s_b_prime = self.gen.encode(x_b, 2)
            # decode (cross domain)
            if self.guided == 1:
                x_ba, embedding_xba = self.gen.decode(c_b, s_a_prime, 1, return_content = True)
                x_ab, embedding_xab = self.gen.decode(c_a, s_b_prime, 2, return_content = True)
            else:
                print("self.guided unknown value:", self.guided)
        else:
            print("self.gen_state unknown value:", self.gen_state)

        # Downsampling part
        Downsampled_x_a = self.gen.localDown(x_a_HD)
        Downsampled_x_b = self.gen.localDown(x_b_HD)
        
        # Upsampling part
        upsampled_xab = self.gen.localUp(embedding_xab+Downsampled_x_a)
        upsampled_xba = self.gen.localUp(embedding_xba+Downsampled_x_b)
        
        # Dis_HD loss
        # input_fake, input_fake_HD, input_real_HD, lambda_D =
        self.loss_dis_HD_a = self.dis_a.calc_dis_loss(x_ba, upsampled_xba.detach(), x_a_HD, lambda_D = lamda_dis) 
        self.loss_dis_HD_b = self.dis_b.calc_dis_loss(x_ab, upsampled_xab.detach(), x_b_HD, lambda_D = lamda_dis)

        if comet_exp is not None:
            comet_exp.log_metric("loss_dis_HD_b", self.loss_dis_HD_b)
            comet_exp.log_metric("loss_dis_HD_a", self.loss_dis_HD_a)

        self.loss_dis_total = (
            hyperparameters["gan_w_HD"] * self.loss_dis_HD_a
            + hyperparameters["gan_w_HD"] * self.loss_dis_HD_b
        )
        self.loss_dis_total.backward()
        self.dis_HD_opt.step()

    def dis_global_update(self, x_a, x_a_HD, x_b, x_b_HD, hyperparameters, comet_exp=None):
        """
        Update the weights of the global discriminator        
        """
        self.dis_opt_global.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        if self.gen_state == 1:
            # encode
            c_a, s_a_prime = self.gen.encode(x_a, 1)
            c_b, s_b_prime = self.gen.encode(x_b, 2)
            # decode (cross domain)
            if self.guided == 1:
                x_ba, embedding_xba = self.gen.decode(c_b, s_a_prime, 1, return_content =True)
                x_ab, embedding_xab = self.gen.decode(c_a, s_b_prime, 2, return_content =True)
            else:
                print("self.guided unknown value:", self.guided)
        else:
            print("self.gen_state unknown value:", self.gen_state)

        # Downsampling part
        Downsampled_x_a = self.gen.localDown(x_a_HD)
        Downsampled_x_b = self.gen.localDown(x_b_HD)
        
        # Upsampling part
        upsampled_xab = self.gen.localUp(embedding_xab+Downsampled_x_a)
        upsampled_xba = self.gen.localUp(embedding_xba+Downsampled_x_b)
        
        # Dis_HD loss
        self.loss_dis_HD_a = self.dis_a.calc_dis_loss(upsampled_xba.detach(),x_a_HD) #x_ba.detach(), x_a)
        self.loss_dis_HD_b = self.dis_b.calc_dis_loss(upsampled_xab.detach(),x_b_HD) #x_ab.detach(), x_b)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(nn.Upsample(scale_factor=2, mode='nearest')(x_ba).detach(), \
                                                   nn.Upsample(scale_factor=2, mode='nearest')(x_a))
        self.loss_dis_b = self.dis_b.calc_dis_loss(nn.Upsample(scale_factor=2, mode='nearest')(x_ab).detach(), \
                                                   nn.Upsample(scale_factor=2, mode='nearest')(x_b))
        
        if comet_exp is not None:
            
            comet_exp.log_metric("loss_dis_b", self.loss_dis_b)
            comet_exp.log_metric("loss_dis_a", self.loss_dis_a)
            comet_exp.log_metric("loss_dis_HD_b", self.loss_dis_HD_b)
            comet_exp.log_metric("loss_dis_HD_a", self.loss_dis_HD_a)

        self.loss_dis_total = (
            hyperparameters["gan_w"] * self.loss_dis_a
            + hyperparameters["gan_w"] * self.loss_dis_b
            + hyperparameters["gan_w_HD"] * self.loss_dis_HD_a
            + hyperparameters["gan_w_HD"] * self.loss_dis_HD_b
        )
        self.loss_dis_total.backward()
        self.dis_opt_global.step()

    def domain_classifier_update(self, x_a, x_b, hyperparameters, comet_exp=None):
        """
        Update the weights of the domain classifier
        
        Arguments:
            x_a {torch.Tensor} -- Image from domain A after transform in tensor format
            x_b {torch.Tensor} -- Image from domain B after transform in tensor format
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters 
        
        Keyword Arguments:
            comet_exp {cometExperience} -- CometML object use to log all the loss and images (default: {None})        
        """
        self.dann_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        if self.gen_state == 0:
            # encode
            c_a, _ = self.gen_a.encode(x_a)
            c_b, _ = self.gen_b.encode(x_b)
        elif self.gen_state == 1:
            # encode
            c_a, _ = self.gen.encode(x_a, 1)
            c_b, _ = self.gen.encode(x_b, 2)
        else:
            print("self.gen_state unknown value:", self.gen_state)

        # domain classifier loss
        self.domain_class_loss, out_a, out_b = self.compute_domain_adv_loss(
            c_a, c_b, compute_accuracy=True,minimize=True)

        if comet_exp is not None:
            comet_exp.log_metric("domain_class_loss", self.domain_class_loss)
            comet_exp.log_metric("probability A being identified as A", out_a)
            comet_exp.log_metric("probability B being identified as B", out_b)

        self.domain_class_loss.backward()
        self.dann_opt.step()

    def update_learning_rate(self):
        """ 
        Update the learning rate
        """
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.dann_scheduler is not None:
            self.dann_scheduler.step()

    def update_learning_rate_HD(self):
        if self.dis_HD_scheduler is not None:
            self.dis_HD_scheduler.step()
        if self.gen_HD_scheduler is not None:
            self.gen_HD_scheduler.step()
            
    def update_learning_rate_global(self):
        if self.dis_global_scheduler is not None:
            self.dis_global_scheduler.step()
        if self.gen_global_scheduler is not None:
            self.gen_global_scheduler.step() 

    def resume(self, checkpoint_dir, hyperparameters, HD_ckpt = False):
        """
        Resume the training loading the network parameters
        
        Arguments:
            checkpoint_dir {string} -- path to the directory where the checkpoints are saved
            hyperparameters {dictionnary} -- dictionnary with all hyperparameters 
        
        Returns:
            int -- number of iterations (used by the optimizer)
        """
        if not HD_ckpt:
            # Load generators
            last_model_name = get_model_list(checkpoint_dir, "gen")
            state_dict = torch.load(last_model_name)

            # overwrite entries in the existing state dict
            gen_dict = self.gen.state_dict()

            if self.gen_state == 1:
                gen_dict.update(state_dict["2"]) 
                self.gen.load_state_dict(gen_dict)
            else:
                print("self.gen_state unknown value:", self.gen_state)

            # Load domain classifier
            if self.domain_classif == 1:
                last_model_name = get_model_list(checkpoint_dir, "domain_classif")
                state_dict = torch.load(last_model_name)
                self.domain_classifier.load_state_dict(state_dict["d"])

            iterations = int(last_model_name[-11:-3])
            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis")
            state_dict = torch.load(last_model_name)
            
            #########################################################################
            #         Load discriminator weights into HD discriminator
            #########################################################################
            
            # state_dict a
            state_dict_ = state_dict['a']
            
            #             # store state_dict keys
            #             keys = []
            #             for key in state_dict_:
            #                 keys.append(key)

            #             # Iterate through the keys and replace them with their new name in the extanded architecture
            #             for key in keys:
            #                 split_list = key.split('.')
            #                 n_layer = int(split_list[1]) 
            #                 split_list[1] = str(n_layer + 1)
            #                 new_key = ".".join(split_list)
            #                 state_dict_[new_key] = state_dict_[key]
            #                 if n_layer == 0:
            #                     del state_dict_[key]

            # state_dict a
            pretrained_dict = state_dict_
            # Define the weight of the global discriminator
            model_dict      = self.dis_a.state_dict()
            # Overwrite entries in the global discriminator state dict
            model_dict.update(pretrained_dict) 
            # Load the new state dict in the network
            self.dis_a.load_state_dict(model_dict)
  
            # state_dict b
            state_dict_ = state_dict['b']
            
            #             # store state_dict keys
            #             keys = []
            #             for key in state_dict_:
            #                 keys.append(key)

            #             # Iterate through the keys and replace them with their new name in the extanded architecture
            #             for key in keys:
            #                 split_list = key.split('.')
            #                 n_layer = int(split_list[1]) 
            #                 split_list[1] = str(n_layer + 1)
            #                 new_key = ".".join(split_list)
            #                 state_dict_[new_key] = state_dict_[key]
            #                 if n_layer == 0:
            #                     del state_dict_[key]

            # state_dict b
            pretrained_dict = state_dict_
            # Define the weight of the global discriminator
            model_dict      = self.dis_b.state_dict()
            # Overwrite entries in the global discriminator state dict
            model_dict.update(pretrained_dict) 
            # Load the new state dict in the network
            self.dis_b.load_state_dict(model_dict)
            
            #########################################################################
            #         Load discriminator weights into HD discriminator
            #########################################################################
            
            #             self.dis_a.load_state_dict(state_dict["a"])
            #             self.dis_b.load_state_dict(state_dict["b"])
            # Load optimizers
            # if not self.train_global:
            #                 state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
            #                 self.dis_opt.load_state_dict(state_dict["dis"])
            #                 self.gen_opt.load_state_dict(state_dict["gen"])
            #                 if self.domain_classif == 1:
            #                     self.dann_opt.load_state_dict(state_dict["dann"])
            #                     self.dann_scheduler = get_scheduler(
            #                         self.dann_opt, hyperparameters, iterations
            #                     )
            #                 # Reinitilize schedulers
            #                 self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
            #                 self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
            print("Resume from iteration %d" % iterations, " without reinit schedulers")
            return 0
        
        # IS HD
        if HD_ckpt:
            # Load discriminators HD
            last_model_name = get_model_list(checkpoint_dir, "dis_HD")
            iterations_HD = int(last_model_name[-11:-3])
            state_dict = torch.load(last_model_name)
            self.dis_a.load_state_dict(state_dict["a"])
            self.dis_b.load_state_dict(state_dict["b"])

            # Load G2 Local Upsampler and Downsampler
            last_model_name = get_model_list(checkpoint_dir, "G2_HD")
            state_dict = torch.load(last_model_name)
            self.gen.localUp.load_state_dict(state_dict["localUp"])
            self.gen.localDown.load_state_dict(state_dict["localDown"])
            if not self.train_global:
                # Load G2 optimizers
                state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer_G2.pt"))
                self.dis_HD_opt.load_state_dict(state_dict["dis_HD"])
                self.gen_opt_HD.load_state_dict(state_dict["gen_HD"])
                # Reinitilize schedulers
                self.dis_HD_scheduler = get_scheduler(self.dis_HD_opt,hyperparameters,iterations_HD)
                self.gen_HD_scheduler = get_scheduler(self.gen_opt_HD,hyperparameters,iterations_HD)
            print("Resume from iterations_HD %d" % iterations_HD)
            return iterations_HD
        else:
            print('This situation is not Handled:(loading both ckpt from the fine tuning_')

    def save(self, snapshot_dir, iterations, iterations_HD = 0, save_HD = False):
        """
        Save generators, discriminators, and optimizers
        
        Arguments:
            snapshot_dir {string} -- directory path where to save the networks weights
            iterations {int} -- number of training iterations
        """
        if self.train_global:
            # Define generator and discriminator HD ckpt name
            gen_HD_name = os.path.join(snapshot_dir, "G2_HD_%08d.pt" % (iterations_HD + 1))
            dis_HD_name = os.path.join(snapshot_dir, "dis_HD_%08d.pt" % (iterations_HD + 1))
            
            # Save discriminators and generator HD ckpt
            torch.save(
                {"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_HD_name
            )
            torch.save(
                {"localUp": self.gen.localUp.state_dict(), "localDown": self.gen.localDown.state_dict()}, gen_HD_name
            )
            
            # Define generator and discriminator ckpt name
            gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations_HD + 1))
            dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations_HD + 1))
            
            # Save generators, discriminators ckpt
            torch.save(
                {"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_name
            )
            torch.save({"2": self.gen.state_dict()}, gen_name)
            
            # Save the global optimizer
            opt_name = os.path.join(snapshot_dir, "optimizer_global.pt")
            torch.save(
                {"gen_global": self.gen_opt_global.state_dict(), "dis_global": self.dis_opt_global.state_dict()},
                opt_name,
            )

        else:
            if self.train_G2_only:
                # Save generators, discriminators, and optimizers
                gen_HD_name = os.path.join(snapshot_dir, "G2_HD_%08d.pt" % (iterations_HD + 1))
                dis_HD_name = os.path.join(snapshot_dir, "dis_HD_%08d.pt" % (iterations_HD + 1))
                opt_name = os.path.join(snapshot_dir, "optimizer_G2.pt")
                torch.save(
                    {"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_HD_name
                )
                torch.save(
                    {"localUp": self.gen.localUp.state_dict(), "localDown": self.gen.localDown.state_dict()}, gen_HD_name
                )
                torch.save(
                    {"gen_HD": self.gen_opt_HD.state_dict(), "dis_HD": self.dis_HD_opt.state_dict()},
                    opt_name,
                )            
                #             else :
                #                 # Save generators, discriminators, and optimizers
                #                 gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations + 1))
                #                 dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations + 1))
                #                 domain_classifier_name = os.path.join(
                #                     snapshot_dir, "domain_classifier_%08d.pt" % (iterations + 1)
                #                 )
                #                 opt_name = os.path.join(snapshot_dir, "optimizer.pt")
                #                 if self.gen_state == 1:
                #                     torch.save({"2": self.gen.state_dict()}, gen_name)
                #                     torch.save({"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_name)
                #                 else:
                #                     print("self.gen_state unknown value:", self.gen_state)



class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters["lr"]
        # Initiate the networks
        self.gen_a = VAEGen(
            hyperparameters["input_dim_a"], hyperparameters["gen"]
        )  # auto-encoder for domain a
        self.gen_b = VAEGen(
            hyperparameters["input_dim_b"], hyperparameters["gen"]
        )  # auto-encoder for domain b
        self.dis_a = MsImageDis(
            hyperparameters["input_dim_a"], hyperparameters["dis"]
        )  # discriminator for domain a
        self.dis_b = MsImageDis(
            hyperparameters["input_dim_b"], hyperparameters["dis"]
        )  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters["init"]))
        self.dis_a.apply(weights_init("gaussian"))
        self.dis_b.apply(weights_init("gaussian"))

        # Load VGG model if needed
        if "vgg_w" in hyperparameters.keys() and hyperparameters["vgg_w"] > 0:
            self.vgg = load_vgg16(hyperparameters["vgg_model_path"] + "/models")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = (
            self.gen_a.decode(h_a_recon + n_a_recon)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )
        x_bab = (
            self.gen_b.decode(h_b_recon + n_b_recon)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
       
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = (
            self.compute_vgg_loss(self.vgg, x_ba, x_b)
            if hyperparameters["vgg_w"] > 0
            else 0
        )
        self.loss_gen_vgg_b = (
            self.compute_vgg_loss(self.vgg, x_ab, x_a)
            if hyperparameters["vgg_w"] > 0
            else 0
        )
        # total loss
        self.loss_gen_total = (
            hyperparameters["gan_w"] * self.loss_gen_adv_a
            + hyperparameters["gan_w"] * self.loss_gen_adv_b
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_a
            + hyperparameters["recon_kl_w"] * self.loss_gen_recon_kl_a
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_b
            + hyperparameters["recon_kl_w"] * self.loss_gen_recon_kl_b
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cyc_x_a
            + hyperparameters["recon_kl_cyc_w"] * self.loss_gen_recon_kl_cyc_aba
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cyc_x_b
            + hyperparameters["recon_kl_cyc_w"] * self.loss_gen_recon_kl_cyc_bab
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_a
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
        )
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean(
            (self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2
        )

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = (
            hyperparameters["gan_w"] * self.loss_dis_a
            + hyperparameters["gan_w"] * self.loss_dis_b
        )
        self.loss_dis_total.backward()
        self.dis_opt.step()


    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict["a"])
        self.gen_b.load_state_dict(state_dict["b"])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict["a"])
        self.dis_b.load_state_dict(state_dict["b"])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.dis_opt.load_state_dict(state_dict["dis"])
        self.gen_opt.load_state_dict(state_dict["gen"])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print("Resume from iteration %d" % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, "optimizer.pt")
        torch.save(
            {"a": self.gen_a.state_dict(), "b": self.gen_b.state_dict()}, gen_name
        )
        torch.save(
            {"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_name
        )
        torch.save(
            {"gen": self.gen_opt.state_dict(), "dis": self.dis_opt.state_dict()},
            opt_name,
        )

