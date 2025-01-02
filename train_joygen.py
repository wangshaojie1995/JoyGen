# Copyright (c) 2025 JD.com, Inc. and affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import accelerate
from packaging import version
from utils.dataset import FaceDataset
from torch.utils.data import Dataset, DataLoader
import time
import torch.optim as optim

from src.modules.unet import UNet,PositionalEncoding
from torchvision import transforms
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from diffusers import AutoencoderKL, UNet2DConditionModel
import json
from accelerate import Accelerator
import math
from diffusers.training_utils import EMAModel
from safetensors.torch import load_file

import logging
from accelerate.logging import get_logger
from omegaconf import OmegaConf
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)


class JoyGen(object):
    def __init__(self, config_file):
        self.cfg = OmegaConf.load(config_file)
        self.mask_tensor = self.get_mask_tensor() 
        print(self.cfg) 

    def get_mask_tensor(self):
        """
        Creates a mask tensor for image processing.
        :return: A mask tensor.
        """
        mask_tensor = torch.zeros((self.cfg.dataset.img_size,self.cfg.dataset.img_size))
        mask_tensor[:self.cfg.dataset.img_size//2,:] = 1
        mask_tensor[mask_tensor< 0.5] = 0
        mask_tensor[mask_tensor>= 0.5] = 1
        return mask_tensor


    def preprocess_image_tensor(self, image_tensor, half_mask=False):
        image_tensor = image_tensor[:,:,:,[2,1,0]]/255
        image_tensor = image_tensor.permute(0, 3, 1, 2)
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if half_mask:
            image_tensor = image_tensor * self.mask_tensor.to(image_tensor.device)
        #print(image_tensor.shape, self.mask_tensor.shape)
        image_tensor = transform(image_tensor)
        return image_tensor

    def preprocess_depth_tensor(self, image_tensor):
        image_tensor = image_tensor[:,:,:,[2,1,0]]/255
        image_tensor = image_tensor.permute(0, 3, 1, 2)
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        image_tensor = image_tensor * (1-self.mask_tensor.to(image_tensor.device))
        #print(image_tensor.shape, self.mask_tensor.shape)
        image_tensor = transform(image_tensor)
        return image_tensor
    
    def pil_to_opencv(self, img_tensor):
        tensor_list = []
        for ind in range(img_tensor.shape[0]):
            tensor = img_tensor[ind,:,:,:].squeeze()
            tensor = transforms.ToPILImage()(tensor)
            tensor = np.array(tensor)
            tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
            tensor_list.append(torch.tensor(tensor).to(torch.float).unsqueeze(dim=0))
        image_tensor = torch.cat(tensor_list, dim=0)
        return image_tensor

    def vae_decode(self, vae,  latents):
        latents = (1/ vae.config.scaling_factor) * latents
        image = vae.decode(latents.to(vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = image[...,::-1]
        return image
    
    
    def validation(self, 
                   vae: torch.nn.Module,
                   pe: torch.nn.Module,
                   unet:torch.nn.Module, 
                   unet_config,
                   weight_dtype: torch.dtype,
                   val_loader,
                   epoch_train,
                   step_train,
         ):
        
         # Get the validation pipeline
        unet_copy = UNet2DConditionModel(**unet_config)
        unet_copy.load_state_dict(unet.state_dict())
        unet_copy.to(vae.device).to(dtype=weight_dtype)
        unet_copy.eval()
        pe.to(vae.device, dtype=weight_dtype)

        path_image_show = os.path.join(self.cfg.checkpoint_dir, "val_images", f"{epoch_train}_{step_train}")
        if not os.path.exists(path_image_show):
            os.makedirs(path_image_show)
        
        start_time = time.time()
        with torch.no_grad():
            for step, (whisper, gt_ref_face_enhance, depth) in enumerate(val_loader):
                # split
                gt_ref_face_enhance_opencv = self.pil_to_opencv(gt_ref_face_enhance)
                tensor_split = torch.split(gt_ref_face_enhance_opencv, self.cfg.dataset.img_size, dim=-3)
                gt_face = tensor_split[0]
                ref_face = tensor_split[1]
                depth = self.pil_to_opencv(depth)

                # audio feature
                whisper = pe(whisper.to(vae.device, dtype=weight_dtype))       
            
                # 预处理VAE图片输入
                ref_image = self.preprocess_image_tensor(ref_face, half_mask=False).to(vae.device)
                gt_image_nomask = self.preprocess_image_tensor(gt_face, half_mask=False).to(vae.device)
                gt_image_mask = self.preprocess_image_tensor(gt_face, half_mask=True).to(vae.device)
                depth_image_mask = self.preprocess_depth_tensor(depth).to(vae.device)
                
                # VAE编码 
                ref_latent = vae.config.scaling_factor * vae.encode(ref_image.to(dtype=weight_dtype)).latent_dist.sample() 
                gt_latent_nomask = vae.config.scaling_factor * vae.encode(gt_image_nomask.to(weight_dtype)).latent_dist.sample() 
                gt_latent_mask = vae.config.scaling_factor * vae.encode(gt_image_mask.to(weight_dtype)).latent_dist.sample() 
                depth_latent_mask = vae.config.scaling_factor * vae.encode(depth_image_mask.to(weight_dtype)).latent_dist.sample() 
                input_latent = torch.cat([gt_latent_mask, ref_latent, depth_latent_mask], dim=1)

                # 单步unet
                timesteps = torch.tensor([0], device=vae.device)
                latent_pred = unet(input_latent, timesteps, encoder_hidden_states=whisper).sample
                    
                # 转化为opencv格式
                h,w = self.cfg.dataset.img_size, self.cfg.dataset.img_size
                image_pred = self.vae_decode(vae, latent_pred)
                image_gt_from_vae = self.vae_decode(vae, gt_latent_nomask)
                image_gt_mask_from_vae = self.vae_decode(vae, gt_latent_mask)
                image_ref_from_vae = self.vae_decode(vae, ref_latent)
                depth_from_vae = self.vae_decode(vae, depth_latent_mask)
                image_show = np.zeros((self.cfg.dataset.img_size*image_pred.shape[0], self.cfg.dataset.img_size*5, 3), np.uint8)
                for idx in range(image_pred.shape[0]):
                    image_show[idx*h:(idx+1)*h, 0:w, :] = image_ref_from_vae[idx]
                    image_show[idx*h:(idx+1)*h, w:2*w, :] = image_gt_mask_from_vae[idx]
                    image_show[idx*h:(idx+1)*h, 2*w:3*w, :] = image_gt_from_vae[idx]
                    image_show[idx*h:(idx+1)*h, 3*w:4*w, :] = image_pred[idx]
                    #image_show[idx*h:(idx+1)*h, 4*w:5*w, :] = depth[idx].detach().cpu().numpy().astype(np.uint8)
                    image_show[idx*h:(idx+1)*h, 4*w:5*w, :] = depth_from_vae[idx]
                cv2.imwrite(os.path.join(path_image_show, f"{step}.jpg"), image_show)
                end_time = time.time()
                logger.info(f"validation step: {step}, time:{end_time-start_time}")
                start_time = time.time()
                
    
    def train(self):
        # accelerator
        accelerator = Accelerator(gradient_accumulation_steps=self.cfg.opti.gradient_accumulation_steps)
       
        # vae 
        vae = AutoencoderKL.from_pretrained(self.cfg.vae_path)
       
        # unet 
        with open(self.cfg.unet_config_file, 'r') as f:
            unet_config = json.load(f)
        unet = UNet2DConditionModel(**unet_config)

        # initialize weights of unet
        if os.path.exists(self.cfg.unet_resume_file):
            if self.cfg.unet_resume_file.endswith("safetensors"):
                state_dict = load_file(self.cfg.unet_resume_file)
                unet.load_state_dict(state_dict)
            else:
                state_dict = torch.load(self.cfg.unet_resume_file)
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                unet.load_state_dict(new_state_dict)
            logger.info(f"initialize model's weights from {self.cfg.unet_resume_file}", main_process_only=True)
        else:
            logger.info(f"not initialize model's weight", main_process_only=True)

        # create ema for the unet
        if self.cfg.opti.use_ema:
            ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        pe = PositionalEncoding(d_model=384)

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if self.cfg.opti.use_ema:
                        ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()


            def load_model_hook(models, input_dir):
                if self.cfg.opti.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                    ema_unet.load_state_dict(load_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del load_model

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)




        train_dataset = FaceDataset(
            self.cfg.dataset.root_dir, 
            self.cfg.dataset.train_file, 
            self.cfg.dataset.img_size
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.cfg.dataset.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.dataset.num_workers
        )  

        val_dataset = FaceDataset(
            self.cfg.dataset.root_dir, 
            self.cfg.dataset.val_file, 
            self.cfg.dataset.img_size
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.dataset.num_workers
        )   


        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            unet.parameters(),
            lr=self.cfg.opti.lr,
            betas=(0.9, 0.99),
            weight_decay=0.01,
            eps=1e-8,
        )
        """
        lr_scheduler = get_scheduler(
            'cosine',
            optimizer=optimizer,
            num_warmup_steps=self.warmup_step * accelerator.num_processes,
            num_training_steps=self.max_step * accelerator.num_processes,
        )
        """
        num_warmup_steps=self.cfg.opti.warmup_steps * accelerator.num_processes
        num_training_steps=self.cfg.opti.max_steps * accelerator.num_processes

        # 预热阶段的学习率调度器（线性增加）
        def lr_lambda(step):
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda)

        # 余弦退火阶段的学习率调度器
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=self.cfg.opti.min_lr)

        # 组合调度器
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps])



        unet, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_loader, val_loader, lr_scheduler
        )

        l1_loss_func = torch.nn.L1Loss()
        vae.requires_grad_(False)
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16   
        logger.info(f"precision: {accelerator.mixed_precision}, dtyp: {weight_dtype}", main_process_only=True) 

        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        pe.to(accelerator.device, dtype=weight_dtype)

        if self.cfg.opti.use_ema:
            ema_unet.to(accelerator.device, dtype=weight_dtype)
       
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.cfg.opti.gradient_accumulation_steps)
        num_train_epochs = math.ceil(self.cfg.opti.max_steps / num_update_steps_per_epoch)
        total_batch_size = self.cfg.dataset.batch_size * accelerator.num_processes * self.cfg.opti.gradient_accumulation_steps

        logger.info("***** Running training *****", main_process_only=True)
        logger.info(f"  Num examples = {len(train_dataset)}", main_process_only=True)
        logger.info(f"  Num batches each epoch = {len(train_loader)}", main_process_only=True)
        logger.info(f"  Num Epochs = {num_train_epochs}", main_process_only=True)
        logger.info(f"  Instantaneous batch size per device = {self.cfg.dataset.batch_size}", main_process_only=True)
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}", main_process_only=True)
        logger.info(f"  Gradient Accumulation steps = {self.cfg.opti.gradient_accumulation_steps}", main_process_only=True)
        logger.info(f"  Total optimization steps = {self.cfg.opti.max_steps}", main_process_only=True)



        global_step = 0
        accumulation_step = 0

        for epoch in range(0, num_train_epochs):
            unet.train()
            time1 = time.time()
            for step, (whisper, gt_ref_face_enhance, depth) in enumerate(train_loader):
                with accelerator.accumulate(unet):
                    t3 = time.time()
                    # 
                    gt_ref_face_enhance_opencv = self.pil_to_opencv(gt_ref_face_enhance)
                    tensor_split = torch.split(gt_ref_face_enhance_opencv, self.cfg.dataset.img_size, dim=-3)
                    gt_face = tensor_split[0]
                    ref_face = tensor_split[1]
                    depth = self.pil_to_opencv(depth)
 
                    whisper = pe(whisper.to(vae.device, dtype=weight_dtype))      
                    t4 = time.time()
            
                    # 预处理VAE图片输入
                    ref_image = self.preprocess_image_tensor(ref_face, half_mask=False).to(vae.device)
                    gt_image_nomask = self.preprocess_image_tensor(gt_face, half_mask=False).to(vae.device)
                    gt_image_mask = self.preprocess_image_tensor(gt_face, half_mask=True).to(vae.device)
                    depth_image_mask = self.preprocess_depth_tensor(depth).to(vae.device) 
 
                    # VAE编码 
                    ref_latent = vae.config.scaling_factor * vae.encode(ref_image.to(dtype=weight_dtype)).latent_dist.sample() 
                    gt_latent_nomask = vae.config.scaling_factor * vae.encode(gt_image_nomask.to(weight_dtype)).latent_dist.sample() 
                    gt_latent_mask = vae.config.scaling_factor * vae.encode(gt_image_mask.to(weight_dtype)).latent_dist.sample() 
                    depth_latent_mask = vae.config.scaling_factor * vae.encode(depth_image_mask.to(weight_dtype)).latent_dist.sample() 
                    input_latent = torch.cat([gt_latent_mask, ref_latent, depth_latent_mask], dim=1)
                    t5 = time.time()

                    # 单步unet
                    timesteps = torch.tensor([0], device=vae.device)
                    latent_pred = unet(input_latent, timesteps, encoder_hidden_states=whisper).sample
                    
                    image_pred = (1 / vae.config.scaling_factor) * latent_pred
                    image_pred = vae.decode(image_pred).sample

                    # 取下半张脸
                    image_pred_half = image_pred[:, :, image_pred.shape[2]//2:, :]
                    image_gt_half = gt_image_nomask[:, :, gt_image_nomask.shape[2]//2:, :]
                    
                    l1 = l1_loss_func(latent_pred, gt_latent_nomask)
                    # 像素空间进行下半张脸对比
                    l2 = l1_loss_func(image_pred_half, image_gt_half)       

                    alpha = 1
                    beta = 2

                    loss = alpha * l1 + beta * l2
                    t6 = time.time()
                    
                    # backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        max_grad_norm = 1.0
                        accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm) 
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    t7 = time.time()

                    if accelerator.sync_gradients:
                        global_step += 1
                        if self.cfg.opti.use_ema:
                            ema_unet.step(unet.parameters())

                    if step % self.cfg.console_log_interval == 0:     
                        time2 = time.time()
                        during_time = time2 - time1
                        samples_speed = (self.cfg.dataset.batch_size * accelerator.num_processes * self.cfg.console_log_interval) / during_time
                        log_str = f"step: {step}, global_step: {global_step}, num_step_per_epoch: {len(train_loader)}, epoch: {epoch}, latent loss: {l1:.5f}, alpha: {alpha}, latent loss(*{alpha}): {alpha*l1:.5f}, face loss: {l2:.5f}, face loss(*{beta}): {beta*l2:.5f}, total loss: {loss:.5f}, lr: {lr_scheduler.get_last_lr()[0]:.8f}, during time: {during_time:.2f}, samples/sec: {samples_speed:.2f}"
                        logger.info(log_str, main_process_only=True)
                        time1 = time.time()

                    if step > 0 and step % self.cfg.checkpoint_valiation_interval == 0:
                        if accelerator.is_main_process:
                            self.validation(
                                vae=accelerator.unwrap_model(vae),
                                pe=pe,
                                unet=accelerator.unwrap_model(unet),
                                unet_config=unet_config,
                                weight_dtype=weight_dtype,
                                val_loader=val_loader,
                                epoch_train=epoch,
                                step_train=step,
                            )
     
                    if global_step > 0 and global_step % (self.cfg.checkpoint_save_interval)== 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(self.cfg.checkpoint_dir, f"{epoch}_{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}", main_process_only=True)

                    if global_step >= self.cfg.opti.max_steps:
                        break
                    accumulation_step += 1 
            accelerator.wait_for_everyone()
        accelerator.end_training()


if __name__ == "__main__":
    joygen = JoyGen("./config/joygen.yaml")
    joygen.train()