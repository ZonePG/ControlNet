from share import *
import config
import json
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from collections import defaultdict
import time as time
import itertools
import os

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model_name = "control_v11p_sd15_openpose"
model_config_name = f"/root/autodl-tmp/models/{model_name}.yaml"
# model_checkpoint_name = "/root/autodl-tmp/zoupeng/ControlNet/checkpoint-coco-epoch4-batchsize16/lightning_logs/version_0/checkpoints/epoch=4-step=13084.ckpt"
model_checkpoint_name = "/root/autodl-tmp/models/epoch=26-step=35343.pt"
base_model_checkpoint_name = "/root/autodl-tmp/models/v1-5-pruned.ckpt"
model = create_model(model_config_name).cpu()
model.load_state_dict(
    load_state_dict(base_model_checkpoint_name, location="cuda"),
    strict=False,
)
model.load_state_dict(
    load_state_dict(model_checkpoint_name, location="cuda"),
    strict=False,
)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

tt = 0
with torch.no_grad():
    folder_path = '/root/autodl-tmp/coco_controlnet/hq'
    pc_name = os.listdir(folder_path)
    # del pc_name[0:3130] 
    for nn in range(len(pc_name)):
        # 判断文件是否是图片
        file = pc_name[nn]
        tt+=1
        print(tt)
        if file.endswith('.jpg') or file.endswith('.png'):
            # 这是一张图片，可以进行相应的操作
            imgpth='/root/autodl-tmp/coco_controlnet/ori_img/'
            htpth='/root/autodl-tmp/coco_controlnet/hq/'
            txtpth = '/root/autodl-tmp/coco_controlnet/coco_txt/'
            txt_name = file.replace('.jpg','.txt')

            image_path = os.path.join(imgpth, file)
            detected_map_path = os.path.join(htpth, file)
            txt_file = os.path.join(txtpth,txt_name)

            input_image = cv2.imread(image_path)  # 读取通道为BGR 需要转换成RGB
            detected_map = cv2.imread(detected_map_path)

            with open(txt_file,'r') as f:
                prompt = f.read()
            # prompt = ''
            a_prompt = 'masterpiece, (photorealistic:1.5), best quality, beautiful lighting, real life'
            n_prompt = '(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark,extra person,extra arm,extra leg, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy'
            # a_prompt = ''
            # n_prompt = ''
            num_samples = 1
            image_resolution = 512
            ddim_steps = 28
            guess_mode = False
            strength = 1.3
            scale = 9
            seed = 0
            eta = 0.0

            input_image = HWC3(input_image)
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)

            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                            shape, cond, verbose=False, eta=eta,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=un_cond)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            for i in range(num_samples):
                results = x_samples[i]
                cv2.imwrite("./" + file, results[:, :, [2, 1, 0]])

