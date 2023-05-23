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

apply_openpose = OpenposeDetector()
model = create_model('/root/data/code/ControlNet/models/cldm_v15.yaml').cpu()
# print(model)
model.load_state_dict(load_state_dict('/root/data/code/ControlNet/models/control_sd15_openpose.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

same_index = [36]

def mse(imgA, imgB):
    err = np.sum((imgA.astype("float")-imgB.astype("float"))**2)
    err /= float(imgA.shape[0]*imgA.shape[1])
    return err

with torch.no_grad():
    for i in range(len(same_index)):
        file = str(same_index[i]).zfill(12)+'.jpg'
        txt_name = file.replace('.jpg','.txt')
        # 这是一张图片，可以进行相应的操作
        imgpth='/root/data/code/ori_img/'
        htpth='/root/data/code/hq/'
        txtpth = '/root/data/code/coco_txt/'

        image_path = os.path.join(imgpth, file)
        detected_map_path = os.path.join(htpth, file)
        txt_file = os.path.join(txtpth,txt_name)

        input_image = cv2.imread(image_path)  # 读取通道为BGR 需要转换成RGB
        detected_map = cv2.imread(detected_map_path)

        with open(txt_file,'r') as f:
            prompt = f.read()

        n_prompt = '(((simple background))),(semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck'

        a_prompt = 'masterpiece, (photorealistic:1.5), best quality, beautiful lighting, real life'
        # n_prompt = '(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark,extra person,extra arm,extra leg, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy'

        num_samples = 1
        image_resolution = 512
        ddim_steps = 30
        guess_mode = False
        strength = 1.3
        scale = 13
        seed = -1
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
            H, W, C = input_image.shape
            H1,W1,C1 = detected_map.shape
            new_img = cv2.resize(results, (W, H), interpolation=cv2.INTER_NEAREST)
            detect_resolution = W1
            gcmap, _ = apply_openpose(cv2.resize(input_image, (W1, H1), interpolation=cv2.INTER_NEAREST))
            detected_map_gray = cv2.cvtColor(detected_map,cv2.COLOR_BGR2GRAY)
            gcmap_gray = cv2.cvtColor(gcmap,cv2.COLOR_BGR2GRAY)
            pic_same = mse(detected_map_gray,gcmap_gray)

            print(pic_same)
            cv2.imwrite("/root/data/code/ControlNet/result/"+'det_htm' + file, gcmap_gray)
            cv2.imwrite("/root/data/code/ControlNet/result/"+'ori_htm' + file, detected_map_gray)
            cv2.imwrite("/root/data/code/ControlNet/result/" + file, new_img[:, :, [2, 1, 0]])

