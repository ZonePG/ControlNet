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


preprocessor = None

model_name = "control_v11p_sd15_openpose"
model_config_name = f"/root/autodl-tmp/zoupeng/ControlNet/models/{model_name}.yaml"
base_model_checkpoint_name = "/root/autodl-tmp/zoupeng/ControlNet/models/v1-5-pruned.ckpt"

# fix
model_checkpoint_name = "/root/autodl-tmp/zoupeng/ControlNet/checkpoint-coco-batchsize32/lightning_logs/version_0/checkpoints/epoch=26-step=35343.pt"
SAVE_PATH = "/root/autodl-tmp/coco_controlnet/gclist-epoch5-batchsize32"
# with open("/root/code/1_3people.txt", "r") as f:
#     S = f.read().replace(",", "")
#     O = S.split()
#     same_index = list(int(i) for i in O)

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

os.makedirs(SAVE_PATH, exist_ok=True)
num = 0
with torch.no_grad():
    file = "test-big.jpg"
    detected_map = cv2.imread(file)
    input_image = cv2.imread(file)
    prompt = "There are one people in the picture, One is jumping."
    a_prompt = "masterpiece, (photorealistic:1.5), best quality, beautiful lighting, real life"
    n_prompt = "(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark,extra person,extra arm,extra leg, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy"
    num_samples = 1

    ddim_steps = 50
    guess_mode = False
    strength = 1.5
    scale = 9
    seed = 12953
    eta = 0.0

    input_image = HWC3(input_image)
    detected_map = HWC3(detected_map)
    ############################################
    pixel_perfect = True
    resize_mode = "Resize and Fill"
    if pixel_perfect:
        raw_H, raw_W, _ = detected_map.shape
        target_H, target_W = raw_H, raw_W  # 希望输入图像的大小跟输出的大小一致

        k0 = float(target_H) / float(raw_H)
        k1 = float(target_W) / float(raw_W)

        if resize_mode == "Resize and Fill":
            estimation = min(k0, k1) * float(min(raw_H, raw_W))
        else:
            estimation = max(k0, k1) * float(min(raw_H, raw_W))

        preprocessor_resolution = int(np.round(float(estimation) / 64.0)) * 64
        print(f"estimation = {estimation}")
        print(f"H W = {raw_H,raw_W}")

    print(f"preprocessor resolution = {preprocessor_resolution}")
    #########################################################

    img = resize_image(input_image, preprocessor_resolution)

    H, W, C = img.shape
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {
        "c_concat": [control],
        "c_crossattn": [
            model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
        ],
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
    }
    shape = (4, H // 8, W // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = (
        [strength * (0.825 ** float(12 - i)) for i in range(13)]
        if guess_mode
        else ([strength] * 13)
    )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        num_samples,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
    )

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    for i in range(num_samples):
        results = x_samples[i]
        H1, W1, C = input_image.shape
        k = float(preprocessor_resolution) / min(H1, W1)
        new_img = cv2.resize(results, (W1, H1), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    cv2.imwrite("write.jpg", new_img[:, :, [2, 1, 0]])
