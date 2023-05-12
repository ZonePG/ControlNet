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
same_index = sorted([48636, 38029, 74369, 272110, 360926, 438723, 477010, 362368, 430359, 503707, 232309, 560323, 323682, 199602, 142667, 426175, 568623, 41687, 356708, 318671, 461063, 309120, 511204, 124629, 503005, 78707, 204049, 550444, 535668, 247285, 513681, 519706, 287570, 113914, 506552, 335578, 65415, 557556, 489023, 522778, 229643, 500062, 538859, 562150, 381106, 517822, 54277, 190081, 138975, 351683, 181714, 196053, 62060, 43635, 404464, 461885, 505099, 132615, 569768, 146193, 175831, 384553, 293554, 427135, 420775, 272262, 341041, 189845, 377385, 375521, 52759, 235597, 1369, 430125, 249720, 557981, 521400, 65227, 140007, 449981, 338948, 553442, 295589, 280819, 293802, 476280, 139530, 284350, 248242, 336493, 198448, 284220, 515289, 278506, 496768, 216228, 226278, 377715, 504304, 529917, 327807, 227879, 136501, 10442, 450500, 554114, 308441, 336182, 165029, 268556, 564745, 303298, 503311, 345998, 157269, 579056, 480726, 182245, 93725, 122851, 240028, 410632, 282225, 426523, 153570, 463836, 99734, 170629, 85160, 400538, 578292, 353807, 76292, 5802, 577826, 483234, 54796, 234572, 352194, 237669, 5064, 554625, 231163, 406404, 328289, 224757, 329717, 314154, 20965, 150410, 212663, 453757, 369826, 276580, 62790, 184613, 471009, 387173, 571034, 288403, 462565, 319908, 127474, 317595, 127451, 372938, 437609, 511117, 260166, 14990, 466211, 542510, 317560, 493174, 551334, 263136, 321674, 368402, 197254, 46859, 497312, 32965, 137003, 434494, 289423, 191096, 554348, 347170, 146723, 57387, 16497, 185036, 394892, 235302, 369763, 165133, 261779, 481635, 53015, 391735, 37038, 376549, 383419, 412151, 420532, 455859, 501762, 315601, 485709, 59202, 308026, 483108, 46743, 321107, 391895, 440329, 467477, 560108, 472854, 300814, 429580, 112085, 204805, 167854, 573349, 195829, 365366, 6005, 514083, 418949, 291380, 385861, 577858, 450263, 569839, 413321, 536831, 448269, 295837, 158015, 109819, 457754, 562614, 34180, 72833, 299116, 122934, 365426, 72944, 289444, 384012, 262873, 334321, 285291, 146627, 574769, 45864, 154971, 16574, 459912, 98590, 299319, 113588, 570579, 66423, 188958, 421010, 97434, 147980, 236189, 103579, 108094, 316795, 335472, 10142, 66412, 527023, 173704, 541010, 182784,161586, 360716, 108056, 45920, 500814, 156375, 256035, 440045, 5038, 267643, 331250, 576607, 461491, 183843, 5247, 432239, 336166, 425462, 177407, 539888, 267251, 538092, 252020, 281111, 461860, 159372, 186026, 356708, 331403, 298331, 360182, 8923, 330348, 113989, 542938, 549683, 336802, 296848, 562345, 255112, 395665, 161079, 519706, 304008, 530619, 264853, 62198, 186711, 246649, 515219, 527718, 250594, 155873, 499252, 136285, 283186, 331180, 270738, 211722, 571012, 355137, 202825, 418092, 449191, 382797, 99615, 251404, 74832, 297676, 416059, 54277, 526576, 463325, 64902, 165012, 305195, 149500, 190081, 552188, 223276, 494139, 450599, 303320, 330754, 196053, 564163, 475398, 371135, 342711, 84460, 32947, 421131, 478550, 25005, 448175, 203734, 53431, 511066, 154254, 301670, 499755, 189845, 170181, 505788, 483368, 150358, 112905, 388654, 81303, 357096, 192656, 544334, 158887, 320715, 140007, 308630, 460676, 134285, 402405, 563680, 408327, 431208, 14874, 258078, 522163, 537124, 284350, 140860, 432146, 35313, 576463, 244151, 259513, 12269, 299631, 376521, 579073, 578344, 442962, 554114, 264568, 578427, 216198, 289263, 361382, 134551, 554582, 313169, 195408, 181786, 75595, 476925, 240137, 322937, 513129, 240028, 1948, 29080, 282225, 362369, 107360, 47498, 182167, 208135, 141509, 320864, 578292, 374368, 445567, 279420, 324634, 455414, 430259, 25864, 356937, 451951, 33645, 405529, 275034, 459301, 77473, 33900, 478621, 508119, 103873, 200058, 192095, 378334, 543692, 295092, 261050, 160351, 403672, 127451, 194499, 510182, 382848, 289152, 507935, 511117, 443429, 13892, 512116, 434805, 150533, 166624, 241318, 366502, 479379, 462486, 194525, 229889, 428015, 435136, 111109, 450400, 528091, 265374, 363126, 347170, 331196, 437732, 145378, 390435, 323552, 475856, 165133, 238065, 112066, 525705, 311309, 95051, 184810, 220277, 62878, 199403, 207545, 476569, 519744, 434976, 501762, 254638, 199540, 185988, 465986, 304834, 480663, 333691, 275393, 207178, 34234, 404367, 40016, 170000, 269311, 357356, 509192, 551107, 390585, 84097, 258399, 259345, 516813, 179199, 45388, 560108, 316107, 558406, 426453, 367706, 114158, 514083, 444719, 375755, 248701, 511622, 232054, 536831, 315908, 426070, 546029, 553852, 33158, 458275, 207239, 158015, 450567, 187352, 345961, 535106, 502766, 48169, 97479, 524047, 560718, 350668, 144379, 410339, 404613, 323288, 486172, 523696, 279259, 539079, 15017, 246064, 512139, 303342, 343394, 535506, 240340, 320106, 555131, 143333, 271639, 559470, 530998, 536791, 460781, 187244, 177194, 515792, 526892, 241174, 102903, 63549, 446473, 316795, 15559, 277533, 335472, 334713, 517807, 508822, 32812, 391365, 44029, 505242, 112830, 191000, 71699, 259037, 25508])

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
    for i in range(len(same_index)):
        num += 1
        print(num)
        file = str(same_index[i]).zfill(12) + ".jpg"
        print("img_name:", file)
        txt_name = file.replace(".jpg", ".txt")
        # 这是一张图片，可以进行相应的操作
        imgpth = "/root/autodl-tmp/coco_controlnet/ori_img/"
        htpth = "/root/autodl-tmp/coco_controlnet/hq/"
        txtpth = "/root/autodl-tmp/coco_controlnet/coco_txt/"

        file = "test-small.jpg"

        # image_path = os.path.join(imgpth, file)
        detected_map_path = os.path.join(htpth, file)
        txt_file = os.path.join(txtpth, txt_name)

        # input_image = cv2.imread(image_path)  # 读取通道为BGR 需要转换成RGB
        detected_map = cv2.imread(detected_map_path)
        # print(image_path)
        print(detected_map_path)
        with open(txt_file, "r") as f:
            prompt = f.read()

        prompt = "There are one people in the picture, One is jumping."
        a_prompt = "masterpiece, (photorealistic:1.5), best quality, beautiful lighting, real life"
        n_prompt = "(painting by bad-artist-anime:0.9), (painting by bad-artist:0.9), watermark,extra person,extra arm,extra leg, text, error, blurry, jpeg artifacts, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, (worst quality, low quality:1.4), bad anatomy"
        num_samples = 1

        ddim_steps = 50
        guess_mode = False
        strength = 1.5
        scale = 9
        seed = 0
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
            # cv2.imwrite(os.path.join(SAVE_PATH, file), new_img[:, :, [2, 1, 0]])
            cv2.imwrite("write.jpg", new_img[:, :, [2, 1, 0]])
