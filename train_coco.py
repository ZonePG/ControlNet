from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint


# Configs
batch_size = 16
logger_freq = 600
learning_rate = 1e-6
sd_locked = True
only_mid_control = False

model_name = "control_v11p_sd15_openpose"
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(
    f"/root/autodl-tmp/zoupeng/ControlNet/models/{model_name}.yaml"
).cpu()
model.load_state_dict(
    load_state_dict(
        "/root/autodl-tmp/zoupeng/ControlNet/models/v1-5-pruned.ckpt", location="cuda"
    ),
    strict=False,
)
model.load_state_dict(
    load_state_dict(
        f"/root/autodl-tmp/zoupeng/ControlNet/models/{model_name}.pth", location="cuda"
    ),
    strict=False,
)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
strategy = DeepSpeedStrategy()
dataset = MyDataset("/root/autodl-tmp/coco_controlnet")
dataloader = DataLoader(dataset, num_workers=16, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
    strategy=strategy,
    devices=[0, 1],
    precision=32,
    callbacks=[logger, ModelCheckpoint(save_top_k=-1)],
    default_root_dir="/root/autodl-tmp/zoupeng/ControlNet/checkpoint-coco-batchsize32",
)


# Train!
trainer.fit(model, dataloader)
