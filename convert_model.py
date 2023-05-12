from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
# Lightning deepspeed has saved a directory instead of a file
convert_zero_checkpoint_to_fp32_state_dict(
    "/root/autodl-tmp/zoupeng/ControlNet/checkpoint-coco-batchsize32/lightning_logs/version_0/checkpoints/epoch=5-step=7854.ckpt",
    "/root/autodl-tmp/zoupeng/ControlNet/checkpoint-coco-batchsize32/lightning_logs/version_0/checkpoints/epoch=5-step=7854.pt",
)