"""
    [PositionCNN] Parking Position
        - To run: (PositionCNN) $ python test.py --ckpt_file wandb/PositionCNN/ygeiua2t/checkpoints/epoch=19-step=62500.ckpt
"""
# Python packages
import argparse

# PyTorch & Pytorch Lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning import Trainer
from torch.utils.flop_counter import FlopCounterMode
import torch

# Custom packages
from src.dataset import ParkingDataModule
from src.network import ParkingPoseRegressor
import src.config as cfg

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--ckpt_file',
        type = str,
        help = 'Model checkpoint file name')
    args = args.parse_args()

    model = ParkingPoseRegressor(
        model_name       = cfg.MODEL_NAME,
    )

    datamodule = ParkingDataModule(
        train_images_path=cfg.TRAIN_IMAGES_PATH,
        train_labels_path=cfg.TRAIN_LABELS_PATH,
        val_images_path=  cfg.VAL_IMAGES_PATH,
        val_labels_path=  cfg.VAL_LABELS_PATH,
        test_images_path= cfg.TEST_IMAGES_PATH,
        test_labels_path= cfg.TEST_LABELS_PATH,
        batch_size = cfg.BATCH_SIZE,
    )

    wandb_logger = WandbLogger(
        project = cfg.WANDB_PROJECT,
        save_dir = cfg.WANDB_SAVE_DIR,
        entity = cfg.WANDB_ENTITY,
        name = f"(test){cfg.WANDB_NAME}",
        group = cfg.WANDB_NAME,
    )

    trainer = Trainer(
        accelerator = 'cpu',
        devices = 1,
        precision = 32,
        benchmark = False,
        inference_mode = True,
        logger = wandb_logger,
    )

    trainer.test(model, ckpt_path = args.ckpt_file, datamodule = datamodule)

    # FLOP counter
    x, y = next(iter(datamodule.test_dataloader()))
    flop_counter = FlopCounterMode(model, depth=1)

    with flop_counter:
        model(x)
