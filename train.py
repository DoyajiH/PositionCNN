"""
    [AUE8088] PA1: Image Classification
        - To run: (PositionCNN) $ python train.py
        - For better flexibility, consider using LightningCLI in PyTorch Lightning
"""
# PyTorch & Pytorch Lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning import Trainer
import torch

# Custom packages
from src.dataset import ParkingDataModule
from src.network import ParkingPoseRegressor
import src.config as cfg

torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":

    model = ParkingPoseRegressor(
        model_name       = cfg.MODEL_NAME,
        optimizer_params = cfg.OPTIMIZER_PARAMS,
        scheduler_params = cfg.SCHEDULER_PARAMS,
    )

    datamodule = ParkingDataModule(
        train_images_path=cfg.TRAIN_IMAGES_PATH,
        train_labels_path=cfg.TRAIN_LABELS_PATH,
        val_images_path=  cfg.VAL_IMAGES_PATH,
        val_labels_path=  cfg.VAL_LABELS_PATH,
        test_images_path= cfg.TEST_IMAGES_PATH,
        test_labels_path= cfg.TEST_LABELS_PATH,
    )

    wandb_logger = WandbLogger(
        project = cfg.WANDB_PROJECT,
        save_dir = cfg.WANDB_SAVE_DIR,
        entity = cfg.WANDB_ENTITY,
        name = cfg.WANDB_NAME,
        group = cfg.WANDB_NAME,
    )

    trainer = Trainer(
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        max_epochs = cfg.NUM_EPOCHS,
        check_val_every_n_epoch = cfg.VAL_EVERY_N_EPOCH,
        logger = wandb_logger,
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1, monitor='loss/val', mode='min'),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(ckpt_path='best', datamodule=datamodule)
