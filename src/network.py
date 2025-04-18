# src/network.py

import copy
import torch
from torch import nn
from torchvision import models

from lightning.pytorch import LightningModule

import src.config as cfg
from src.util import show_setting

class ParkingPoseRegressor(LightningModule):
    """
    후방 카메라 이미지를 입력받아 차량의 위치 (x, y, psi)를 회귀 예측하는 모델
    """

    def __init__(self,
                 model_name: str = cfg.MODEL_NAME,
                 optimizer_params: dict = cfg.OPTIMIZER_PARAMS,
                 scheduler_params: dict = cfg.SCHEDULER_PARAMS):
        super().__init__()
        # 1) 백본 모델 로드 (사전 학습된 가중치 없이 새로 학습)
        backbone = models.get_model(model_name, weights=None)

        # 2) 마지막 FC 레이어를 회귀용 3개 출력으로 교체
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 3)
        self.model = backbone

        # 3) 손실 함수 및 평가 지표 설정
        self.loss_fn = nn.MSELoss()

        # 4) 하이퍼파라미터(optim, scheduler) 저장
        self.save_hyperparameters()

    def on_train_start(self):
        # 학습 시작 시 config 값 출력
        show_setting(cfg)

    def configure_optimizers(self):
        # Optimizer 설정
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type   = optim_params.pop('type')
        optimizer    = getattr(torch.optim, optim_type)(
            self.parameters(), **optim_params
        )

        # Scheduler 설정
        sched_params = copy.deepcopy(self.hparams.scheduler_params)
        sched_type   = sched_params.pop('type')
        scheduler    = getattr(torch.optim.lr_scheduler, sched_type)(
            optimizer, **sched_params
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor':   'loss/val',  # validation loss 모니터링
            }
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 순전파: (B, 3) 크기의 텐서 반환 (x, y, psi)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y   = batch
        preds  = self(x)
        loss   = self.loss_fn(preds, y)
        self.log('loss/train', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        preds  = self(x)
        loss   = self.loss_fn(preds, y)
        self.log('loss/val', loss, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y   = batch
        preds  = self(x)
        loss   = self.loss_fn(preds, y)
        self.log('loss/test', loss, on_epoch=True, prog_bar=True)
        return {'test_loss': loss}
