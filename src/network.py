# src/network.py
import time
import copy
import torch
from torch import nn
from torchvision import models

from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger

import wandb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import src.config as cfg
from src.util import show_setting

FONT = ImageFont.truetype("arial.ttf", size=16)

class ParkingPoseRegressor(LightningModule):
    def __init__(self,
                 model_name: str = cfg.MODEL_NAME,
                 optimizer_params: dict = cfg.OPTIMIZER_PARAMS,
                 scheduler_params: dict = cfg.SCHEDULER_PARAMS):
        super().__init__()
        # backbone + head
        backbone = models.get_model(model_name, weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 3)
        self.model = backbone

        # loss
        self.loss_fn = nn.MSELoss()

        # save hparams
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type   = optim_params.pop('type')
        optimizer    = getattr(torch.optim, optim_type)(
            self.parameters(), **optim_params
        )
        sched_params = copy.deepcopy(self.hparams.scheduler_params)
        sched_type   = sched_params.pop('type')
        scheduler    = getattr(torch.optim.lr_scheduler, sched_type)(
            optimizer, **sched_params
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor':   'loss/val',
            }
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y   = batch
        preds  = self(x)
        loss   = self.loss_fn(preds, y)
        self.log('loss/train', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        preds  = self(x)
        loss   = self.loss_fn(preds, y)
        self.log('loss/val', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._wandb_log_image(batch, batch_idx, preds, frequency=cfg.WANDB_IMG_LOG_FREQ)
        return loss

    # ─── 테스트 단계 전용 훅 ────────────────────────────────────

    def on_test_start(self):
        # 테스트 시작 시 빈 리스트 초기화
        self.all_preds  = []
        self.all_tgts   = []
        self.all_losses = []

    def test_step(self, batch, batch_idx):
        x, y   = batch
        preds  = self(x)
        loss   = self.loss_fn(preds, y)
        # test loss 로깅만
        self.log('loss/test', loss, on_step=False, on_epoch=True, prog_bar=True)

        # on_test_end 에서 사용할 데이터만 모아두기
        self.all_preds .append(preds.detach().cpu())
        self.all_tgts  .append(y.detach().cpu())
        self.all_losses.append(((preds - y)**2).detach().cpu())
        return loss

    def on_test_end(self):

        # ── 1) 전체 preds/tgts 수집 ───────────────────────────
        preds = torch.cat(self.all_preds).cpu().numpy()   # (N,3)
        tgts  = torch.cat(self.all_tgts ).cpu().numpy()  # (N,3)

        self.model.to('cpu')

        # ── 2) 전체 RMSE 계산 & 로깅 ─────────────────────────
        overall_rmse = np.sqrt(((preds - tgts) ** 2).mean())
        # Lightning 로그 대신 직접 wandb 로깅해도 됩니다
        self.logger.experiment.log({"overall_RMSE": overall_rmse})

        # ── 3) 샘플당 추론 시간 측정 & 로깅 ─────────────────────
        # 테스트용 배치에서 첫 배치만 꺼내서 측정
        dm = self.trainer.datamodule
        batch = next(iter(self.trainer.datamodule.test_dataloader()))
        imgs, _ = batch
        imgs = imgs.to('cpu')
        start = time.time()
        with torch.no_grad():
            _ = self.model(imgs)
        inference_time = (time.time() - start) / imgs.size(0)
        self.logger.experiment.log({"inference_time_per_sample": inference_time})

        # ── 4) ψ를 degree로 변환 ───────────────────────────────
        preds[:, 2] = np.rad2deg(preds[:, 2])
        tgts[:, 2]  = np.rad2deg(tgts[:, 2])

        # ── 5) GT vs Pred 그래프 ───────────────────────────────
        N  = preds.shape[0]
        dt = 0.05
        t  = np.arange(N) * dt

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        dims = ['x', 'y', 'ψ']
        units = ['[m]', '[m]', '[deg]']

        for i, ax in enumerate(axes):
            ax.plot(t, tgts[:, i], label=f'GT {dims[i]}',   color='red')
            ax.plot(t, preds[:, i], label=f'Pred {dims[i]}', color='blue')
            max_err = np.max(np.abs(preds[:, i] - tgts[:, i]))
            ax.text(0.02, 0.85,
                    f"MaxErr: {max_err:.4f}",
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
            ax.set_ylabel(f"{dims[i]} {units[i]}")
            ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Time [s]')
        fig.suptitle("GT vs Pred", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        self.logger.experiment.log({"chart/all": wandb.Image(fig)}, commit=True)
        plt.close(fig)

        # ── 6) 6개 샘플 그리드 ─────────────────────────────────
        N_ds = len(dm.test_ds)
        idxs = np.linspace(0, N_ds-1, num=6, dtype=int)
        grid = []
        for idx in idxs:
            img, tgt = dm.test_ds[idx]
            p = self(img.unsqueeze(0).to(self.device)).detach().cpu().numpy()[0]
            rmse = np.sqrt(((p - tgt.numpy())**2).mean())

            # PIL 변환
            raw = dm.test_ds.images[idx]
            arr = raw.transpose(1,2,0) if raw.shape[0] in (1,3) else raw
            pil = Image.fromarray(arr.astype('uint8'))
            draw = ImageDraw.Draw(pil)

            draw.text((5,5),
                      f"GT:  x={tgt[0]:.2f}, y={tgt[1]:.2f}, ψ={tgt[2]:.2f}",
                      font=FONT, fill=(255,0,0), stroke_width=1, stroke_fill=(0,0,0))
            draw.text((5,25),
                      f"Pr:  x={p[0]:.2f}, y={p[1]:.2f}, ψ={p[2]:.2f}",
                      font=FONT, fill=(0,255,0), stroke_width=1, stroke_fill=(0,0,0))
            draw.text((5,45),
                      f"RMSE: {rmse:.4f}",
                      font=FONT, fill=(0,0,255), stroke_width=1, stroke_fill=(0,0,0))

            grid.append(wandb.Image(pil, caption=f"sample{idx}"))

        self.logger.experiment.log({"pred/test_samples": grid}, commit=True)


    # ─── 이미지 로깅 함수 (재사용) ────────────────────────────────────

    def _wandb_log_image(self, batch, batch_idx, preds, frequency=100):
        if not isinstance(self.logger, WandbLogger):
            return
        if batch_idx % frequency != 0:
            return

        images, targets = batch
        images  = images.detach().cpu()
        targets = targets.detach().cpu().numpy()
        preds   = preds.detach().cpu().numpy()
        losses_RMSE  = np.sqrt(((preds - targets)**2).mean(axis=1))

        log_imgs = []
        for i in range(min(6, len(images))):
            img_np = images[i].permute(1,2,0).numpy()
            img_np = (img_np*255).clip(0,255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            gt_str   = f"GT:   x={targets[i][0]:.2f}, y={targets[i][1]:.2f}, ψ={targets[i][2]:.2f}"
            pred_str = f"Pred: x={preds[i][0]:.2f}, y={preds[i][1]:.2f}, ψ={preds[i][2]:.2f}"
            loss_RMSE_str = f"RMSE: {losses_RMSE[i]:.4f}"

            draw = ImageDraw.Draw(img_pil)
            draw.text((5,5),   gt_str,  font=FONT,
                      fill=(255,0,0), stroke_width=1, stroke_fill=(0,0,0))
            draw.text((5,25),  pred_str,font=FONT,
                      fill=(0,255,0), stroke_width=1, stroke_fill=(0,0,0))
            draw.text((5,45),  loss_RMSE_str,font=FONT,
                      fill=(0,0,255), stroke_width=1, stroke_fill=(0,0,0))

            log_imgs.append(wandb.Image(img_pil,
                              caption=f"batch{batch_idx}_sample{i}"))

        self.logger.experiment.log({"pred/val": log_imgs}, commit=True)
