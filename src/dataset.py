import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import src.config as cfg
from lightning.pytorch import LightningDataModule

class ParkingDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert len(images) == len(labels), "images/labels 길이 불일치"
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        # 전체 샘플 개수
        return len(self.labels)

    def __getitem__(self, idx):
        # 1) NumPy 배열 → PIL.Image
        img = Image.fromarray(self.images[idx].astype('uint8'))
        # 2) transform 적용 (크기 조정 → Tensor → 정규화)
        if self.transform:
            img = self.transform(img)
        # 3) 레이블 → FloatTensor([x, y, psi])
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, lbl

class ParkingDataModule(LightningDataModule):
    def __init__(self,
                 train_images_path: str,
                 train_labels_path: str,
                 val_images_path: str,
                 val_labels_path: str,
                 test_images_path: str,
                 test_labels_path: str,
                 batch_size: int = cfg.BATCH_SIZE,
                 num_workers: int = cfg.NUM_WORKERS):
        super().__init__()
        # 파일 경로 저장
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.val_images_path   = val_images_path
        self.val_labels_path   = val_labels_path
        self.test_images_path  = test_images_path
        self.test_labels_path  = test_labels_path

        # config에서 DataLoader 파라미터
        self.batch_size  = batch_size
        self.num_workers = num_workers

        # config에서 전처리 파라미터
        self.transform = transforms.Compose([
            transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.IMAGE_MEAN,
                                 std=cfg.IMAGE_STD),
        ])

    def setup(self, stage=None):
        # ── fit 단계(train+val) ───────────────────
        if stage in (None, 'fit'):
            train_imgs = np.load(self.train_images_path)
            train_lbls = np.load(self.train_labels_path)
            val_imgs   = np.load(self.val_images_path)
            val_lbls   = np.load(self.val_labels_path)

            self.train_ds = ParkingDataset(train_imgs, train_lbls,
                                           transform=self.transform)
            self.val_ds   = ParkingDataset(val_imgs,   val_lbls,
                                           transform=self.transform)

        # ── test 단계 ───────────────────
        if stage in (None, 'test'):
            test_imgs = np.load(self.test_images_path)
            test_lbls = np.load(self.test_labels_path)
            self.test_ds = ParkingDataset(test_imgs, test_lbls,
                                          transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)
