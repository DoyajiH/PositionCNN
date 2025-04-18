#!/usr/bin/env python3
# scripts/split_dataset.py

import os
import numpy as np

# ─── 설정 ─────────────────────────────────────────────
# 프로젝트 루트 기준으로 datasets 디렉토리
BASE_DIR = os.path.join(os.getcwd(), "datasets")

# 원본 데이터가 있는 디렉토리
SOURCE_DIR = os.path.join(BASE_DIR, "total_frame_image_train")

# 결과를 저장할 디렉토리
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR  = os.path.join(BASE_DIR, "val")

# 분할 비율: train : val = 10 : 1
# val_size = 전체 개수 // 11
# ────────────────────────────────────────────────────

def main():
    # 1) 원본 로딩
    imgs = np.load(os.path.join(SOURCE_DIR, "vehicle_images.npy"))
    lbls = np.load(os.path.join(SOURCE_DIR, "vehicle_labels.npy"))

    N = imgs.shape[0]
    np.random.seed(42)
    perm = np.random.permutation(N)

    val_size  = N // 11
    train_size = N - val_size

    train_idx = perm[:train_size]
    val_idx  = perm[train_size:]

    train_imgs = imgs[train_idx]
    train_lbls = lbls[train_idx]
    val_imgs  = imgs[val_idx]
    val_lbls  = lbls[val_idx]

    # 2) 디렉토리 준비
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR,  exist_ok=True)

    # 3) 저장
    np.save(os.path.join(TRAIN_DIR, "vehicle_images.npy"), train_imgs)
    np.save(os.path.join(TRAIN_DIR, "vehicle_labels.npy"), train_lbls)
    np.save(os.path.join(VAL_DIR,  "vehicle_images.npy"), val_imgs)
    np.save(os.path.join(VAL_DIR,  "vehicle_labels.npy"), val_lbls)

    print(f"총 샘플: {N}, train: {train_size}, val: {val_size}")
    print(f"Train 파일 → {TRAIN_DIR}")
    print(f"Val  파일 → {VAL_DIR}")

if __name__ == "__main__":
    main()
