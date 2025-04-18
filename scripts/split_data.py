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
TEST_DIR  = os.path.join(BASE_DIR, "test")

# 분할 비율: train : test = 10 : 1
# test_size = 전체 개수 // 11
# ────────────────────────────────────────────────────

def main():
    # 1) 원본 로딩
    imgs = np.load(os.path.join(SOURCE_DIR, "images.npy"))
    lbls = np.load(os.path.join(SOURCE_DIR, "labels.npy"))

    N = imgs.shape[0]
    np.random.seed(42)
    perm = np.random.permutation(N)

    test_size  = N // 11
    train_size = N - test_size

    train_idx = perm[:train_size]
    test_idx  = perm[train_size:]

    train_imgs = imgs[train_idx]
    train_lbls = lbls[train_idx]
    test_imgs  = imgs[test_idx]
    test_lbls  = lbls[test_idx]

    # 2) 디렉토리 준비
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR,  exist_ok=True)

    # 3) 저장
    np.save(os.path.join(TRAIN_DIR, "images.npy"), train_imgs)
    np.save(os.path.join(TRAIN_DIR, "labels.npy"), train_lbls)
    np.save(os.path.join(TEST_DIR,  "images.npy"), test_imgs)
    np.save(os.path.join(TEST_DIR,  "labels.npy"), test_lbls)

    print(f"총 샘플: {N}, train: {train_size}, test: {test_size}")
    print(f"Train 파일 → {TRAIN_DIR}")
    print(f"Test  파일 → {TEST_DIR}")

if __name__ == "__main__":
    main()
