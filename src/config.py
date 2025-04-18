import os

# Data file Paths
BASE_DATA_DIR = os.path.join(os.getcwd(), "dataset")

TRAIN_IMAGES_PATH = os.path.join(BASE_DATA_DIR, "train", "vehicle_images.npy")
TRAIN_LABELS_PATH = os.path.join(BASE_DATA_DIR, "train", "vehicle_labels.npy")

VAL_IMAGES_PATH   = os.path.join(BASE_DATA_DIR, "val",   "vehicle_images.npy")
VAL_LABELS_PATH   = os.path.join(BASE_DATA_DIR, "val",   "vehicle_labels.npy")

TEST_IMAGES_PATH  = os.path.join(BASE_DATA_DIR, "test",  "vehicle_images.npy")
TEST_LABELS_PATH  = os.path.join(BASE_DATA_DIR, "test",  "vehicle_labels.npy")

# Training Hyperparameters
BATCH_SIZE          = 512
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 40
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}

# Dataaset
IMG_HEIGHT = 240
IMG_WIDTH = 320
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.485, 0.456, 0.406]
IMAGE_STD           = [0.229, 0.224, 0.225]

# Network
MODEL_NAME          = 'resnet18'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
