import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch, argparse
from src.network import ParkingPoseRegressor
import src.config as cfg

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str,
    default=os.path.join(os.path.dirname(__file__),
                         '..','wandb','PositionCNN','irojkmbm','checkpoints',
                         'epoch=36-step=666.ckpt'))
args = parser.parse_args()

# 1) LightningModule 로딩
model = ParkingPoseRegressor(model_name=cfg.MODEL_NAME)

# 2) 실제 경로로 체크포인트 로드
print(f"🔍 Loading checkpoint from: {args.ckpt}")
ckpt = torch.load(args.ckpt, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 3) 순수 nn.Module 꺼내기
base_model = model.model

# 4) Trace & 저장
example = torch.randn(1, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
traced  = torch.jit.trace(base_model, example)
traced.save("parking_cnn_model.pt")

print("✅ Saved TorchScript model to parking_cnn_model.pt")
