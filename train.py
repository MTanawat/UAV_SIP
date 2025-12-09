import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

warnings.filterwarnings('ignore')
os.environ["WANDB_MODE"] = "disabled"


def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 获取当前脚本所在的目录
    current_dir = Path(__file__).parent
    # 构建相对路径
    yaml_path = '/project/lt200246-mmacma/nuke/swamp/UAVSwarm-dataset/yolov12/air_bird.yaml'
    check_path(yaml_path)
    # model = RTDETR('ultralytics/cfg/models/uavdetr-r50.yaml')
    model = RTDETR('/project/lt200246-mmacma/nuke/swamp/UAV-DETR/ultralytics/cfg/models/uavdetr-r50.yaml')
    model.train(data=str(yaml_path),
                cache=False,
                imgsz=640,
                epochs=50,
                batch=16,
                workers=8,
                device='0',
                # resume='', # last.pt path
                project='train640',
                name='exp',
                patience = 40, # early stopping patience
                )