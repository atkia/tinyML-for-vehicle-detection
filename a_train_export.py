import torch
from PIL.Image import module

from ultralytics import YOLO 
import wandb
import torch.nn.utils.prune as prune
load = False
exp_id = 'exp'

version = 'v9'

# if version == 'v1':
#     print('Please, check to modify ultralytics/nn/modules/head/Detect')
#     print('for TinyissimoYOLOv1.3 small and big change')
#     print('line 36 to: self.reg_max=16')
#     exit(1)

device = torch.device("cuda")
if load:
    model_name = f'./results/{exp_id}/weights/last.pt'
    model = YOLO(model_name) 
else:
    # model_name = f"./ultralytics/cfg/models/tinyissimo/tinyissimo-{version}.yaml"
    # model_name = f"./ultralytics/cfg/models/v9/yolov9c.yaml"
    model_name = f"./ultralytics/cfg/models/tinyissimo/our_v9c_m.yaml"
    model = YOLO(model_name)


img_size =96
input_size = (1, 1, img_size, img_size)  
 
# Train
model.train(data="custom.yaml",  project="Custom_vehicle", name="mod-our-v9c_m", optimizer='SGD',  imgsz=img_size,  epochs=100,  batch=64)

# Export
model.export(format="onnx", project="results", name="mod-our-v9c_m", imgsz=[img_size,img_size])