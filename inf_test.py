import copy
import itertools
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from tqdm import tqdm
import cv2
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from mask_former import (
    add_mask_former_config,
)
from detectron2.data import MetadataCatalog, DatasetCatalog
# MaskFormer
import numpy as np
from detectron2.modeling import build_model
import albumentations as A
from albumentations.core.composition import Compose, OneOf

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
device="cuda:1"
cfg = get_cfg()
# for poly lr schedule
add_deeplab_config(cfg)
add_mask_former_config(cfg)
cfg.merge_from_file("configs/ade20k-150/swin/maskformer_swin_small_bs16_160k.yaml")
cfg.MODEL.DEVICE=device
model=build_model(cfg)

checkpoint=torch.load("output_split1/model_final.pth", map_location=device)

model.load_state_dict(checkpoint["model"])
path_img="../../../Data/liver_only_kfold/fold_1/images/"
path_msk="../../../Data/liver_only_kfold/fold_1/masks/0/"
image_names=os.listdir(path_img)

import time
from detectron2.data import detection_utils as utils
model.eval()
ious=[]
time_init=time.time()
for image_name in tqdm(image_names):
    data={}
    data["name"]=image_name
    #print(path_img+"images/"+image_name)
    image_test=cv2.imread(path_img+image_name,-1)
    sem_seg_gt = utils.read_image(path_msk+image_name).astype("double")[..., None]
    _,sem_seg_gt=cv2.threshold(sem_seg_gt,5,255,cv2.THRESH_BINARY)
    transform = Compose([
                    A.Resize(512,512),
                ])
    if image_test.ndim == 2:
                image_test = image_test[..., None]
    augmented = transform(image=image_test, mask=sem_seg_gt)
    image_test = augmented['image']
    sem_seg_gt = augmented['mask']
    image_test = image_test.astype('float32') / 255
    image_test = image_test.transpose(2, 0, 1)
    sem_seg_gt = sem_seg_gt.astype('float32') / 255
    image_test = torch.tensor(image_test).float()
    dataset_dict={}
    dataset_dict["image"] = image_test
    sem_seg_gt[sem_seg_gt>0]=1
    prediction=model([dataset_dict])[0]['sem_seg'].argmax(dim=0)

print("Tiempo y score: ",np.mean(ious),time.time()-time_init)
