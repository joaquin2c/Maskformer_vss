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

from detectron2.data import MetadataCatalog, DatasetCatalog
# MaskFormer
from mask_former import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapperVSS,
    SemanticSegmentorWithTTA,
    add_mask_former_config
)
import numpy as np
from detectron2.modeling import build_model
import albumentations as A
from albumentations.core.composition import Compose, OneOf


import time
from detectron2.data import detection_utils as utils
import pandas as pd
from skimage.morphology import skeletonize


cfg = get_cfg()
# for poly lr schedule
add_deeplab_config(cfg)
add_mask_former_config(cfg)
cfg.merge_from_file("configs/ade20k-150/swin/maskformer_swin_base_IN21k_384_bs16_160k_res640.yaml")




def get_pred(model,img,msk):
    MEAN=[0.485,0.456,0.406]
    STD=[0.229,0.224,0.225]
    
    image = img.astype(np.float32)
    img_min, img_max = -360, 440
    image = np.clip(image, img_min, img_max)
    image = (image - img_min) / (img_max - img_min)
    sem_seg_gt = msk.astype("double")
    
    if image.ndim == 2:
        image = image[..., None]
        image=image.repeat(3,axis=-1)
    
    transform = Compose([
        A.Resize(512,512),
    ])
    
    image=((image)-MEAN)/STD
    
    augmented = transform(image=image, mask=sem_seg_gt)
    image_test = augmented['image']
    sem_seg_gt = augmented['mask']
    
    sem_seg_gt[sem_seg_gt>0]=1
    image_test = image_test.transpose(2, 0, 1).astype(np.float32) 
    
    
    image_test = torch.tensor(image_test)
    dataset_dict={}
    dataset_dict["image"] = image_test
    with torch.no_grad():
        prediction=model([dataset_dict])[0]['sem_seg'].argmax(dim=0)
    prediction=prediction.detach().cpu().numpy()
    return prediction, sem_seg_gt


def create_dict(image_names):
    dict_info={"name":[],"number":[],"slice":[]}
    for i in image_names:
        info=str(i.split("-")[-1])
        numbers=info.split("_")
        patient=numbers[0]
        slide=numbers[1].split(".")[0]
        dict_info["number"].append(patient)
        dict_info["slice"].append(slide.zfill(4))
        dict_info["name"].append(i)
    dataframe=pd.DataFrame(dict_info)
    dataframe=dataframe.sort_values(by=["number","slice"])
    return dataframe

def get_scores(model,number,dataframe,path_img,path_msk):
    masks_patient=[]
    preds_patient=[]
    
    patient_imgs_slices=dataframe[dataframe["number"]==number]["name"]
    for patient_slice in patient_imgs_slices:
        name_img=os.path.join(path_img,patient_slice)
        name_msk=os.path.join(path_msk,patient_slice)
        img=np.load(name_img)
        msk=np.load(name_msk)
        prediction, sem_seg_gt=get_pred(model,img,msk)
        masks_patient.append(sem_seg_gt)
        preds_patient.append(prediction)
    cl_score_patient=cl_score(np.array(preds_patient), np.array(masks_patient))
    clDice_patient=clDice(np.array(preds_patient), np.array(masks_patient))
    binary_dc_patient=binary_dc(np.array(preds_patient), np.array(masks_patient))
    return cl_score_patient,clDice_patient,binary_dc_patient


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    cl = np.sum(v*s)/np.sum(s+1e-12)
    return cl


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    tprec = cl_score(v_p,skeletonize(v_l))
    tsens = cl_score(v_l,skeletonize(v_p))
    cl_dice = 2*tprec*tsens/(tprec+tsens+1e-12)
    return cl_dice

def binary_dc(result, reference):
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:A is the first and :math:B the second set of samples (here: binary objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in 
result
 and the
        object(s) in 
reference
. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval maskformer model")
    parser.add_argument("-i", "--imgpth", type=str, default="../../Data/colorectal_fold/fold_1/train/images",
                        help="image path")
    parser.add_argument("-m", "--mskpth", type=str, default="../../Data/colorectal_fold/fold_1/train/masks",
                        help="masks path")
    parser.add_argument("-mpth", "--model_path", type=str, default="",
                        help="model path")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="gpu device")
    args = parser.parse_args()

    path_img_train = args.imgpth
    path_msk_train = args.mskpth
    model_path = args.model_path
    gpu = args.gpu

    results={"patient":[],"cl_score":[],"clDice":[],"binary_dc":[]}
    
    image_names_train=os.listdir(path_img_train)
    
    dict_train=create_dict(image_names_train)
    
    patients_train=np.unique(dict_train["number"])

    model=build_model(cfg)
    checkpoint=torch.load(model_path, map_location=f"cuda:{gpu}")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    cl_score_train=[]
    clDice_patient_train=[]
    binary_dc_patient_train=[]
    
    for number_train in patients_train:
        cl_score_patient,clDice_patient,binary_dc_patient=get_scores(model,number_train,dict_train,path_img_train,path_msk_train)
        cl_score_train.append(cl_score_patient)
        clDice_patient_train.append(clDice_patient)
        binary_dc_patient_train.append(binary_dc_patient)
        results["patient"].append(number_train)
        results["cl_score_patient"].append(cl_score_patient)
        results["clDice_patient"].append(clDice_patient)
        results["binary_dc_patient"].append(binary_dc_patient)
    cl_score=np.mean(cl_score_train)
    clDice=np.mean(clDice_patient_train)
    binary_dc=np.mean(binary_dc_patient_train))

    print("Results:\n")
    print("cl_score:",cl_score)
    print("clDice:",clDice)
    print("binary_dc:",binary_dc)
    
    dataframe_final=pd.DataFrame(results)
    dataframe_final.to_csv('results.csv', index=False)  