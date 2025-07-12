import itertools
import json
import logging
import numpy as np
import os
import csv
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator


class SemSegIoU(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        csv_data,
        csv_epoch
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
        """
        print("NEW IOU!!!!!!!!!\n!!!!!!!!!!!\n!!!!!!!!!!\n!!!!!!!!!!!")
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self.csv_data=csv_data
        self.csv_epoch=csv_epoch

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        #print("CLASES: ",self._class_names)
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label
        #print("ignore: ",self._ignore_label)
        
        self.iou=[]
        self.dice=[]
        
        self.name=os.path.join(self._output_dir, "log.csv")
        self.name_CSV=os.path.join(self._output_dir, "log_CSV.csv")
        #file=open(self.name, 'a')
        #writer = csv.DictWriter(file, fieldnames=["mIoU"])
        #writer.writeheader()
        #file.close()



    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            #print("INPUT:",input["file_name"])
            outputs = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(outputs, dtype=int)
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(np.load(f), dtype=int)
            """
            print("before")
            print("output shape:",np.shape(output["sem_seg"].to(self._cpu_device)))
            print("output:",np.unique(output["sem_seg"].to(self._cpu_device)))
            print("pred shape:",np.shape(pred))
            print("pred:",np.unique(pred))
            print("gt:",np.unique(gt))
            """
            pred[pred > 0] = 1
            gt[gt > 0] = 1
            """
            print("after")
            print("pred:",np.unique(pred))
            print("gt:",np.unique(gt))
            """
            IoU_value  = self.binary_jc(pred, gt)
            DICE_value = self.binary_dc(pred, gt) 
            
            self.iou.append(IoU_value)
            self.dice.append(DICE_value)
            #print("IOU",IoU_value)
            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def writeCSV(self,data):
        #myfile=open(name,'r+')
        #writer=csv.DictWriter(myfile,fieldnames=list(newLine.keys()))
        #writer.writerow(newLine)
        #myfile.close()
        with open(self.name, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            #writer.writeheader()
            writer.writerow(data)        

    def binary_dc(self,result, reference):
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

    def binary_jc(self,result, reference):
        result = np.atleast_1d(result.astype(bool))
        reference = np.atleast_1d(reference.astype(bool))
        intersection = np.count_nonzero(result & reference)
        union = np.count_nonzero(result | reference) 
        try:
            jc = float(intersection) / float(union)
        except ZeroDivisionError:
            jc = 0.0
        return jc


    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
    
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        res = {}
        res["mIoU"] = np.mean(self.iou)
        res["mDICE"] = np.mean(self.iou)
        self.csv_data["epoch"].append(self.csv_epoch)
        self.csv_data["iou"].append(res["mIoU"])
        self.csv_data["dice"].append(res["mDICE"])
        self.csv_epoch=self.csv_epoch+1
        pd.DataFrame(self.csv_data).to_csv(self.name_CSV,index=True)
        
        print("---Metrics (Image)---")
        print("IoU  : %.4f" % (res["mIoU"]))
        print("DICE  : %.4f" % (res["mDICE"]))

        
        self.iou=[]
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            self.writeCSV(res)
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list


