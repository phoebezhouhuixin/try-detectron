import os
import sys
print(f"Installing into {os.getcwd()}")  # may not be the same as sys.path[0]

# Install pytorch and torchvision; install pycocotools
os.system("pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html ; \
pip install cython pyyaml==5.1; \
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")

# Install detectron2
# os.system("git clone https://github.com/facebookresearch/detectron2.git ;\
# cd detectron2 && pip install -e .")

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# Convert VOC format to COCO format
os.system("python voc2coco.py --ann_dir=data/train --labels=data/labels.txt --output=data/train_annots.json")

register_coco_instances("train", {}, "data/train_annots.json", "data/train")

dataset_dicts = DatasetCatalog.get("train")
print(dataset_dicts[0])
trainset_metadata = MetadataCatalog.get("train")
# import random
# for d in random.sample(dataset_dicts, 1):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=trainset_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow("image", vis.get_image()[:, :, ::-1])
#     cv2.waitKey()  # press to exit

cfg = get_cfg()

# Get the basic model configuration from the model zoo 
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

print(f"Default train dataset: {cfg.DATASETS.TRAIN}")
print(f"Default test dataset: {cfg.DATASETS.TEST}")
print(f"Default number of data loading threads: {cfg.DATALOADER.NUM_WORKERS}")
print(f"Default model weights: {cfg.MODEL.WEIGHTS}")
print(f"Default number of images per batch:{cfg.SOLVER.IMS_PER_BATCH}")
print(f"Default max epochs??? solver.max_iter: {cfg.SOLVER.MAX_ITER}")
print(f"Default number of ROIs per image (i.e. batch size for roi head): {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}")
# Total number of RoIs per training minibatch = ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
print(f"Default num classes at roi heads: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
print(f"Default period (in terms of steps) to evaluate the model during training: {cfg.TEST.EVAL_PERIOD}")

# Passing the Train and Validation sets
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("validate",) # supposed to refer to the validation set, not the test set
# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 4
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  
# Number of images per batch across all machines.
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0125  # pick a good LearningRate
cfg.SOLVER.MAX_ITER = 1500  # No. of iterations   
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # No. of classes = [HINDI, ENGLISH, OTHER]
cfg.TEST.EVAL_PERIOD = 500  # No. of iterations after which the Validation Set is evaluated.

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
