import os
import sys
print(f"Installing into {os.getcwd()}")  # may not be the same as sys.path[0]

# Install pytorch and torchvision; install pycocotools
# os.system("pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html ; \
# pip install cython pyyaml==5.1; \
# pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")

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
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# Convert VOC format to COCO format
os.system("python voc2coco.py --ann_dir=data/train --labels=data/labels.txt --output=data/train_annots.json")
os.system("python voc2coco.py --ann_dir=data/validate --labels=data/labels.txt --output=data/validate_annots.json")
os.system("python voc2coco.py --ann_dir=data/test --labels=data/labels.txt --output=data/test_annots.json")
register_coco_instances("train", {}, "data/train_annots.json", "data/train")
register_coco_instances("validate", {}, "data/validate_annots.json", "data/validate")
register_coco_instances("test", {}, "data/test_annots.json", "data/test")

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
cfg.MODEL.DEVICE = "cpu" # TODO: without this, mac gets an error
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
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # no of classes (no need to +1 for background)
cfg.TEST.EVAL_PERIOD = 500  # No. of iterations after which the Validation Set is evaluated.


# We need to make sure that the model validates against our validation set. 
# Unfortunately, this does not happen by default unless we subclass the DefaultTrainer
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
print("Done training")

#############################################
#  EVALUATE THE TRAINED MODEL ON THE TEST SET
register_coco_instances("test", {}, "data/test_annots.json", "data/test")

# change the training configuration a bit
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85 

# Create a simple end-to-end predictor with the given config that runs on single device for a single input image.
# Compared to using the model directly, this class does the following additions: Load checkpoint from cfg.MODEL.WEIGHTS.
predictor = DefaultPredictor(cfg) 
 
# The evaluator helps us calcuate AP
evaluator = COCOEvaluator("test", cfg, False, output_dir="./output/")

# DataLoader object for the test set. The training and validation sets did not explicity need this because it was specified in the CocoTrainer cfg 
test_loader = build_detection_test_loader(cfg, "test")

# Evaluate the AP
inference_on_dataset(trainer.model, test_loader, evaluator)

##########################################
#  VISUALIZE THE INFERENCE ON THE TEST SET

# Edit the config a bit
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("test", ) # TODO: Not sure why this was not necessary when evaluating AP on the test set (see above)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model

# Create a simple end-to-end predictor again
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("test")

from detectron2.utils.visualizer import ColorMode
import glob
for imageName in glob.glob('data/test/*jpg'):
	im = cv2.imread(imageName)
	outputs = predictor(im)
	v = Visualizer(im[:, :, ::-1],
			metadata=test_metadata, 
			scale=0.8)
	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

	cv2.imshow("", out.get_image()[:, :, ::-1])
	cv2.waitKey() # press key to exit

