import os
import sys
import numpy as np
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from utils import visualize

# Run one of the code blocks

# Shapes toy dataset
# import shapes
# config = shapes.ShapesConfig()

# MS COCO Dataset
from scripts.coco import coco
config = coco.CocoConfig()
COCO_DIR = "coco_dataset"  # TODO: enter value here

# Load dataset
if config.NAME == "coco":
    dataset = coco.CocoDataset()
    dataset.load_coco(COCO_DIR, "val", year=2017)

# Must call before using the dataset
dataset.prepare()

x_test = np.load('x_test.npy')
y_pred = np.load('y_pred.npy')


# Load and display random scripts
# dataset.coco.imgs[dataset.image_info[image_id]['id']]['coco_url']

# image_ids = np.random.choice(dataset.image_ids, 4)

visualize.display_top_masks(np.squeeze(x_test), np.round(np.squeeze(y_pred)) * 255, range(90), dataset.class_names)
