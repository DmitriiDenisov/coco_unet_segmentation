import os
import sys
import numpy as np
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from utils_folder import visualize, config

# Run one of the code blocks

# Shapes toy dataset
# import shapes
# config = shapes.ShapesConfig()

# MS COCO Dataset
config = config.CocoConfig()
COCO_DIR = "coco_dataset"  # TODO: enter value here

# Load dataset
if config.NAME == "coco":
    dataset = config.CocoDataset()
    dataset.load_coco(COCO_DIR, "val", year=2017)

# Must call before using the dataset
dataset.prepare()


# Load and display random scripts
# dataset.coco.imgs[dataset.image_info[image_id]['id']]['coco_url']

image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
