import os
import sys

# Root directory of the project
from samples.coco.keras_coco import coco_json_to_segmentation

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


from samples.coco import coco

config = coco.CocoConfig()
COCO_DIR = "coco_dataset"  # TODO: enter value here


# Load dataset
if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "coco":
    dataset = coco.CocoDataset()
    dataset.load_coco(COCO_DIR, "val", year=2017)
# Must call before using the dataset
dataset.prepare()


coco_json_to_segmentation(seg_mask_image_paths=['../../coco_dataset/val2017'],
                          annotation_paths=['../../coco_dataset/annotations/instances_val2017.json'],
                          seg_mask_output_paths=['../../coco_dataset/seg_masks'],
                          verbose=1)