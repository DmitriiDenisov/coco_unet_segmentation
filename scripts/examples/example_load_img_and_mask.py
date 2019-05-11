import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from utils_folder import visualize, config, coco_dataset

class_names = ['BG',
 'person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush']

PREDICTED = False

if PREDICTED:
    image = np.load('../main/x_test.npy')
    mask = np.load('../main/y_pred.npy')

    image = np.squeeze(image)
    mask = np.squeeze(mask)
    mask = np.round(mask) * 255

    class_names.extend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'f', 'q', 'e'])

    visualize.display_top_masks(image, mask, np.arange(91), class_names)
else:
    config = config.CocoConfig()
    COCO_DIR = "coco_dataset"
    if config.NAME == "main":
        dataset = coco_dataset.CocoDataset()
        dataset.load_coco(COCO_DIR, "val", year=2017)
    dataset.prepare()

    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


