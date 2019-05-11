import os
import sys
import numpy as np
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from utils_folder import visualize, utils, coco_dataset, config
config = config.CocoConfig()

PREDICTED = True

if PREDICTED:
    image = np.load('../main/x_test.npy')
    mask = np.load('../main/y_pred.npy')

    image = np.squeeze(image)
    mask = np.squeeze(mask)
    mask = np.round(mask) * 255

    bbox = utils.extract_bboxes(mask)
    # Display image and instances
    visualize.display_instances(image, bbox, mask, np.arange(81), config.class_names)
else:
    COCO_DIR = "coco_dataset"  # TODO: enter value here

    dataset = coco_dataset.CocoDataset()
    dataset.load_coco(COCO_DIR, "val", year=2017)
    # Must call before using the dataset
    dataset.prepare()

    # Load random image and mask.
    image_id = np.random.choice(dataset.image_ids, 1)[0]
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    # Resize
    image, window, scale, padding, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding)
    # Display image and additional stats
    print("image_id: ", image_id, dataset.image_reference(image_id))
    print("Original shape: ", original_shape)
    bbox = utils.extract_bboxes(mask)

    visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

 # Compute Bounding box





