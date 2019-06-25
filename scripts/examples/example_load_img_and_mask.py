import os
import sys
import numpy as np
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from utils_folder import visualize, coco_dataset, config
config = config.CocoConfig()



PREDICTED = True

if PREDICTED:
    image = np.load('../main/x_test.npy')
    mask = np.load('../main/y_pred.npy')
    image_id = np.load('../main/image_id.npy')

    image = np.squeeze(image)
    mask = np.squeeze(mask)
    mask = np.round(mask) * 255

    visualize.display_top_masks(image, mask, np.arange(81), config.class_names)

    COCO_DIR = "coco_dataset"
    dataset = coco_dataset.CocoDataset()
    dataset.load_coco(COCO_DIR, "val", year=2017)
    dataset.prepare()
    image_id = dataset.image_from_source_map['main.{}'.format(image_id)]

    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

else:
    COCO_DIR = "coco_dataset"
    dataset = coco_dataset.CocoDataset()
    dataset.load_coco(COCO_DIR, "val", year=2017)
    dataset.prepare()

    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


