from __future__ import division, print_function, unicode_literals
from itertools import islice

import os
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать

from utils_folder import utils, config
import skimage.color
import skimage.io
import skimage.transform
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

config = config.CocoConfig()


class KerasGenerator:
    def __init__(self, annFile, batch_size, dataset_dir, subset, year, shuffle=True):
        self.coco = COCO(annFile)
        self.batch_size = batch_size
        self.num_cats = len(list(self.coco.cats.keys())) # без background !
        self.total_imgs = len(self.coco.imgToAnns.keys())
        self.all_images_ids = list(self.coco.imgToAnns.keys())
        self.shuffle = shuffle
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.year = year

        self.map_id_cat = {cat_id: i for i, cat_id in enumerate(list(self.coco.cats.keys()))} # нулевой слой больше не BackGround

        # Add paths
        image_dir = "{}/images/{}{}".format(self.dataset_dir, self.subset, self.year)
        for i in self.all_images_ids:
            self.coco.imgs[i]['path'] = os.path.join(image_dir, self.coco.imgs[i]['file_name'])

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image

        path = os.path.join('../../', self.coco.imgs[image_id]['path'])

        try:
            image = skimage.io.imread(path)
        except:
            file_name = self.coco.imgs[image_id]['file_name']
            file_path = "../../{}/images/{}{}/{}".format(self.dataset_dir, self.subset, self.year, file_name)
            url = self.coco.imgs[self.coco.imgs[image_id]['id']]['coco_url']
            image = skimage.io.imread(url)
            im = Image.fromarray(image)
            im.save(file_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def generate_batch(self):
        idx_global = 0
        while True:
            # Если конец эпохи:
            if idx_global == 0 or idx_global >= len(self.all_images_ids):
                idx_global = 0
                images_ids = np.copy(self.all_images_ids)
                if self.shuffle:
                    np.random.shuffle(images_ids)
                iterable = np.copy(images_ids)
                i = iter(iterable)
                batch_train_indecies = list(islice(i, self.batch_size))
                self.batch_train_indecies = batch_train_indecies
            batch_x = np.array([])
            batch_y = np.array([])
            # Цикл по батчу
            for id_image in batch_train_indecies:
                # 'Converting Annotations to Segmentation Masks...'
                img = self.coco.imgs[id_image]

                if True:
                    target_shape = (img['height'], img['width'],  self.num_cats)
                ann_ids = self.coco.getAnnIds(imgIds=id_image, iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)
                mask_one_hot = np.zeros(target_shape, dtype=np.uint8)
                # mask_one_hot[:, :, 0] = 1  # every pixel begins as background

                for ann in anns:
                    mask_partial = self.coco.annToMask(ann)
                    # mask_partial = cv2.resize(mask_partial,
                    #                           (target_shape[1], target_shape[0]),
                    #                           interpolation=cv2.INTER_NEAREST)
                    mask_one_hot[mask_partial > 0, self.map_id_cat[ann['category_id']]] = 1
                    # mask_one_hot[mask_partial > 0, 0] = 0 # Соотв. пиксели в нулевом слое (background) обнуляются

                # load_image
                img = self.load_image(image_id=id_image)

                # Rescale image and mask:
                image, window, scale, padding, _ = utils.resize_image(
                    img,
                    min_dim=config.IMAGE_MIN_DIM,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE)
                mask = utils.resize_mask(mask_one_hot, scale, padding)

                # Закидываем в батч:
                image = np.expand_dims(image, axis=0)
                mask = np.expand_dims(mask, axis=0)

                try:
                    batch_x = np.concatenate([batch_x, image], axis=0)
                    batch_y = np.concatenate([batch_y, mask], axis=0)
                except:
                    batch_x = np.copy(image)
                    batch_y = np.copy(mask)

                idx_global += 1
                del mask_one_hot
            yield batch_x, batch_y
            batch_train_indecies = list(islice(i, self.batch_size))
            self.batch_train_indecies = batch_train_indecies
