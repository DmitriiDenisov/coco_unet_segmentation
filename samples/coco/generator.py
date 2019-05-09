from __future__ import division, print_function, unicode_literals
from itertools import islice

import os
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать

from mrcnn import utils
import skimage.color
import skimage.io
import skimage.transform

from sacred import Experiment
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from samples.coco import coco
config = coco.CocoConfig()

data_coco = Experiment("dataset")


def ids():
    return [0,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
            74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

class KerasGenerator:
    def __init__(self, annFile, batch_size, dataset_dir, subset, year, shuffle=True):
        self.coco = COCO(annFile)
        self.batch_size = batch_size
        self.total_imgs = len(self.coco.imgToAnns.keys())
        self.image_info = dict()
        self._image_ids = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.all_images_ids = list(self.coco.imgToAnns.keys())
        self.shuffle = shuffle
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.year = year

        class_ids = sorted(self.coco.getCatIds())
        image_ids = []
        for id in class_ids:
            image_ids.extend(list(self.coco.getImgIds(catIds=[id])))
        # Remove duplicates
        image_ids = list(set(image_ids))
        # Add images
        for i in image_ids:

            image_dir = "{}/images/{}{}".format(self.dataset_dir, self.subset, self.year)

            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, self.coco.imgs[i]['file_name']),
                width=self.coco.imgs[i]["width"],
                height=self.coco.imgs[i]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)),
            file_name=self.coco.imgs[i]['file_name'])

    def add_image(self, source, image_id, path, **kwargs):
        image_info_temp = {
            "id": image_id,
            "source": source,
            "path": path}
        image_info_temp.update(kwargs)
        self.image_info[image_id] = image_info_temp

    @property
    def image_ids(self):
        return self._image_ids

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)


        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        path = os.path.join('../../', self.image_info[image_id]['path'])

        try:
            image = skimage.io.imread(path)
        except:
            file_name = self.image_info[image_id]['file_name']
            file_path = "../../{}/images/{}{}/{}".format(self.dataset_dir, self.subset, self.year, file_name)
            url = self.coco.imgs[self.image_info[image_id]['id']]['coco_url']
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

    @data_coco.command
    def generate_batch(self):
        idx_global = 0
        while True:
            # Если конец эпохи:
            if idx_global == 0 or idx_global > len(self.all_images_ids):
                idx_global = 0
                images_ids = np.copy(self.all_images_ids)
                if self.shuffle:
                    np.random.shuffle(images_ids)
                iterable = np.copy(images_ids)
                i = iter(iterable)
                batch_train_indecies = list(islice(i, self.batch_size))
            batch_x = np.array([])
            batch_y = np.array([])
            # Цикл по батчу
            for id_image in batch_train_indecies:
                # 'Converting Annotations to Segmentation Masks...'
                img = self.coco.loadImgs(int(id_image))[0]

                if True:
                    target_shape = (img['height'], img['width'], max(ids()) + 1)
                ann_ids = self.coco.getAnnIds(imgIds=id_image, iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)
                mask_one_hot = np.zeros(target_shape, dtype=np.uint8)
                mask_one_hot[:, :, 0] = 1  # every pixel begins as background

                for ann in anns:
                    mask_partial = self.coco.annToMask(ann)
                    # mask_partial = cv2.resize(mask_partial,
                    #                           (target_shape[1], target_shape[0]),
                    #                           interpolation=cv2.INTER_NEAREST)
                    mask_one_hot[mask_partial > 0, ann['category_id']] = 1
                    mask_one_hot[mask_partial > 0, 0] = 0

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
            yield batch_x, batch_y
            batch_train_indecies = list(islice(i, self.batch_size))



