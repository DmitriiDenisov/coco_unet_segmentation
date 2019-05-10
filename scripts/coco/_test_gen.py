from keras import Input
from tqdm import tqdm

from scripts.coco.unet import get_unet
from scripts.examples.try_generator import stupid_gen
from scripts.coco import coco
config = coco.CocoConfig()

img_size_target = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3)
# Если обучать с нуля:
input_img = Input(img_size_target, name='img')
model = get_unet(input_img, n_filters=8, dropout=0.05, batchnorm=True)

gen = stupid_gen()
for i in tqdm(range(10)):
    next(gen)

