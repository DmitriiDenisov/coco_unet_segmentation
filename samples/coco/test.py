from keras import Input
from tqdm import tqdm

from samples.coco.unet import get_unet
from samples.coco.try_generator import stupid_gen
from samples.coco import coco
config = coco.CocoConfig()

img_size_target = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3)
# Если обучать с нуля:
input_img = Input(img_size_target, name='img')
model = get_unet(input_img, n_filters=8, dropout=0.05, batchnorm=True)

gen = stupid_gen()
for i in tqdm(range(10)):
    next(gen)

