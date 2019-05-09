from keras import Input
import os
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать
from samples.coco.unet import get_unet
from samples.coco.try_generator import stupid_gen
from samples.coco import coco
config = coco.CocoConfig()

img_size_target = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3)
# Если обучать с нуля:
input_img = Input(img_size_target, name='img')
model = get_unet(input_img, n_filters=8, dropout=0.05, batchnorm=True)
model.summary()

tr_gen = stupid_gen()

history = model.fit_generator(tr_gen, steps_per_epoch=7000 // 32)