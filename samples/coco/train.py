from keras import Input
import os
import sys
from keras.callbacks import ModelCheckpoint

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать
from samples.coco.unet import get_unet
from samples.coco.generator import KerasGenerator
from samples.coco import coco
config = coco.CocoConfig()

img_size_target = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3)
# Если обучать с нуля:
input_img = Input(img_size_target, name='img')
model = get_unet(input_img, n_filters=8, dropout=0.05, batchnorm=True)
model.summary()

keras_gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_train2017.json',
                     dataset_dir='coco_dataset',
                     subset='train',
                     year='2017',
                     batch_size=4)
keras_gen.prepare()
gen = keras_gen.generate_batch()

model_check = ModelCheckpoint('../../models', monitor='loss', verbose=0, save_best_only=True)

history = model.fit_generator(gen,
                              steps_per_epoch=keras_gen.total_imgs // keras_gen.batch_size,
                              epochs=5,
                              callbacks=[model_check])
