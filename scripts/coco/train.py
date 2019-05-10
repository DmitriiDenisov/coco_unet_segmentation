import os
import sys
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать
from keras import Input
from keras.callbacks import ModelCheckpoint
from utils.logger import TensorBoardBatchLogger
from scripts.coco.unet import get_unet
from scripts.coco.generator import KerasGenerator
from scripts.coco import coco
config = coco.CocoConfig()

# Генератор данных:
keras_gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_val2017.json',
                     dataset_dir='coco_dataset',
                     subset='val',
                     year='2017',
                     batch_size=4)
gen = keras_gen.generate_batch()

# Сетка:
img_size_target = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3)
input_img = Input(img_size_target, name='img')
model = get_unet(input_img, exit_channels=keras_gen.num_cats, n_filters=8, dropout=0.05, batchnorm=True)
model.summary()

# Обучение:
model_check = ModelCheckpoint('../../models/weights.{epoch:02d}-{loss:.5f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
tf_logger = TensorBoardBatchLogger(project_path=PROJECT_PATH, step_size_train=3, batch_size=keras_gen.batch_size)
history = model.fit_generator(gen,
                              steps_per_epoch=keras_gen.total_imgs // keras_gen.batch_size,
                              epochs=3,
                              callbacks=[model_check, tf_logger])
