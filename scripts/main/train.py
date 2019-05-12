import os
import sys
from keras import Input, Model
from keras.callbacks import ModelCheckpoint

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_PATH) # чтобы из консольки можно было запускать

from scripts.main.another_unet import CreateModel
from utils_folder.logger import TensorBoardBatchLogger, SaveModelEachBatch
from scripts.main.unet import get_unet, build_model
from scripts.main.metrics import iou_loss_core
from scripts.main.generator import KerasGenerator
from utils_folder import config

config = config.CocoConfig()

# Генератор данных:
keras_gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_train2017.json',
                     dataset_dir='coco_dataset',
                     subset='train',
                     year='2017',
                     batch_size=1)
gen = keras_gen.generate_batch()

img_size_target = (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3)
# Сетка 1:
# input_img = Input(img_size_target, name='img')
# model = get_unet(input_img, exit_channels=keras_gen.num_cats, n_filters=8, dropout=0.05, batchnorm=True)
# model.summary()

# Сетка 2:
model_ = CreateModel(img_size_target=config.IMAGE_MAX_DIM)
model_.compile(loss='binary_crossentropy', optimizer="adam", metrics=[iou_loss_core])
model_.summary()

# Обучение:
model_check = ModelCheckpoint('../../models/weights.{epoch:02d}-{loss:.5f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
tf_logger = TensorBoardBatchLogger(project_path=PROJECT_PATH, batch_size=keras_gen.batch_size)
save_model_batch = SaveModelEachBatch('../../models')
history = model_.fit_generator(gen,
                              steps_per_epoch=keras_gen.total_imgs // keras_gen.batch_size,
                              epochs=3,
                              callbacks=[tf_logger, model_check, save_model_batch])
