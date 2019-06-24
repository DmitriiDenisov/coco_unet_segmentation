from keras.models import load_model
import numpy as np
from scripts.main.generator import KerasGenerator
from scripts.main.metrics import iou_loss_core

model = load_model('../../models/weights_time_5.0loss_0.022.h5', custom_objects={'iou_loss_core': iou_loss_core})

keras_gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_val2017.json',
                     dataset_dir='coco_dataset',
                     subset='val',
                     year='2017',
                     batch_size=1)
gen = keras_gen.generate_batch()
x_train, y_train = next(gen)

y_pred = model.predict(x_train)
image_id = np.array(keras_gen.batch_train_indecies[0])

np.save('y_pred.npy', y_pred)
np.save('x_test.npy', x_train)
np.save('image_id.npy', image_id)
