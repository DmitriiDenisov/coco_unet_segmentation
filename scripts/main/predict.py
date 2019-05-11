from keras.models import load_model
import numpy as np
from scripts.main.generator import KerasGenerator

model = load_model('../../models/weights.02-0.36766.hdf5')

keras_gen = KerasGenerator(annFile='../../coco_dataset/annotations/instances_val2017.json',
                     dataset_dir='coco_dataset',
                     subset='val',
                     year='2017',
                     batch_size=1)
gen = keras_gen.generate_batch()
x_train, y_train = next(gen)

y_pred = model.predict(x_train)

np.save('y_pred.npy', y_pred)
np.save('x_test.npy', x_train)
