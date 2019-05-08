from keras import Input

from samples.coco.unet import get_unet
from samples.coco.try_generator import stupid_gen

img_size_target = (800, 800, 3)
# Если обучать с нуля:
input_img = Input(img_size_target, name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.summary()

epochs = 50
tr_gen = stupid_gen()

history =  model.fit_generator(tr_gen,
                    steps_per_epoch=7000 // 32,
                    verbose=1)
print('Fitted!')