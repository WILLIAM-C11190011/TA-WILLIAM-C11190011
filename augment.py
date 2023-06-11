from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='reflect')

i = 0
for batch in datagen.flow_from_directory(directory='BaruLabel',
                                         batch_size=11,
                                         target_size=(720, 1280),
                                         color_mode="rgb",
                                         save_to_dir='LabelAugment',
                                         save_prefix='Label',
                                         save_format='jpg'):
    i += 1
    if i > 13:
        break
