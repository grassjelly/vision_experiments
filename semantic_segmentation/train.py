import random
import copy
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import model
from coco_image import CocoImage
from data import DataGen

image_size = 128 
epochs = 10
batch_size = 8
annotation_file = '/home/juan/DATASET/HOUGANG/label.json'
classes = ['Path']
coco_image = CocoImage(annotation_file, classes)

image_ids = copy.deepcopy(coco_image.image_ids)
dataset_size = coco_image.total_samples
val_size = int(dataset_size * 0.2)
train_size = dataset_size - val_size

train_image_ids = image_ids[:-val_size]
val_image_ids = image_ids[-val_size:]

model = model.UNet(input_shape = (128,128,3))
model.summary()
model.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

callbacks = [
    callbacks.ModelCheckpoint("semantic_segmentation.h5", save_best_only=True)
]

train_gen = DataGen(coco_image, train_image_ids, batch_size = batch_size , image_size = image_size)
val_gen = DataGen(coco_image, train_image_ids, batch_size = batch_size , image_size = image_size)

train_steps =  train_size // batch_size

model.fit(train_gen, validation_data = val_gen, steps_per_epoch = train_steps, epochs=epochs, callbacks=callbacks)