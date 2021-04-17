import copy
import random

import tensorflow as tf
from tensorflow.keras.models import load_model
from coco_image import CocoImage

# A generator that provides a representative dataset
def representative_data_gen():
    annotation_file = '/home/juan/DATASET/HOUGANG/label.json'
    classes = ['Path']
    image_size = 128
    coco_image = CocoImage(annotation_file, classes)
    dataset_list = copy.deepcopy(coco_image.image_ids)
    random.Random(1337).shuffle(dataset_list)
    for i in range(100):
        image = dataset_list[i]
        image = coco_image.get_image_string(image)
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]

model = load_model('semantic_segmentation.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = False

# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open('semantic_segmentation.tflite', 'wb') as f:
    f.write(tflite_model)