import copy
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from coco_image import CocoImage

ANNOTATION_FILE = '/home/juan/DATASET/HOUGANG/label.json'
MODEL_FILE = 'semantic_segmentation.h5'
output_file = MODEL_FILE.split(".")[0] + '.tflite'

def representative_data_gen():
    classes = ['Path']
    coco_image = CocoImage(ANNOTATION_FILE, classes)
    dataset_list = copy.deepcopy(coco_image.image_ids)
    random.Random(1337).shuffle(dataset_list)
    for i in range(100):
        image = dataset_list[i]
        image = coco_image.get_image_string(image)
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [image_height, image_width])
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]

model = load_model(MODEL_FILE)
_, image_height, image_width, _ = model.layers[0].get_input_at(0).get_shape()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = False

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open(output_file, 'wb') as f:
    f.write(tflite_model)