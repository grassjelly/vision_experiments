import argparse
import time
import collections
import operator
import numpy as np
import time
from PIL import Image

import tflite_runtime.interpreter as tflite
import platform
import matplotlib.pyplot as plt

IMAGE_SAMPLE = '/home/juan/DATASET/HOUGANG/HOUGANG_538686_20.jpg'
MODEL_FILE = 'semantic_segmentation_edgetpu.tflite'

def predict(interpreter, image):
	tensor_index = interpreter.get_input_details()[0]['index']
	interpreter.tensor(tensor_index)()[0]  =  image

	interpreter.invoke()

	output_details = interpreter.get_output_details()[0]
	output_data = np.squeeze(interpreter.tensor(output_details['index'])())

	if  np.issubdtype(output_details['dtype'], np.integer):
		scale, zero_point = output_details['quantization']
		output_data =  scale * (output_data - zero_point)
	
	output_data *= 255

	return (output_data > 125).astype(int)

MODEL_FILE, *device = MODEL_FILE.split('@')
interpreter = tflite.Interpreter(
		model_path=MODEL_FILE,
		experimental_delegates=[
			tflite.load_delegate('libedgetpu.so.1',
								{'device': device[0]} if device else {})
		])

interpreter.allocate_tensors()

_, in_h, in_w, _ = interpreter.get_input_details()[0]['shape']
size = (in_h, in_w)
image = Image.open(IMAGE_SAMPLE).convert('RGB').resize(size, Image.ANTIALIAS)

for i in range(5):
	start = time.perf_counter()
	mask = predict(interpreter, image)
	inference_time = time.perf_counter() - start
	print('%.1fms' % (inference_time * 1000))

	plt.imshow(mask)
	plt.show()
