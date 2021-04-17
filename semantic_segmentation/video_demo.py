import sys, os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import time
from image_segmentation import ImageSegmentation

try:
    video_path = sys.argv[1]
except:
    print("Invalid video path")
    sys.exit(0)

if os.path.isfile(video_path):
    pass
else:
    print ("Video file does not exist")
    sys.exit(0)

vid = cv2.VideoCapture(video_path)
img_seg = ImageSegmentation('semantic_segmentation.h5')
out = None
prev_time = time.time()
while(vid.isOpened()):
    ret, img = vid.read()
    if not ret:
        continue
    
    now = time.time()
    if now - prev_time > 0.08:
        start = time.perf_counter()
        out = img_seg.predict(img)
        inference_time = time.perf_counter() - start
        print('%.1fms' % (inference_time * 1000))
        fps = vid.get(cv2.CAP_PROP_FPS)
        cv2.imshow('frame', out)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()
