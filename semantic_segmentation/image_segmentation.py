from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

class ImageSegmentation:
    def __init__(self, model):
        self.model = load_model(model)
        _, in_h, in_w, _ = self.model.layers[0].get_input_at(0).get_shape()
        self.image_size = (m_h, m_w)

    def predict(self, img):
        img_height, img_width, _ = img.shape
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img , self.image_size)
        input_img = np.array(input_img).astype('float32')
        input_img = input_img / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        prediction = self.model.predict(input_img)

        mask = cv2.resize(prediction[0] * 255, (img_width, img_height))
        mask = (mask > 125).astype(int)
        mask = img * mask[..., None]
        mask = [0,255,0] * mask

        return (mask + img).astype(np.uint8)

