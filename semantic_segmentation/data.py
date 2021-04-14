import os
import sys
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import utils
from coco_image import CocoImage

class DataGen(utils.Sequence):
    def __init__(self , coco_obj, image_ids, batch_size = 8 , image_size = 128):
        self.batch_size = batch_size
        self.image_size = image_size
        self.coco = coco_obj
        self.image_ids = image_ids
        self.on_epoch_end()
  
    def __load__(self , image_id):        
        image = self.coco.get_image(image_id)
        image = cv2.resize(image , (self.image_size , self.image_size)) 
        
        mask = self.coco.get_mask(image_id)
        mask = cv2.resize(mask , (self.image_size , self.image_size))
        
        image = image / 255.0
        mask = mask / 255.0
        
        return image , mask
  
    def __getitem__(self , index):
        if (index + 1)*self.batch_size > len(self.image_ids):
            self.batch_size = len(self.image_ids) - index * self.batch_size
            
        file_batch = self.image_ids[index * self.batch_size : (index + 1) * self.batch_size]
        
        images = []
        masks = []
        
        for id_name in file_batch : 
            _img , _mask = self.__load__(id_name)
            images.append(_img)
            masks.append(_mask)
        
        images = np.array(images)
        masks = np.array(masks)
        
        return images , masks
  
    def on_epoch_end(self):
        pass
  
    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))