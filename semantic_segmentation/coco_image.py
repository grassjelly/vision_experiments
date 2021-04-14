import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pycocotools import coco, cocoeval, _mask
from pycocotools import mask as maskUtils

class CocoImage:
    def __init__(self, annotation_path, class_names):
        self._images_folder = os.path.dirname(annotation_path) + '/'
        self.coco = coco.COCO(annotation_path)
        self.cat_ids = self.coco.getCatIds(catNms=class_names)
        self.image_ids = self.coco.getImgIds(catIds=self.cat_ids)
        self.image_dicts = self.coco.loadImgs(self.image_ids)
        self.total_samples = len(self.image_ids)

    def get_mask(self, image_id):
        sample_img_id = self.coco.getImgIds(imgIds = [image_id])
        sample_img_dict = self.coco.loadImgs(sample_img_id[np.random.randint(0,len(sample_img_id))])[0]
        annotation_ids = self.coco.getAnnIds(imgIds=sample_img_dict['id'], catIds=self.cat_ids, iscrowd=0)
        annotations = self.coco.loadAnns(annotation_ids)

        mask = self.coco.annToMask(annotations[0])
        for i in range(len(annotations)):
            mask = mask | self.coco.annToMask(annotations[i])

        return mask * 255

    def get_image(self, image_id):
        image_file = self.get_image_string(image_id)
        image = mpimg.imread(image_file)
        return image

    def get_image_string(self, image_id):
        sample_img_id = self.coco.getImgIds(imgIds = [image_id])
        sample_img_dict = self.coco.loadImgs(sample_img_id[np.random.randint(0,len(sample_img_id))])[0]
        return self._images_folder + sample_img_dict['file_name']
         