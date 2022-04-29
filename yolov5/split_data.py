import os
import shutil
from sklearn.model_selection import train_test_split

images = [os.path.join('images', x) for x in os.listdir('images') if x != 'train' and x != 'val' and x !='test']
annotations = [os.path.join('labels', x) for x in os.listdir('labels') if x[-3:] == "txt"]

train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

def move_files_to_folder(list_of_files, destination_folder):
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)

    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'labels/train/')
move_files_to_folder(val_annotations, 'labels/val/')
move_files_to_folder(test_annotations, 'labels/test/')