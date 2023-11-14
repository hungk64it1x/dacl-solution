import shutil
import os
import cv2

path = '/home/solashi/hungpv/compet/dacl/dataset/masks/train_multi'
image_path = '/home/solashi/hungpv/compet/dacl/dataset/images/train'
images = os.listdir(path)

for image_id in images:
    id = image_id.split('.')[0]
    shutil.copy(f'{image_path}/{id}.jpg', '/home/solashi/hungpv/compet/dacl/dataset/images/train_multi')