from utils import open_json, labelme2mask, TARGET_LIST
import cv2
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

train_ori_path = 'dataset/annotations/train'
valid_ori_path = 'dataset/annotations/validation'
test_ori_path = 'dataset/annotations/testdev'

train_save_path = 'dataset/masks/train'
valid_save_path = 'dataset/masks/validation'

os.makedirs(train_save_path, exist_ok=True)
os.makedirs(valid_save_path, exist_ok=True)
# test_ori_path = 'dataset/images/testdev'

target_dict = dict(zip(TARGET_LIST, range(len(TARGET_LIST))))

for anno_file in tqdm(os.listdir(train_ori_path)):
    anno_name = anno_file.split('.')[0]
    anno = open_json(f'{train_ori_path}/{anno_file}')
    mask = labelme2mask(anno)
    M, N, _ = mask.shape
    new_mask = np.full((M, N, 3), 255, dtype=np.uint8)
    
    for class_idx in range(0, 19):
        class_mask = mask[:, :, class_idx]
        new_mask[class_mask == 1] = class_idx
        
    cv2.imwrite(f'{train_save_path}/{anno_name}.png', new_mask)
    
for anno_file in tqdm(os.listdir(valid_ori_path)):
    anno_name = anno_file.split('.')[0]
    anno = open_json(f'{valid_ori_path}/{anno_file}')
    mask = labelme2mask(anno)
    M, N, _ = mask.shape
    new_mask = np.full((M, N, 3), 255, dtype=np.uint8)
    
    for class_idx in range(0, 19):
        class_mask = mask[:, :, class_idx]
        new_mask[class_mask == 1] = class_idx
        
    cv2.imwrite(f'{valid_save_path}/{anno_name}.png', new_mask)