import cv2
import os
from tqdm import tqdm

mask_valid_path = 'dataset/masks/validation'

mask_valid_ids = os.listdir(mask_valid_path)

for id in tqdm(mask_valid_ids):
    path = os.path.join(mask_valid_path, id)
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'dataset/masks/validation/{id}', gray_image)
    
mask_train_path = 'dataset/masks/train'

mask_train_ids = os.listdir(mask_train_path)

for id in tqdm(mask_train_ids):
    path = os.path.join(mask_train_path, id)
    image = cv2.imread(path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'dataset/masks/train/{id}', gray_image)