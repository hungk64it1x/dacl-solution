import numpy as np
from PIL import Image
import sys
import cv2
import os
from tqdm import tqdm

from mmsegmentation.mmseg.apis.inference import init_segmentor, inference_segmentor

def check_output(prediction):
    assert np.all(np.isin(prediction, 
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])), \
            "Only valid label values are allowed"

class MMSegModel:
    def __init__(self):
        config_path = 'mmsegmentation/configs/convnext/upernet_convnext_base_fp16_640x640_120k_dacl.py'
        weights_path = 'mmsegmentation/work_dirs/upernet_convnext_base_fp16_640x640_120k_all/iter_120000.pth'
     
        self.model = init_segmentor(config_path, weights_path)
        self.class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    def segment_single_image(self, image_to_segment):

        segmentation_results = inference_segmentor(self.model, [image_to_segment])[0]
        return segmentation_results
    
    

if __name__ == '__main__':
    # testdev_path = 'dataset/dacl10k_v2_testchallenge/images/testchallenge'
    # save_path = 'output/test/mask_output'
    testdev_path = 'dataset/images/testdev'
    save_path = 'output/dev/mask_output'
    os.makedirs(save_path, exist_ok=True)
    model = MMSegModel()
    for test_id in tqdm(os.listdir(testdev_path)):
        mask_id = test_id.split('.')[0] + ".png"
        image_path = os.path.join(testdev_path, test_id)
        image = np.array(Image.open(image_path))
        
        prediction = model.segment_single_image(image)
        # print(np.unique(prediction))
        cv2.imwrite(f'{save_path}/{mask_id}', prediction)
