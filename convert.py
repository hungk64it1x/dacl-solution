import os
import sys
import json
import time
import cv2
import argparse
import numpy as np

from tqdm import tqdm
from glob import glob
from PIL import Image
from os.path import join

class DataTransformer:
    def __init__(self, mask, img_path, cls_dict):
        self.data = {
            "version": "1",
            "shapes": [],
            "imageName": img_path.split('/')[-1],
            "imagePath": img_path,
            "imageHeight": 0,
            "imageWidth": 0,
            "dacl10k_version": "v2",
            "split": "testdev"
        }
        self.cls_dict = cls_dict
        self.ori_mask = mask


    def _getShapes(self):
        shapes = []
        poly_point_dict = self._mask2Poly()
        for cls, poly_points in poly_point_dict.items():
            for poly_point in poly_points:
                shapes.append({
                    "label": cls,
                    "points": poly_point,
                    "shape_type": "polygon",
                    
                })
        self.data["shapes"] = shapes

    def _getWH(self):
        h, w = self.ori_mask.shape[:2]
        self.data["imageHeight"] = h
        self.data["imageWidth"] = w

    def _mask2Poly(self):
        """
        :param mask_path: mask_path mask format like [[0, 0, 0, 1, 1],
                                                        [0, 0, 0, 1, 1],
                                                        [2, 2, 0, 0, 0],
                                                        [2, 2, 0, 0, 0]]
        :return: polygon points dict {"cls1": [[x1, y1,], [x2, y2], ...](point1),
                                            [[x1, y1,], [x2, y2], ...](point2), ...
                                     "cls2": ...,
                                    ...}
        """

        ## init poly point dict
        poly_point_dict = {}
        for k, v in self.cls_dict.items():
            
            if k == "Background":
                continue

            poly_point_dict[k] = []
            self.mask = self.ori_mask[:,:, v] # 0 - 1 
            if len(np.unique(self.mask)) == 1:
                continue
            cls_mask = (self.mask == 1).astype(np.uint8)
            
            poly_points, _ = cv2.findContours(cls_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # approx_points = []
            # for contour in poly_points:
            #     epsilon = 0.5 * cv2.arcLength(contour, True)
            #     approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            #     approx_points.append(approx_contour)
            for points in poly_points:
                if len(points) <= 2:
                    continue
                poly_point_dict[k].append(points.squeeze().tolist())
        return poly_point_dict

    def mask2Json(self, output_path):
        self._getWH()
        self._getShapes()
        with open(output_path, "w") as f:
            context = json.dumps(self.data, indent=4)
            f.write(context)
        return self.data
    
if __name__ == '__main__':
    img_path = '/home/solashi/hungpv/compet/dacl/dataset/images/all/dacl10k_v2_train_0000.jpg'
    mask = np.load('/home/solashi/hungpv/compet/dacl/dataset/masks/all_multi/dacl10k_v2_train_0000.npz')['image_array']
    new_mask = np.full((mask.shape[0], mask.shape[1], 3), 255, dtype=np.uint8)
    
    for class_idx in range(0, 19):
        class_mask = mask[:, :, class_idx]
        
        new_mask[class_mask == 1] = class_idx
    category_ids = {
        'Crack': 0,
        'ACrack': 1,
        'Wetspot': 2,
        'Efflorescence': 3,
        'Rust': 4,
        'Rockpocket': 5,
        'Hollowareas': 6,
        'Cavity': 7,
        'Spalling': 8,
        'Graffiti': 9,
        'Weathering': 10,
        'Restformwork': 11,
        'ExposedRebars': 12,
        'Bearing': 13,
        'EJoint': 14,
        'Drainage': 15,
        'PEquipment': 16,
        'JTape': 17,
        'WConccor': 18,
    }
    data_transformer = DataTransformer(mask, img_path, category_ids)
    output_path = '/home/solashi/hungpv/compet/dacl/segmentation_pytorch/saved_json/test.json'
    data = data_transformer.mask2Json(output_path)