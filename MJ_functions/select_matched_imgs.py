# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:36:01 2022

@author: atlas
"""

import os
import shutil
import tqdm
from PIL import Image
import numpy as np


def pick_matched_ir_thermal(ir_path, thermal_path, ir_matched_path, thermal_matched_path):
    ir_img_list = os.listdir(ir_path)
    print("{:<15}: {}".format("Num of ir", len(ir_img_list)))

    thermal_img_list = os.listdir(thermal_path)
    print("{:<15}: {}".format("Num of thermal", len(thermal_img_list)))

    num_matched = 0
    for ir in tqdm.tqdm(ir_img_list):
        for thermal in thermal_img_list:
            if ir[12:-4] == thermal[12:-4]:
                ir_src = os.path.join(ir_path, ir)
                ir_tar = os.path.join(ir_matched_path, ir)
                shutil.copy(ir_src, ir_tar)
                
                thermal_src = os.path.join(thermal_path, thermal)
                thermal_tar = os.path.join(thermal_matched_path, thermal)
                shutil.copy(thermal_src, thermal_tar)
                
                # print(ir, "---", thermal)
                num_matched += 1
                break
        continue
    print("{:<15}: {}".format("\nNumber of matched", num_matched))


def crop_ir(ir_path, out_path, size=[640,480]):
    imgs_list = os.listdir(ir_path)
    for img in tqdm.tqdm(imgs_list):
        img_path = os.path.join(ir_path, img)
        # print(img_path)
        current_img = Image.open(img_path)
        width, height = current_img.size
        
        assert width>=size[0], "Width is too small to crop"
        assert height>=size[1], "Height is too small to crop"
        # print("old",width, height)
        
        center = [width/2, height/2]
        # print(center)
        # new_width = width - size[0]
        new_height = height - size[1]
        
        left = (width - size[0]) / 2
        right = width - left
        top = (height - size[1]) / 2
        bottom = height - top
        new_img = current_img.crop((left, top, right, bottom))
        # print("new",new_img.size)
        
        new_img.save(os.path.join(out_path,"crop_"+img))
        

    
if __name__ == "__main__":
    ir_path = r'E:\calibration\Azure\vis_ir'
    thermal_path = r'E:\calibration\thermal\PC005_vis_thermal'
    ir_matched_path = r'E:\calibration\Azure\ir_matched'
    thermal_matched_path = r'E:\calibration\thermal\thermal_matched'
    
    ir_matched_crop_path = r'E:\calibration\Azure\ir_matched_crop'
    
    crop_ir(ir_matched_path, ir_matched_crop_path)