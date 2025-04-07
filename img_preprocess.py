import os
import h5py
from PIL import Image
import torch
import numpy as np
import pickle
from tqdm import tqdm
from torch import nn as nn
from transformers import CLIPProcessor, CLIPModel,CLIPImageProcessor,CLIPVisionModel

import re

class image_CLIP:
    def __init__(self, default_data_path,default_data_processed_path,split="train"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.A_path = os.path.join(default_data_path, split,"A")
        self.B_path = os.path.join(default_data_path, split,"B")
        self.out_path =  os.path.join(default_data_processed_path,split+'_image.pickle')
        version="openai/clip-vit-large-patch14"
        self.image_processor = CLIPImageProcessor.from_pretrained(version)
        self.image_clip =  CLIPVisionModel.from_pretrained(version).cuda()
        self.split = split

    def natural_sort_key(self, s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

    def process_images(self, folder_a_path, folder_b_path, output_hdf5):
        images_a = [f for f in os.listdir(folder_a_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images_a = sorted(images_a, key=self.natural_sort_key)
        # images_b = [f for f in os.listdir(folder_b_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        data = []
        
        for image_name in tqdm(images_a):
            image_path_a = os.path.join(folder_a_path, image_name)
            image_path_b = os.path.join(folder_b_path, image_name)

            # clip start
            image_a = Image.open(image_path_a)
            image_b = Image.open(image_path_b)

            data_all = {
                image_name:{
                    'image_before':np.asarray(image_a).astype(np.float32),
                    'image_after':np.asarray(image_b).astype(np.float32),
                    }
                }
            data.append(data_all)
            #print(data_all)
        with open(self.out_path, 'wb') as file:
            pickle.dump(data, file)
        file.close()
        
            

    def process_folders(self):

        self.process_images(self.A_path, self.B_path, self.out_path)
 
        print(f"the result will be saved ar{self.out_path}")

# 使用示例
default_data_path = 'data_large/images'
default_data_processed_path = 'datasets_large'
train_processor = image_CLIP(default_data_path,default_data_processed_path,split="train")
train_processor.process_folders()

test_processor = image_CLIP(default_data_path,default_data_processed_path,split="test")
test_processor.process_folders()

test_processor = image_CLIP(default_data_path,default_data_processed_path,split="val")
test_processor.process_folders()
