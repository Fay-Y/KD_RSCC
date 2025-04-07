import h5py
import json
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader    
import numpy as np
import pickle
 
class CCDataset(Dataset):
    def __init__(self, data_folder, split='train'):
        self.image_file = os.path.join(data_folder,f'{split}_image_attention_pool.pickle')
        # self.image_file = os.path.join(data_folder,f'{split}_image_clip.pickle')
        self.cation_file = os.path.join(data_folder,f'{split}_caption_encoded.json')
        # self.small_cation_file = os.path.join(data_folder,f'{split}_smallcaption_encoded.json')
        self.split = split
        

        with open(self.cation_file, 'r') as f:
            self.caption_data = json.load(f)
        # with open(self.small_cation_file, 'r') as sf:
        #     self.small_caption_data = json.load(sf)
        
        self.image_list = list(self.caption_data.keys())

        with open(self.image_file, 'rb') as file:
            self.image_data = pickle.load(file)

    
    def __len__(self):
        if self.split == 'test':
            return len(self.caption_data)
        else:
            return sum(len(captions) for captions in self.caption_data.values())
    
    def __getitem__(self, idx):
        if self.split == 'test':
            img_idx = idx
        else:
            # img_idx = idx
            img_idx = idx//5
        image_key = self.image_list[img_idx] 
 
        image_info = self.image_data[img_idx]
        # print("image_info",image_info)
        image_before = np.transpose(image_info[image_key]['image_before'], (2, 0, 1))
        image_after = np.transpose(image_info[image_key]['image_after'], (2, 0, 1))
        # image_before = image_info[image_key]['feature_before'].squeeze(0)
        # image_after = image_info[image_key]['feature_after'].squeeze(0)
        if self.split == 'test':
            text_info = self.caption_data[image_key]
            # small_text_info = self.caption_data[image_key]
            # return image_before,image_after,np.array(text_info),np.array(small_text_info),image_key
            return image_before,image_after,np.array(text_info),image_key
        else:
            sentence_count = 0
            for _, captions in self.caption_data.items():
                if idx < sentence_count + len(captions):

                    caption_idx = idx - sentence_count
                    text_info = captions[caption_idx]
                    break
                sentence_count += len(captions)
                
            # small_sentence_count = 0
            # for _, small_captions in self.small_caption_data.items():
            #     if idx < small_sentence_count + len(small_captions):

            #         small_caption_idx = idx - small_sentence_count
            #         small_text_info = small_captions[small_caption_idx]
            #         break
            #     small_sentence_count += len(captions)

            # return image_before,image_after, np.array(text_info),np.array(small_text_info),image_key
            return image_before,image_after, np.array(text_info),image_key

# # # # # 使用示例
# data_folder = 'datasets_large'
# split = 'train'


# batch_size = 1

# dataset = CCDataset(data_folder, split)
# # dataloader = DataLoader(dataset, batch_size, shuffle=False)
# dataloader = DataLoader(dataset, 32, shuffle=True, drop_last=True,
#                                num_workers=8, persistent_workers=True, pin_memory=True)
# i= 0 
# for img1,img2,cap,key in dataloader:
    
#     print(i,key,cap)

#     i+=1