import glob
import os
import pickle
from tqdm import tqdm
import pandas as pd
import mxnet as mx
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import time
# from transformers import AutoTokenizer, AutoModel, AutoConfig, Wav2Vec2Processor 
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
# import torchaudio
import cv2
import random
from scipy.special import softmax
from scipy.stats import pearsonr
import pandas

def adaptive_avg_pool1d_axis0(arr, output_size):
    """
    Apply 1D adaptive average pooling to a 2D array along the first axis.

    Parameters:
    arr (numpy.ndarray): 2D input array.
    output_size (int): Desired output size along the first axis.

    Returns:
    numpy.ndarray: Pooled 2D array with shape (output_size, arr.shape[1]).
    """
    input_size, num_features = arr.shape
    segment_size = input_size / output_size
    pooled_array = np.zeros((output_size, num_features))

    for i in range(output_size):
        # Determine the start and end indices of the current segment
        start = int(i * segment_size)
        if i == output_size - 1:  # Last segment may need to extend to the end of the array
            end = input_size
        else:
            end = int((i + 1) * segment_size)

        # Calculate the mean for the current segment across all features
        pooled_array[i] = np.mean(arr[start:end], axis=0)

    return pooled_array


# RAF SINGLE
# Surprised, FEAR, DISGUST, HAPPINESS, SADNEES, ANGER, NEURAL.
class DatasetChallenge4_single(Dataset):
    def __init__(self, root, split, transforms=None) -> None:
        self.root = root
        self.img_root =os.path.join(root, 'aligned')
        self.imgs = os.listdir(self.img_root)
        self.imgs = [im for im in self.imgs if split in im][:]
        self.transforms = transforms
        with open(os.path.join(root, 'list_patition_label.txt'), 'r') as f:
            annos = f.readlines()
        self.annos = {e.split(' ')[0]:int(e.split(' ')[1].strip()) - 1 for e in annos}
        # self.imgs = [im for im in self.imgs if self.annos[im.replace('_aligned', '')] != 6]
        print(f'loaded {split} data, total {len(self.imgs)}')
    
    
    def __getitem__(self, index):
        imgname = self.imgs[index]
        img = Image.open(os.path.join(self.img_root, imgname)).convert('RGB')
        label = self.annos[imgname.replace('_aligned', '')]
        
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
    
# RAF SINGLE
# Surprised, FEAR, DISGUST, HAPPINESS, SADNEES, ANGER, NEURAL.
class DatasetChallenge4_Compound(Dataset):
    def __init__(self, root, split, fold_i=0, transforms=None) -> None:
        self.root = root
        self.imgs = [y for x in os.walk(root) for y in glob.glob(os.path.join(x[0], '*.jpg'))]
        if fold_i == 0:
            self.imgs = [im for im in self.imgs if split in im][:]
        else:
            imgs_test = self.imgs[fold_i::5]
            if split == 'test':
                self.imgs = imgs_test
            else:
                self.imgs = [im for im in self.imgs if im not in imgs_test]
                
        CLASSES = ['Angrily Disgusted', 'Angrily Surprised', 'Disgustedly Surprised', 'Fearfully Angry', 'Fearfully Surprised',
               'Happily Disgusted', 'Happily Surprised', 'Sadly Angry', 'Sadly Disgusted', 'Sadly Fearful', 'Sadly Surprised']

        self.transforms = transforms

        self.annos = {e:CLASSES.index(e.split('/')[-2]) for e in self.imgs}
        print(f'loaded {split} data, total {len(self.imgs)}')
    
    
    def __getitem__(self, index):
        imgname = self.imgs[index]
        img = Image.open(imgname).convert('RGB')
        label = self.annos[imgname]
        
        if self.transforms:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


# def load_feature_cache():
#     root = '/project/ABAW6/data/challenge5'
#     feature_names = [
#         # 'audio_hubertbase',
#         # 'audio_wav2vec2',
#         # 'text_chatglm3',
#         # 'text_chatglm2',
#         # 'visual_mae_emb_large',
#         # 'visual_mae_emb',
#         # 'visual_mae_affectnet',
#         # 'visual_vit'
#         # 'text_chatglm3_whisper'
#         # 'text_chatglm3_whipser_large_lora_14th'
#         'visual_mae_au'
#         ]
#     for feature_name in feature_names:
#         print('processing:', feature_name)
#         fea_root = os.path.join(root, feature_name)
#         save_root = fea_root+'.pkl'
#         # if os.path.exists(save_root):
#         #     continue
#         filenames = os.listdir(fea_root)[:]
#         feat = {}
#         for fname in tqdm(filenames):
#             vname = int(fname.split('.')[0])
#             if filenames[0].endswith('.npy'):
#                 try:
#                     fea = np.load(os.path.join(fea_root, fname))
#                 except:
#                     fea = np.zeros(1,1,4096)
#                     print('error:', vname)
#             elif filenames[0].endswith('.pkl'):
#                 with open(os.path.join(fea_root, fname), 'rb') as f:
#                     fea = pickle.load(f)
#             feat[vname] = fea.squeeze()
#         with open(save_root, 'wb') as f:
#             pickle.dump(feat, f)
            
if __name__ == '__main__':
    import torchvision.transforms as transforms
    root = '/project/ABAW6/data/RAF_single'
    
    
    transform1 = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.RandomResizedCrop([224, 224], ratio=[0.8,1.2]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation([-10,10]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,hue=0.15),
        transforms.ToTensor(),

    ])
             
    dataset_ERI2 = DatasetChallenge4_single(root, 'test', transforms=transform1)
    data_loader = DataLoader(
    dataset=dataset_ERI2,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    ) 

    for i, inputs in tqdm(enumerate(data_loader), total=len(data_loader)):
        pass