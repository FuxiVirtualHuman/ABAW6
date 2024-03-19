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


class DatasetChallenge5(Dataset):
    def __init__(self, root, split, fold_i, feature_v=[], feature_a=[], feature_t=[], img_max=100, audio_max=320, text_max=32, transforms=None, shuffle_ratio=0) -> None:
        self.root = root
        self.img_max, self.audio_max, self.text_max = img_max, audio_max, text_max
        self.shuffle_ratio = shuffle_ratio
        self.feature_v, self.feature_a, self.feature_t = feature_v, feature_a, feature_t
        self.data = {}

        if fold_i:
            for _split in ['train', 'valid']:
                csvdata = pandas.read_csv(os.path.join(root, f'{_split}_split.csv'))
                for line in csvdata.values:
                    self.data[int(line[0])] = {'label':line[1:]}
            data_list_valid = list(self.data.keys())[fold_i ::5]
            if split == 'valid':
                self.data_list = data_list_valid
            elif split == 'train':
                self.data_list = [e for e in self.data.keys() if e not in data_list_valid]
        else:
            if split in ['train', 'valid']:
                csvdata = pandas.read_csv(os.path.join(root, f'{split}_split.csv'))
                for line in csvdata.values:
                    self.data[int(line[0])] = {'label':line[1:]}
                self.data_list = list(self.data.keys())[:]
            elif split == 'test':
                self.data_list = [int(e.split('.')[0]) for e in os.listdir(os.path.join(root, 'raw'))]
                self.data_list.sort()
                self.data = {e:{'label':np.zeros(6)} for e in self.data_list}
        self.feature_dims = {}
        self.load_feature_v(feature_v)
        self.load_feature_a(feature_a)
        self.load_feature_t(feature_t)
        
        print('features dims:', self.feature_dims)
        print(f'loaded {split} data, total {len(self.data_list)}')
    
    def load_feature_v(self, feature_names):
        print(f'loading visual feature: {feature_names}')
        for feature_name in feature_names:
            fea_pkl = os.path.join(self.root, f'visual_{feature_name}.pkl')
            with open(fea_pkl, 'rb') as f:
                features = pickle.load(f)
            self.feature_dims[feature_name] = list(features.values())[-1].shape[-1]
            for vname, fea in tqdm(features.items()):
                if vname not in self.data:
                    continue
                self.data[vname][f'{feature_name}'] = fea.reshape(-1, self.feature_dims[feature_name])
        return
    
    def load_feature_a(self, feature_names):
        """_summary_

        Args:
            feature_name (_type_): ['wav2vec2',  'hubertbase']
        """
        print(f'loading audio feature: {feature_names}')
        
        for feature_name in feature_names:
            fea_pkl = os.path.join(self.root, f'audio_{feature_name}.pkl')
            with open(fea_pkl, 'rb') as f:
                features = pickle.load(f)
            self.feature_dims[feature_name] = list(features.values())[-1].shape[-1]
            for vname, fea in tqdm(features.items()):
                if vname not in self.data:
                    continue
                self.data[vname][f'{feature_name}'] = fea.reshape(-1, self.feature_dims[feature_name])
            self.feature_dims[feature_name] = fea.shape[-1]
        return
    
        
    def load_feature_t(self, feature_names):
        """_summary_

        Args:
            feature_name (_type_): ['chatglm2',  'chatglm3']
        """
        print(f'loading text feature: {feature_names}')
        for feature_name in feature_names:
            fea_pkl = os.path.join(self.root, f'text_{feature_name}.pkl')
            with open(fea_pkl, 'rb') as f:
                features = pickle.load(f)
            self.feature_dims[feature_name] = list(features.values())[-1].shape[-1]
            for vname, fea in tqdm(features.items()):
                if vname not in self.data:
                    continue
                self.data[vname][f'{feature_name}'] = fea.reshape(-1, self.feature_dims[feature_name])
            self.feature_dims[feature_name] = fea.shape[-1]
        return
    
    def __getitem__(self, index):
        vname = self.data_list[index]
        data = self.data[vname]
        data['vid'] = vname
        for feaname_v in self.feature_v:
            fea = data[feaname_v]
            n = len(fea)
            step = n / float(self.img_max)
            indexes = np.array([int(i*step+random.uniform(0, step)) for i in range(self.img_max)])
            if np.random.randn() < self.shuffle_ratio:
                np.random.shuffle(indexes)
            fea = fea[indexes]
            data[feaname_v] = fea
        
        for feaname_t in self.feature_t:
            fea = data[feaname_t]
            if fea.shape[0] > self.text_max:
                fea = adaptive_avg_pool1d_axis0(fea, self.text_max)
            data[feaname_t] = fea
        
        for feaname_a in self.feature_a:
            fea = data[feaname_a]
            if fea.shape[0] > self.audio_max:
                fea = adaptive_avg_pool1d_axis0(np.array(fea), self.audio_max)
            data[feaname_a] = fea
            
        return data
     
    def __len__(self):
        return len(self.data_list)


class collate_fn():
    def __init__(self, feaname_a, feaname_t, feaname_v) -> None:
        self.feaname_a, self.feaname_t, self.feaname_v = feaname_a, feaname_t, feaname_v

    def __call__(self, batch):
        
        if self.feaname_a:
            sort_key = self.feaname_a[0]
        elif self.feaname_v:
            sort_key = self.feaname_v[0]
        elif self.feaname_t:
            sort_key = self.feaname_t[0]

        batch = sorted(batch, key=lambda x: x[sort_key].squeeze().shape[0], reverse=True)
        labels = torch.stack([torch.FloatTensor(sample['label']) for sample in batch])
        batch_dict = {'labels': labels, 'lengths':{}, 'vid':[sample['vid'] for sample in batch]}

        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        for name in self.feaname_v:
            batch_dict[name] = pad_sequence([torch.FloatTensor(sample[name]).squeeze() for sample in batch], batch_first=True)            
            batch_dict['lengths'][name] = torch.LongTensor([sample[name].squeeze().shape[0] for sample in batch])

        for name in self.feaname_a:
            audio = pad_sequence([torch.FloatTensor(sample[name]).squeeze() for sample in batch], batch_first=True)
            batch_dict[name] = audio
            batch_dict['lengths'][name] = torch.LongTensor([np.ceil(sample[name].squeeze().shape[0]) for sample in batch])

        for name in self.feaname_t:
            text = pad_sequence([torch.FloatTensor(sample[name]).squeeze() for sample in batch], batch_first=True)
            batch_dict[name] = text
            batch_dict['lengths'][name] = torch.LongTensor([sample[name].squeeze().shape[0] for sample in batch])

        return batch_dict
    
    
def collate_fn2(batch, feaname_a, feaname_t, feaname_v):
    
    if feaname_a:
        sort_key = feaname_a[0]
    elif feaname_v:
        sort_key = feaname_v[0]
    elif feaname_t:
        sort_key = feaname_t[0]

    batch = sorted(batch, key=lambda x: x[sort_key].squeeze().shape[0], reverse=True)
    labels = torch.stack([torch.FloatTensor(sample['label']) for sample in batch])
    batch_dict = {'labels': labels, 'lengths':{}}

    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    for name in feaname_v:
        batch_dict[name] = pad_sequence([torch.FloatTensor(sample[name]).squeeze() for sample in batch], batch_first=True)            
        batch_dict['lengths'][name] = torch.LongTensor([sample[name].squeeze().shape[0] for sample in batch])

    for name in feaname_a:
        audio = pad_sequence([torch.FloatTensor(sample[name]).squeeze() for sample in batch], batch_first=True)
        batch_dict[name] = audio
        batch_dict['lengths'][name] = torch.LongTensor([sample[name].squeeze().shape[0] for sample in batch])

    for name in feaname_t:
        text = pad_sequence([torch.FloatTensor(sample[name]).squeeze() for sample in batch], batch_first=True)
        batch_dict[name] = text
        batch_dict['lengths'][name] = torch.LongTensor([sample[name].squeeze().shape[0] for sample in batch])

    return batch_dict
    
def load_feature_cache():
    root = '/project/ABAW6/data/challenge5/test'
    feature_names = [
        # 'audio_hubertbase',
        # 'audio_wav2vec2',
        # 'text_chatglm3',
        # 'text_chatglm2',
        # 'visual_mae_emb_large',
        # 'visual_mae_emb',
        # 'visual_mae_affectnet',
        # 'visual_vit',
        # 'text_chatglm3_whisper'
        # 'text_chatglm3_whipser_large_lora_21st',
        'text_chatglm3_whipser_large',
        # 'text_chatglm3_whipser_large_lora',
        # 'visual_mae_au',
        # 'visual_mae_expr_abaw6'
        # 'visual_mae_au_abaw6_new'
        # 'visual_mae_au_new',
        ]
    for feature_name in feature_names:
        print('processing:', feature_name)
        fea_root = os.path.join(root, feature_name)
        save_root = fea_root+'.pkl'
        # if os.path.exists(save_root):
        #     continue
        filenames = os.listdir(fea_root)[:]
        feat = {}
        for fname in tqdm(filenames):
            vname = int(fname.split('.')[0])
            if filenames[0].endswith('.npy'):
                try:
                    fea = np.load(os.path.join(fea_root, fname))
                except:
                    fea = np.zeros(1,1,4096)
                    print('error:', vname)
            elif filenames[0].endswith('.pkl'):
                with open(os.path.join(fea_root, fname), 'rb') as f:
                    fea = pickle.load(f)
            feat[vname] = fea.squeeze()
        with open(save_root, 'wb') as f:
            pickle.dump(feat, f)
            
if __name__ == '__main__':
    import torchvision.transforms as transforms
    root = '/project/ABAW6/data/challenge5'
    load_feature_cache()
    exit()
    
    
    transform1 = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.RandomResizedCrop([224, 224], ratio=[0.8,1.2]),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomRotation([-10,10]),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,hue=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
             
    # dataset_ERI = Dataset_ERI(root, 'val', img_max=4, transforms=transform1)
    
    # dataset_ERI1 = Dataset_ERI_MAE(root, 'train', img_max=16, transforms=transform1, use_visual=True, use_audio=True, use_text=True)
    collate = collate_fn(feaname_v=['mae_emb_large'], feaname_a=['hubertbase'], feaname_t=['chatglm3'])
    dataset_ERI2 = DatasetChallenge5(root, 'test', 0, feature_v=['mae_emb_large'], feature_a=['hubertbase'], feature_t=['chatglm3'])
    # pcc = (dataset_ERI1.pcc * len(dataset_ERI1.vid_list) + dataset_ERI2.pcc*len(dataset_ERI2.vid_list)) / (len(dataset_ERI2.vid_list)+len(dataset_ERI1.vid_list))
    data_loader = DataLoader(
    dataset=dataset_ERI2,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    collate_fn=collate, 
    ) 

    for i, inputs in tqdm(enumerate(data_loader), total=len(data_loader)):
        pass