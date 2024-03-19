import pprint
import numpy as np
import pandas as pd
import torch
import os
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import torchvision.transforms.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import PIL
import argparse
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



class ABAW5Seqdata(data.dataset.Dataset):
    def __init__(self, pkl_file, img_path,  audio_path=None, transform=None, config = None, is_s2 = False, is_pred = False):
        self.img_path = img_path
        with open(pkl_file,"rb") as f:
            self.data = pickle.load(f)
        self.config = config
        self.maxsize = 0
        self.names = []
        self.labels = []
        self.audios = []
        self.transform = transform
        self.seqs = []

        if config["use_audio_fea"]:
            with open(audio_path,"rb") as f:
                audio_inps = pickle.load(f)

        for k,v in tqdm(self.data.items()):
            self.names.append(k)
            self.labels.append(v["labels"])
            self.seqs.append(v["frames"])
        
        self.is_s2 = is_s2
        self.is_pred = is_pred

        if self.is_s2 == True:
            self.names = []
            self.labels = []
            self.seqs = []
            hards = config["hard_sample_pkl"]
            with open(hards,"rb") as f:
                tmp = pickle.load(f)
            for k,v in tqdm(tmp.items()):
                self.names.append(k)
                self.labels.append(v["labels"])
                self.seqs.append(v["frames"])

            fulls = list(self.data.keys())
            full_length = len(self.data)
            cha =  full_length - len(tmp)

            samples = np.random.choice(fulls,cha,replace=False)
            for s in samples:
                self.names.append(s)
                v = self.data[s]

                self.labels.append(v["labels"])
                self.seqs.append(v["frames"])
            print(len(self.names))


        if config["use_audio_fea"]:
            self.use_audio = True
            with open(audio_path,"rb") as f:
                self.audio_fea_dict = pickle.load(f)

    
    def __len__(self):
        # return 50
        return len(self.data)
    
    def get_audio_seq_feas(self,start,end, frames):
        vi = start.split("/")[0]
        if "_left" in vi:
            vi = vi.split("_left")[0]
        if "_right" in vi:
            vi = vi.split("_right")[0]
        st = int(start.split("/")[-1].split(".")[0])
        ed = int(end.split("/")[-1].split(".")[0]) 

        audio_dict = self.audio_fea_dict[vi]
        embs = []
        for f in frames:
            key = f.split("/")[-1].split(".")[0]
            if key in audio_dict.keys():
                # print("++++")
                embs.append(audio_dict[key])
            else:
                embs.append(np.zeros(self.config["audio_fea_dim"]))
            
        if len(embs)<self.config["n_segment"]:
            cha = self.config["n_segment"]- len(embs)
            for j in range(cha):
                embs.append(np.zeros(self.config["audio_fea_dim"]))
        embs = np.array(embs)
        embs = torch.from_numpy(embs)
        # embs = torch.tensor(embs)
        return embs
    

    def get_cls_label(self,label):
        for i in range(7):
            if label[i] == 1:
                return i
    
    def check_seq_length(self,frames):
        self.n_segment = self.config["n_segment"]
        if len(frames)<self.n_segment:
            num = self.n_segment - len(frames)
            for i in range(num):
                frames.append(torch.zeros(3,224,224).unsqueeze(0))
        return frames

    def __getitem__(self, index): 
        name = self.names[index]
        
        imgs = []
        frames = self.data[name]["frames"]

        for im in frames:
            img = Image.open(os.path.join(self.img_path,im)).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img.unsqueeze(0))

        imgs = torch.cat(imgs,dim=0)
        sample = {
            "name":name,
            "frames":imgs,
            "frame_names": " ".join(frames)
        }

        if self.is_pred == False:
            if self.config["task"] == "VA":
                label = torch.tensor(self.labels[index])
            elif self.config["task"] == "AU":
                new_label = []
                for tmp in self.labels[index]:
                    tt = tmp.split(",")
                    tt = [int(j) for j in tt]
                    new_label.append(tt)
                label = torch.tensor(new_label).float()
    
            elif self.config["task"] == "EXP":
                tmp = [int(c) for c in self.labels[index]]
                label = torch.tensor(tmp)
            sample["labels"] = label

        if self.config["use_audio_fea"]:
            start = frames[0]
            end = frames[-1]
            sample["audio_inp"] = self.get_audio_seq_feas(start,end,frames).float()
        return sample
        
def build_seq_dataset(config,mode):
    train_transform = build_transform(True)
    val_transform = build_transform(False)
    transform1 = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()])
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor()])  
    
    if mode == "train":
        dataset = ABAW5Seqdata(config["train_pkl"],config["train_img_path"],
        config["train_audio_path"],train_transform,config)
    elif mode =="test":
        dataset = ABAW5Seqdata(config["test_pkl"],config["test_img_path"],
        config["test_audio_path"], val_transform ,config)
    elif mode == "test_hard":
        dataset = ABAW5Seqdata(config["train_pkl"],config["train_img_path"],
        config["train_audio_path"],val_transform,config)
    elif mode == "s2_train":
        dataset = ABAW5Seqdata(
            config["train_pkl"],
            config["train_img_path"],
            config["train_audio_path"],val_transform,config,
            is_s2 = True
        )
    elif mode == "pred":
        dataset = ABAW5Seqdata(
            config["pred_pkl"],
            config["pred_img_path"],
            config["pred_audio_path"],val_transform,config,
            is_s2 = False,
            is_pred = True
        )
    return dataset





