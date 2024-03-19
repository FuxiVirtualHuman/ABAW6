from models.BasicLearningBranch import VA_fusion,EXP_fusion,AU_fusion
from torch.distributions.beta import Beta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DBCNet(nn.Module):
    def __init__(self, config):
        super(DBCNet, self).__init__()
        self.config = config
        self.task = self.config["task"]

        if self.task == "AU":
            self.BLB = AU_fusion(config)
            self.CLB = AU_fusion(config)
        elif self.task == "EXP":
            self.BLB = EXP_fusion(config)
            self.CLB = EXP_fusion(config)
        elif self.task == "VA":
            self.BLB = VA_fusion(config)
            self.CLB = VA_fusion(config)

        ckpt_path = config['BLB_resume']
        self.BLB.load_state_dict(checkpoint, strict=False)
        
        for p in self.BLB.parameters():
            p.requires_grad = False

        self.beta_mode = config["beta_mode"]

    
    def forward(self,sample1,sample2,mode="train"):
        if self.task != "VA":
            with torch.no_grad():
                BLB_logits = self.BLB(sample1)
            CLB_logits = self.CLB(sample2)
        else:
            with torch.no_grad():
                BLB_v,BLB_a = self.BLB(sample1)
            CLB_v,CLB_a = self.CLB(sample2)

        if mode == "train":
            if self.beta_mode == "fixed":
                betas = self.config["betas"]
                a = np.random.choice(betas,1)[0]
            else:
                m = torch.distributions.beta.Beta(torch.tensor([5.0]), torch.tensor([2.0]))
                a = m.sample().cuda()
        else:
            a = torch.tensor(0.8).cuda()
        if self.task != "VA":
            out = a * BLB_logits + (1-a) * CLB_logits  
            return out,a
        else:
            out_v = a * BLB_v + (1-a) * CLB_v
            out_a = a * BLB_a + (1-a) * CLB_a
            return out_v, out_a, a