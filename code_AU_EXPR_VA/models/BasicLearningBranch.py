import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.models_vit as models_vit
import math
import torch
from torch import optim
from torch.nn.modules.distance import PairwiseDistance





class temporal_lstm(nn.Module):
    def __init__(self, n_segment, input_size, output_size, hidden_size, num_layers, last="avg"):
        super(temporal_lstm, self).__init__()
        self.n_segment = n_segment
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers,  bidirectional=True ,dropout = 0.3)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.activ = nn.LeakyReLU(0.1) 
        self.last = last

    def forward(self, x):
        n_batch, t, c = x.shape
        new_x = x.view(n_batch, t, c).permute(1,0,2).contiguous()
        new_x, _ = self.gru(new_x)
        new_x = self.activ(new_x)
        new_x = new_x.view(n_batch*t , 2*self.hidden_size)
        new_x = self.linear(new_x)
        new_x = self.activ(new_x)
        new_x = new_x.view(n_batch, t, -1)
        # if self.last == "avg":
        #     new_x = torch.mean(new_x,dim=1)
        return new_x


class TransEncoder(nn.Module):
    def __init__(self, inc=512, outc=512, dropout=0.6, nheads=1, nlayer=4):
        super(TransEncoder, self).__init__()
        self.nhead = nheads
        self.d_model = outc
        self.dim_feedforward = outc
        self.dropout = dropout
        self.conv1 = nn.Conv1d(inc, self.d_model, kernel_size=1, stride=1, padding=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x):
        out = self.conv1(x)
        out = out.permute(2, 0, 1)
        out = self.transformer_encoder(out)
        return out


class feature_extractor(nn.Module):
    def __init__(self, config):
        super(feature_extractor, self).__init__()
        
        model_name = 'vit_base_patch16'
        if config["task"] == "VA":
            num_classes = 2
        elif config["task"] == "EXP":
            num_classes = 8
        elif config["task"] == "AU":
            num_classes = 12
        ckpt_path = config["stage1_checkpoint"]
        
        self.student = getattr(models_vit, model_name)(
                    global_pool=True,
                    num_classes=num_classes,
                    drop_path_rate=0.1,
                    img_size=224,
                )
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model = checkpoint['model']
        self.student.load_state_dict(checkpoint_model, strict=False)
        
        for p in self.student.parameters():
            p.requires_grad = False

    def forward(self, inp):
        bz,seq,c,H,W = inp.shape
        inp = inp.view(bz*seq,c,H,W)
        with torch.no_grad():
            logits, main = self.student(inp, ret_feature=True)
        
        fea = main.view(bz,seq,-1)
        return fea


class VA_fusion(nn.Module):
    def __init__(self, config):
        super(VA_fusion, self).__init__()
        self.vis_extractor = feature_extractor(config)
        if config["use_audio_fea"] == True:
            self.audio_temporal = temporal_lstm(config["n_segment"], input_size=768, output_size=512, hidden_size=512, num_layers=4, last="identity")
        if config["use_word_fea"] == True:
            self.word_temporal = temporal_lstm(config["n_segment"], input_size=768, output_size=512, hidden_size=512, num_layers=4, last="identity")
        concat_dim = 768
        if config["use_audio_fea"] == True:
            concat_dim += config["audio_fea_dim"]
        
        hidden_size1,hidden_size2,hidden_size3 = config["hidden_size"]
        
        self.feat_fc = nn.Conv1d(concat_dim, hidden_size1, 1, padding=0)
        self.activ = nn.LeakyReLU(0.1) 
        self.dropout = nn.Dropout(p=0.3)
        self.vhead = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size3),
                nn.BatchNorm1d(hidden_size3),
                nn.Linear(hidden_size3, 1),
                )
        self.ahead = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size3),
                nn.BatchNorm1d(hidden_size3),
                nn.Linear(hidden_size3, 1),
                )
        
        self.transformer = TransEncoder(inc=hidden_size1, outc=hidden_size2, dropout=0.3, nheads=4, nlayer=4)
        self.config = config
    
    def forward(self,sample):
        vis_inp = sample["frames"]
        bs,seq_len,c,H,W = vis_inp.shape
        vis_fea = self.vis_extractor(vis_inp)
        concats = [vis_fea]
        if self.config["use_audio_fea"]:
            audio_fea = sample["audios"]
            concats.append(audio_fea)
        
        concat_fea = torch.cat(concats,dim=2)
        feat = torch.transpose(concat_fea,1,2)
        feat = self.feat_fc(feat)
        feat = self.activ(feat)
        out = self.transformer(feat)

        out = torch.transpose(out, 1, 0)
        out = torch.reshape(out, (bs*seq_len, -1))

        vout = self.vhead(out)
        aout = self.ahead(out)
        vout = vout.view(bs,seq_len,-1)
        aout = aout.view(bs,seq_len,-1)

        return vout, aout




class EXP_fusion(nn.Module):
    def __init__(self, config):
        super(EXP_fusion, self).__init__()
        self.vis_extractor = feature_extractor(config)
        if config["use_audio_fea"] == True:
            self.audio_temporal = temporal_lstm(config["n_segment"], input_size=768, output_size=512, hidden_size=512, num_layers=4, last="identity")
        if config["use_word_fea"] == True:
            self.word_temporal = temporal_lstm(config["n_segment"], input_size=768, output_size=512, hidden_size=512, num_layers=4, last="identity")
        concat_dim = 768
        if config["use_audio_fea"] == True:
            concat_dim += config['audio_fea_dim']
        
        hidden_size1,hidden_size2,hidden_size3 = config["hidden_size"]
        
        self.feat_fc = nn.Conv1d(concat_dim, hidden_size1, 1, padding=0)
        self.activ = nn.LeakyReLU(0.1) 
        self.dropout = nn.Dropout(p=0.3)
        self.AUhead = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size3),
                nn.BatchNorm1d(hidden_size3),
                nn.Linear(hidden_size3, 8),
                )
       
        self.transformer = TransEncoder(inc=hidden_size1, outc=hidden_size2, dropout=0.3, nheads=4, nlayer=4)
        self.config = config
    
    def forward(self,sample):
        vis_inp = sample["frames"]
        bs,seq_len,c,H,W = vis_inp.shape
        vis_fea = self.vis_extractor(vis_inp)
        concats = [vis_fea]
        if self.config["use_audio_fea"]:
            audio_fea = sample["audios"]
            concats.append(audio_fea)
        
        if self.config["use_word_fea"]:
            word_fea = sample["words"]
            concats.append(word_fea)

        concat_fea = torch.cat(concats,dim=2)
        feat = torch.transpose(concat_fea,1,2)
        feat = self.feat_fc(feat)
        feat = self.activ(feat)
        out = self.transformer(feat)

        out = torch.transpose(out, 1, 0)
        out = torch.reshape(out, (bs*seq_len, -1))

        out = self.AUhead(out)
        out = out.view(bs,seq_len,-1)
        return out

class AU_fusion(nn.Module):
    def __init__(self, config):
        super(AU_fusion, self).__init__()
        self.vis_extractor = feature_extractor(config)
           if config["use_audio_fea"] == True:
            self.audio_temporal = temporal_lstm(config["n_segment"], input_size=768, output_size=512, hidden_size=512, num_layers=4, last="identity")
        if config["use_word_fea"] == True:
            self.word_temporal = temporal_lstm(config["n_segment"], input_size=768, output_size=512, hidden_size=512, num_layers=4, last="identity")
        concat_dim = 768
        if config["use_audio_fea"] == True:
            concat_dim += config['audio_fea_dim']
        
        hidden_size1,hidden_size2,hidden_size3 = config["hidden_size"]
        
        self.feat_fc = nn.Conv1d(concat_dim, hidden_size1, 1, padding=0)
        self.activ = nn.LeakyReLU(0.1) 
        self.dropout = nn.Dropout(p=0.3)
        self.AUhead = nn.Sequential(
                nn.Linear(hidden_size2, hidden_size3),
                nn.BatchNorm1d(hidden_size3),
                nn.Linear(hidden_size3, 12),
                )
       
        self.transformer = TransEncoder(inc=hidden_size1, outc=hidden_size2, dropout=0.3, nheads=4, nlayer=4)
        self.config = config
    
    def forward(self,sample):
        vis_inp = sample["frames"]
        bs,seq_len,c,H,W = vis_inp.shape
        vis_fea = self.vis_extractor(vis_inp)
        concats = [vis_fea]
        if self.config["use_audio_fea"]:
            audio_fea = sample["audios"]
            concats.append(audio_fea)
        
        if self.config["use_word_fea"]:
            word_fea = sample["words"]
            concats.append(word_fea)

        concat_fea = torch.cat(concats,dim=2)
        feat = torch.transpose(concat_fea,1,2)
        feat = self.feat_fc(feat)
        feat = self.activ(feat)
        out = self.transformer(feat)

        out = torch.transpose(out, 1, 0)
        out = torch.reshape(out, (bs*seq_len, -1))

        out = self.AUhead(out)
        out = out.view(bs,seq_len,-1)
        return out
