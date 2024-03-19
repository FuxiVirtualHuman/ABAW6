import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
import math
import torch
import time
from data.dataset import build_seq_dataset
import utils.misc as misc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.modules.distance import PairwiseDistance
import torch.distributed as dist
import os
from tqdm import tqdm
import pickle
import datetime
import argparse
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import random
from models.BasicLearningBranch import VA_fusion,AU_fusion,EXP_fusion
from engine_DCL import evaluate
from models.dual_branch import DBCNet


def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/original_AU_task.yml")
    parser.add_argument("--task", default="eval")

    args = parser.parse_args()
    yml_path = args.config
    config = read_yaml_to_dict(yml_path)
    use_dp = config["use_dp"]
    num_gpus = len(config["device"])
    config = read_yaml_to_dict(yml_path)


    model = DBCNet(config).cuda()

    if config["resume"] != None:
        print("resume from ",config["resume"])
        state_dict = torch.load(config["resume"],"cpu")
        model.load_state_dict(state_dict)
    
    
    if use_dp:
        num_gpus = torch.cuda.device_count()
        gpus = config["device"]
        model = torch.nn.DataParallel(model, device_ids=gpus)

    
    now = datetime.datetime.now()
    now = now.strftime('%m_%d_%Y')
    log_dir = config["log_dir"] + "/" + config["task"] + "_" + config["dataset_type"] + "_lr_" + str(config["lr"]) + "_" + config["optim"]  + "_bz_" +str(config["batch_size"])+ "_"+ now 
    checkpoint_dir =  config["checkpoint_dir"] + "/" + config["task"] + "_" + config["dataset_type"] + "_lr_" + str(config["lr"]) + "_" + config["optim"]  + "_bz_" +str(config["batch_size"]) + "_"+ now 

    config["save_log_path"] = log_dir
    config["checkpoint_path"] = checkpoint_dir

    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


    if args.task == "eval":
        test_set = build_seq_dataset(config,"test")
        testloader = DataLoader(test_set, batch_size=config["batch_size"],num_workers=config["num_works"],shuffle=False,drop_last=False)
        test_metric_logger = misc.MetricLogger(delimiter="  ")
        avg_loss, res = evaluate(0, model, testloader, test_metric_logger, config, False)
    
    elif args.task == "pred":
        pred_set = build_seq_dataset(config,"pred")
        predloader = DataLoader(pred_set , batch_size=config["batch_size"],num_workers=config["num_works"],shuffle=False)
        print(len(predloader))
        pred(0,model,predloader,config)