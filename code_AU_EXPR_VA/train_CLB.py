import yaml
import torch
import argparse

from torch.utils.data import DataLoader
import torch.distributed as dist
import os
from torch import optim
import utils.misc as misc
from torch.utils.tensorboard import SummaryWriter
from models.dual_branch import DBCNet
import datetime
from data.dataset import build_seq_dataset
from transformers import get_linear_schedule_with_warmup
from engine_DCL import train_one_epoch,evaluate
from tqdm import tqdm
import numpy as np
import random


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def read_yaml_to_dict(yaml_path):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value

def checkfile(img_path):
    lists = os.listdir(img_path)
    for l in tqdm(lists):
        imgs = os.listdir(os.path.join(img_path,l))
        for im in imgs:
            if os.path.getsize(os.path.join(img_path,l,im))<1:
                print(os.path.join(img_path,l,im))

if __name__ == '__main__':
    setup_seed(20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/original_AU_task.yml")
    args = parser.parse_args()
    yml_path = args.config
    config = read_yaml_to_dict(yml_path)
    use_dp = config["use_dp"]
    num_gpus = len(config["device"])
    config = read_yaml_to_dict(yml_path)
    model = DBCNet(config).cuda()

    if config["resume"] != None:
        state_dict = torch.load(config["resume"],"cpu")
        model.load_state_dict(state_dict)

    if use_dp:
        num_gpus = torch.cuda.device_count()
        gpus = config["device"]
        model = torch.nn.DataParallel(model, device_ids=gpus)
    

    
    train_set1 = build_seq_dataset(config,"train")
    trainloader1 = DataLoader(train_set1, batch_size=config["batch_size"],num_workers=config["num_works"],shuffle=True)
    config["s1_data_length"] = len(train_set1)
    train_set2 = build_seq_dataset(config,"s2_train")
    trainloader2 = DataLoader(train_set2, batch_size=config["batch_size"],num_workers=config["num_works"],shuffle=True)
    test_set = build_seq_dataset(config,"test")
    testloader = DataLoader(test_set, batch_size=config["batch_size"],num_workers=config["num_works"],shuffle=False)

    now = datetime.datetime.now()
    now = now.strftime('%m_%d_%Y')
    log_dir = config["log_dir"] + "/" + config["task"] + "_" + config["dataset_type"] + "_lr_" + str(config["lr"]) + "_" + config["optim"]  + "_bz_" +str(config["batch_size"])+ "_"+ now 
    checkpoint_dir =  config["checkpoint_dir"] + "/" + config["task"] + "_" + config["dataset_type"] + "_lr_" + str(config["lr"]) + "_" + config["optim"]  + "_bz_" +str(config["batch_size"]) + "_"+ now 

    config["save_log_path"] = log_dir
    config["checkpoint_path"] = checkpoint_dir

    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = config["num_epochs"]

    if config["optim"] == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], betas=(0.9, 0.999))
    elif config["optim"] == "AdamW":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], betas=(0.9, 0.999), weight_decay=0.05)

    elif config["optim"] == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], momentum=config["momentum"],
                          weight_decay=config["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)

    metric_logger = misc.MetricLogger(delimiter="  ")
    test_metric_logger = misc.MetricLogger(delimiter="  ")
    log_writer = SummaryWriter(log_dir=log_dir)

    best_acc = 0
    best_epoch = 0
    save_best = config["save_best"]
    save_epoch = config["save_epoch"]
    step = 0

    use_BMSE = config["use_balanced_mse"]
    BMC_cri = None
    if use_BMSE:
        from Balanced_MSE import BMCLoss
        init_noise_sigma = 1.0
        BMC_cri = BMCLoss(init_noise_sigma).cuda()
        optimizer.add_param_group({'params': BMC_cri.noise_sigma, 'lr': 1e-4, 'name': 'noise_sigma'})

    for epoch in range(config["start_epoch"],num_epochs):
        is_save = False
        step = train_one_epoch(model, trainloader1, trainloader2, epoch, log_writer, metric_logger, optimizer, scheduler, config, step)
        if epoch % save_epoch == 0:
            is_save = True
        avg_loss, res = evaluate(epoch, model, testloader, test_metric_logger, config, is_save)
        scheduler.step()