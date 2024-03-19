import yaml
import torch
import argparse

from torch.utils.data import DataLoader
import torch.distributed as dist
import os
from torch import optim
import utils.misc as misc
from torch.utils.tensorboard import SummaryWriter
import datetime
from data.dataset import build_seq_dataset
from engine_BLB import train_one_epoch,evaluate
from tqdm import tqdm
import numpy as np
import random
from models.BasicLearningBranch import VA_fusion,AU_fusion,EXP_fusion

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

def get_model_ema(model, ema_ratio=1e-3):
    def ema_func(avg_param, param, num_avg):
        return (1 - ema_ratio) * avg_param + ema_ratio * param
    return torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_func)

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

    if config["task"] == "VA":
        model = VA_fusion(config).cuda()
    elif config["task"] == "AU":
        model = AU_fusion(config).cuda()
    elif config["task"] == "EXP":
        model = EXP_fusion(config).cuda()

    if config["resume"] != None:
        print("resume from ",config["resume"] )
        state_dict = torch.load(config["resume"],"cpu")
        model.load_state_dict(state_dict)

    if use_dp:
        num_gpus = torch.cuda.device_count()
        gpus = config["device"]
        model = torch.nn.DataParallel(model, device_ids=gpus)


    train_set = build_seq_dataset(config,"train")
    test_set = build_seq_dataset(config,"test")
    trainloader = DataLoader(train_set, batch_size=config["batch_size"],num_workers=config["num_works"],shuffle=True,drop_last=True)
    testloader = DataLoader(test_set, batch_size=config["batch_size"],num_workers=config["num_works"],shuffle=False,drop_last=False)

    now = datetime.datetime.now()
    now = now.strftime('%m_%d_%Y')
    log_dir = config["log_dir"] + "/" + config["task"] + "_" + config["dataset_type"] + "_lr_" + str(config["lr"]) + "_" + config["optim"]  + "_bz_" +str(config["batch_size"])+ "_"+ now 
    checkpoint_dir =  config["checkpoint_dir"] + "/" + config["task"] + "_" + config["dataset_type"] + "_lr_" + str(config["lr"]) + "_" + config["optim"]  + "_bz_" +str(config["batch_size"]) + "_"+ now 
    config["save_log_path"] = log_dir
    config["checkpoint_path"] = checkpoint_dir

    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = config["num_epochs"]

    if config["optim"] == "AdamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], betas=(0.9, 0.999), weight_decay=0.05)
    elif config["optim"] == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], betas=(0.9, 0.999))
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

    if config["use_ema"] == True:
        model_wa = get_model_ema(model,config["ema_ratio"])
        model_final = model_wa.module


    for epoch in range(num_epochs):
        if epoch<config["start_epoch"]:
            scheduler.step()
        else:
            is_save = False
            if config["use_ema"]:
                step, model_wa = train_one_epoch(model, trainloader, epoch, log_writer, metric_logger, optimizer, scheduler, config, step, model_wa)
            else:
                step, _ = train_one_epoch(model, trainloader, epoch, log_writer, metric_logger, optimizer, scheduler, config, step, None)
            if epoch % save_epoch == 0:
                is_save = True
            if config["use_ema"]:
                avg_loss, res = evaluate(epoch, model_wa, testloader, test_metric_logger, config, is_save)
            else:
                avg_loss, res = evaluate(epoch, model, testloader, test_metric_logger, config, is_save)
            scheduler.step()