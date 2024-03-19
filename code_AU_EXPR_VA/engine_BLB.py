import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
import math
import torch
import time

import utils.misc as misc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.modules.distance import PairwiseDistance
import torch.distributed as dist
import os
from tqdm import tqdm
import sys
import pickle



def CCC_loss(x, y):
    y = y.view(-1)
    x = x.view(-1)
    vx = x - torch.mean(x) 
    vy = y - torch.mean(y) 
    rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2)))+1e-8)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
    return 1-ccc


def compute_AU_loss_BCE(pred, label, config):
    weights = torch.tensor(config["weights"]).cuda()
    bz,seq,_  = pred.shape
    label = label.view(bz*seq,-1)
    pred = pred.view(bz*seq,-1)

    cri_AU = nn.BCEWithLogitsLoss(weights)
    bz,c = pred.shape
    cls_loss = cri_AU(pred, label)
    
    AU_pred = nn.Sigmoid()(pred)

    return cls_loss,AU_pred

def compute_VA_loss(Vout,Aout,label):
    Vout = torch.clamp(Vout,-1,1)
    Aout = torch.clamp(Aout,-1,1)
    bz,seq,_  = Vout.shape
    label = label.view(bz*seq,-1)
    Vout = Vout.view(bz*seq,-1)
    Aout = Aout.view(bz*seq,-1)
    ccc_loss = CCC_loss(Vout[:,0],label[:,0]) + CCC_loss(Aout[:,0],label[:,1])
    mse_loss = nn.MSELoss()(Vout,label[:,0]) + nn.MSELoss()(Aout,label[:,1])
    
    loss = ccc_loss 
    return loss,mse_loss,ccc_loss


def compute_EXP_loss(pred, label,config):
    weights = torch.tensor(config["weights"]).cuda()
    bz,seq,_  = pred.shape
    label = label.view(bz*seq,-1)
    pred = pred.view(bz*seq,-1)
 
    cri_exp = nn.CrossEntropyLoss(weights)
    cls_loss = cri_exp(pred,label.squeeze(-1))
    EXP_pred = torch.softmax(pred, dim = 1)
    return cls_loss,EXP_pred



def train_one_epoch(model, data_loader, epoch, log_writer, metric_logger, optim, scheduler, config, step, model_wa):
    accum_iter = config["accum_iter"]
    use_dp = config["use_dp"]
    task = config["task"]
    print_freq = config["print_freq"]
    header = 'Training Epoch: [{}]'.format(epoch)
    device = config["device"]
    len_trainloader = len(data_loader)
    if log_writer is not None and is_main_process():
        print('log_dir: {}'.format(log_writer.log_dir))
    model.train(True)
    if use_dp:
        model.module.vis_extractor.eval()
    else:
        model.vis_extractor.eval()
    t = enumerate(metric_logger.log_every(data_loader, print_freq, header))
    for idx, samples in t:
        name, frames, label = samples["name"], samples["frames"], samples["labels"]
        frames, label = frames.cuda(), label.cuda()
        sample = {
            "frames": frames,
        }
        if config["use_audio_fea"] == True:
            audios = samples["audio_inp"].cuda()
            sample["audios"] = audios
        
        model.zero_grad()
        if task == "VA":
            vout, aout = model(sample)
            loss,mse_loss,ccc_loss = compute_VA_loss(vout, aout, label)
            loss_value = loss.item()
            metric_logger.update(loss=loss_value, mse_loss = mse_loss.item(), ccc_loss = ccc_loss.item())

        elif task == "AU":
            out = model(sample)
            loss, AU_pred =  compute_AU_loss_BCE(out,label,config)
            loss_value = loss.item()
            metric_logger.update(loss=loss_value)
        
        elif task =="EXP":
            out = model(sample)
            loss,EXP_pred = compute_EXP_loss(out,label,config)
            
            loss_value = loss.item()
            metric_logger.update(loss=loss_value)


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        loss.backward()

        optim.step()
        
        if config["use_ema"] and (idx + 1)%config["ema_interval"] == 0:
            model_wa.update_parameters(model)

        metric_logger.update(lr=optim.state_dict()['param_groups'][0]['lr'])

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if (step+1) % accum_iter == 0:
            iter = step + 1
            log_writer.add_scalar("loss", loss_value, iter)
        step +=1 

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return step, model_wa


def evaluate(epoch, model, data_loader, test_metric_logger, config, is_save_checkpoints=False):
    header = 'Test:'
    use_dp = config["use_dp"]
    print_freq = config["print_freq"]
    device = config["device"]
    task = config["task"]
    acc_logger = misc.Task_Logger(task, "test", save_log_file=os.path.join(config["save_log_path"],"test_metric.json"), save_pred_pkl=config["save_log_path"], final_test_pkl= config["final_test_pkl"])
    if config["use_ema"]:
        model = model.module
    
    model.eval()

    with torch.no_grad():
        t = enumerate(test_metric_logger.log_every(data_loader, print_freq, header))
        for idx, samples in t:
            name, frames, label, frame_names = samples["name"], samples["frames"], samples["labels"], samples["frame_names"]
            frames, label = frames.cuda(), label.cuda()
            sample = {
                "frames": frames,
            }
            if config["use_audio_fea"] == True:
                audios = samples["audio_inp"].cuda()
                sample["audios"] = audios
  
            if task == "VA":
                vout, aout = model(sample)
                loss,mse_loss,ccc_loss = compute_VA_loss(vout, aout, label)
                bz,seq, _ = vout.shape
                vout = vout.view(bz*seq,-1)
                aout = aout.view(bz*seq,-1)
                pred_arr = torch.cat([vout,aout],dim=1).detach().cpu().numpy()
            elif task =="AU":
                out = model(sample)
                loss,AU_pred =  compute_AU_loss_BCE(out,label,config)
                bz,seq, _ = out.shape
                pred_arr = AU_pred.detach().cpu().numpy()
            elif task =="EXP":
                out = model(sample)
                bz,seq, _ = out.shape
                loss,EXP_pred = compute_EXP_loss(out,label,config)
                pred_arr = EXP_pred.detach().cpu().numpy()
            
            loss_value = loss.item()
            test_metric_logger.update(loss=loss_value)

            
            
            label = label.view(bz*seq,-1)
            label_arr = label.detach().cpu().numpy()
            new_names = []
            
            for i in range(bz):
                tmp = frame_names[i].split(" ")
                for nn in tmp:
                    new_names.append(nn)

            acc_logger.update(new_names, pred_arr, label_arr, loss_value)

    
    test_metric_logger.synchronize_between_processes()
    avg_loss, res = acc_logger.summary()
    test_metric_logger.meters['loss_avg'].update(avg_loss, n=1)
    test_metric_logger.meters['overall_accuracy'].update(res[-1], n=1)
    print('* Overall Accuracy: {overall_accuracy.avg:.3f}  loss {loss_avg.global_avg:.3f}'
          .format(overall_accuracy = test_metric_logger.overall_accuracy, loss_avg = test_metric_logger.meters["loss_avg"]))
    if is_save_checkpoints:
        save_path = os.path.join(config["checkpoint_path"], "epoch_" + str(epoch) +"_time_" + str(time.asctime())+"_metric_" + str(res[-1])[:6] + ".pth")
        if use_dp:
            torch.save(model.module.state_dict(),save_path)
        else:
            torch.save(model.state_dict(),save_path)

    return avg_loss, res


def pred(epoch, model, data_loader, config):
    header = 'Pred:'
    use_dp = config["use_dp"]
    print_freq = config["print_freq"]
    device = config["device"]
    task = config["task"]
    model.eval()
    pred_dict = {}
    with torch.no_grad():
        t = tqdm(enumerate(data_loader))
        for idx, samples in t:
            name, frames, frame_names = samples["name"], samples["frames"], samples["frame_names"]
            frames = frames.cuda()
            sample = {
                "frames": frames,
            }
            if config["use_audio_fea"] == True:
                audios = samples["audio_inp"].cuda()
                sample["audios"] = audios
            
            if task == "VA":
                vout, aout = model(sample)
                bz,seq,_ = vout.shape
                vout[vout<-1] = -1
                vout[vout>1] = 1
                aout[aout<-1] = -1
                aout[aout>1] = 1

                vout = vout.detach().cpu().numpy()
                aout = aout.detach().cpu().numpy()

            elif task =="AU":
                out = model(sample)
                AU_pred = nn.Sigmoid()(out).detach().cpu().numpy()

            elif task =="EXP":
                out = model(sample)
                bz,seq,_ = out.shape
                out = out.view(bz*seq,-1)
                EXP_pred = torch.softmax(out, dim = 2).detach().cpu().numpy()

            new_names = []
            
            for i in range(bz):
                tmp = frame_names[i].split(" ")
                for n in range(len(tmp)):
                    if task == "VA":
                        pred_dict[tmp[n]] = [vout[i,n],aout[i,n]]
                    elif task == "AU":
                        pred_dict[tmp[n]] = AU_pred[i,n]
                    elif task == "EXP":
                        pred_dict[tmp[n]] = EXP_pred[i,n]
                    print( tmp[n], pred_dict[tmp[n]])


    with open(config["save_log_path"] + "/pred_test.pkl","wb") as f:
        pickle.dump(pred_dict,f)
