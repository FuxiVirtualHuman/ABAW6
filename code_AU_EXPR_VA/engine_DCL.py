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
import pickle

def CCC_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + 1e-10)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2 * rho * x_s * y_s / ((x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2) + 1e-10)
    return 1 - ccc


def compute_AU_loss_BCE(pred, label1,label2,alp,config):
    weights = torch.tensor(config["weights"]).cuda()
    bz,seq,_  = pred.shape
    label1 = label1.view(bz*seq,-1)
    label2 = label2.view(bz*seq,-1)

    pred = pred.view(bz*seq,-1)

    cri_AU = nn.BCEWithLogitsLoss(weights)
    alp = alp[0]

    loss_AU = 10*(alp* cri_AU(pred, label1) + (1-alp) * cri_AU(pred, label2))

    return loss_AU

def compute_AU_loss_BCE_test(pred, label,config):
    weights = torch.tensor(config["weights"]).cuda()
    bz,seq,_  = pred.shape
    label = label.view(bz*seq,-1)
    pred = pred.view(bz*seq,-1)

    cri_AU = nn.BCEWithLogitsLoss(weights)
    bz,c = pred.shape
    cls_loss = cri_AU(pred, label)
    
    AU_pred = nn.Sigmoid()(pred)
    loss_AU = 10*cls_loss 
    return loss_AU,AU_pred






def compute_VA_loss(pred, label1, label2, alp):
    alp = alp[0] 
    bz,seq,_  = pred.shape
    label1 = label1.view(bz*seq,-1)
    label2 = label2.view(bz*seq,-1)

    pred = pred.view(bz*seq,-1)

    loss_V1 = CCC_loss(pred[:,0],label1[:,0])
    loss_A1 = CCC_loss(pred[:,1],label1[:,1])
    loss1 = 0.5 * (loss_V1 + loss_A1)
    loss_V2 = CCC_loss(pred[:,0],label2[:,0])
    loss_A2 = CCC_loss(pred[:,1],label2[:,1])
    loss2 = 0.5 * (loss_V2 + loss_A2)

    loss = alp * loss1 + (1-alp) * loss2
   
    return loss


def compute_VA_loss_test(pred, label):
    bz,seq,_  = pred.shape
    pred = pred.view(bz*seq,-1)
    label = label.view(bz*seq,-1)
    loss_V = CCC_loss(pred[:,0],label[:,0])
    loss_A = CCC_loss(pred[:,1],label[:,1])

    loss = 0.5 * (loss_A + loss_V)
    return loss,pred



def compute_EXP_loss(pred, label1, label2,  alp, config):
    weights = torch.tensor(config['weights']).cuda()
    bz,seq,_  = pred.shape

    label1 = label1.view(bz*seq,-1)
    label2 = label2.view(bz*seq,-1)

    pred = pred.view(bz*seq,-1)
    cri_EXP = nn.CrossEntropyLoss(weights)

    loss_EXP = alp * cri_EXP(pred, label1.long().squeeze(-1)) + (1-alp) * cri_EXP(pred, label2.long().squeeze(-1)) 
    EXP_pred = torch.softmax(pred, dim = 1)
    return loss_EXP,EXP_pred


def compute_EXP_loss_test(pred, label, config):
    weights = torch.tensor(config['weights']).cuda()
    bz,seq,_  = pred.shape
    label = label.view(bz*seq,-1)
    pred = pred.view(bz*seq,-1)
 
    cri_exp = nn.CrossEntropyLoss(weights)
    cls_loss = cri_exp(pred,label.squeeze(-1))
    EXP_pred = torch.softmax(pred, dim = 1)
    return cls_loss,EXP_pred



def process_sample(samples,config):
    name, frames, label,frame_names = samples["name"], samples["frames"], samples["labels"] , samples["frame_names"]
    frames, label = frames.cuda(), label.cuda()

    sample = {
        "frames": frames,
        "label":label,
        "frame_names":frame_names

    }
    if config["use_audio_fea"] == True:
        audios = samples["audio_inp"].cuda()
        sample["audios"] = audios
    
    return sample


def train_one_epoch(model, data_loader1, data_loader2, epoch, log_writer, metric_logger, optim, scheduler, config, step):
    accum_iter = 20
    use_dp = config["use_dp"]
    task = config["task"]
    print_freq = config["print_freq"]
    header = 'Training Epoch: [{}]'.format(epoch)
    device = config["device"]
    use_BMSE = config["use_balanced_mse"]
    len_trainloader = len(data_loader1)
    if log_writer is not None and is_main_process():
        print('log_dir: {}'.format(log_writer.log_dir))
    model.train(True)
    ret_CLB =config["output_CLB"]
    
    
    if use_dp:
        model.module.BLB.eval()
        model.module.CLB.vis_extractor.eval()
    else:
        model.BLB.eval()
        model.CLB.vis_extractor.eval()
    

    t = tqdm(enumerate(zip(data_loader1, data_loader2)))
    for idx, data in t:
        samples1,samples2 = data
        samples1 = process_sample(samples1,config)
        samples2 = process_sample(samples2,config)

        label1 = samples1["label"]
        label2 = samples2["label"]

        optim.zero_grad()
        
        if task == "EXP":
            out, a = model(samples1,samples2,"train")
            loss,_ = compute_EXP_loss(out, label1, label2, a, config)

        elif task == "AU":
            out, a = model(samples1,samples2,"train")
            loss = compute_AU_loss_BCE(out, label1, label2, a, config)
        
        elif task == "VA":
            out_v,out_a,a = model(samples1,samples2,"train")
            out = torch.cat([out_v,out_a],dim=2)
            loss = compute_VA_loss(out, label1, label2, a)
                
        

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        optim.step()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optim.state_dict()['param_groups'][0]['lr'])

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if (step + 1) % accum_iter == 0:
            iters = step + 1
            log_writer.add_scalar("loss", loss_value, iters)
            print("Epoch:" + str(epoch) + " step: " + str(step) + " loss: " + str(loss_value))
        
        step = step + 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return step


def evaluate(epoch, model, data_loader, test_metric_logger, config, is_save_checkpoints=False):
    
    header = 'Test:'
    use_dp = config["use_dp"]
    print_freq = config["print_freq"]
    device = config["device"]
    task = config["task"]
    acc_logger = misc.Task_Logger(task, "test", save_log_file=os.path.join(config["save_log_path"],"test_metric.json"), save_pred_pkl=config["save_log_path"],final_test_pkl= config["final_test_pkl"])
    model.eval()

    with torch.no_grad():
        t = enumerate(test_metric_logger.log_every(data_loader, print_freq, header))
        for step, samples in t:
            name = samples["name"]
            samples = process_sample(samples,config)    
            label = samples["label"]
            if task == "EXP":
                out, a = model(samples,samples,"test")
                loss,EXP_pred = compute_EXP_loss_test(out,label,config)
                pred = EXP_pred
                pred_arr = EXP_pred.detach().cpu().numpy()
            elif task == "AU":
                out, a = model(samples,samples,"test")
                loss, AU_pred = compute_AU_loss_BCE_test(out,label,config)
                pred = AU_pred
                pred_arr = AU_pred.detach().cpu().numpy()
            
            elif task == "VA":
                vout, aout, a = model(samples,samples,"test")
                out = torch.cat([vout, aout],dim=2)
                loss,pred= compute_VA_loss_test(out, label)
                pred_arr = pred.detach().cpu().numpy()
            
            bz,seq, _ = out.shape
            loss_value = loss.item()
            test_metric_logger.update(loss=loss_value)

            label = label.view(bz*seq,-1)
            label_arr = label.detach().cpu().numpy()
            new_names = []
            frame_names  = samples["frame_names"]
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


def pred(epoch, model, data_loader, config, is_save=True):
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
                vout, aout, a = model(sample,sample,"test")

                bz,seq,_ = vout.shape
                vout[vout<-1] = -1
                vout[vout>1] = 1
                aout[aout<-1] = -1
                aout[aout>1] = 1

                vout = vout.detach().cpu().numpy()
                aout = aout.detach().cpu().numpy()

            elif task =="AU":
                out, a = model(sample,sample,"test")
                bz,seq,_ = out.shape
                AU_pred = nn.Sigmoid()(out).detach().cpu().numpy()

            elif task =="EXP":
                out, a = model(sample,sample,"test") 
                bz,seq,_ = out.shape       
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

    if is_save:       
        with open(config["save_log_path"] + "/pred_test.pkl","wb") as f:
            pickle.dump(pred_dict,f)

    return pred_dict

                    
        