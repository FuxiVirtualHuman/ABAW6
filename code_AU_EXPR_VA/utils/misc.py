# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf
import utils.metrics as metrics
import json
import os
import pickle
import numpy as np

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter


    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class Task_Logger:
    def __init__(self, task, mode, save_log_file=None,save_pred_pkl=None,final_test_pkl=None):
        self.task = task
        self.preds = {}
        self.labels = {}
        self.losses = []
        self.mode = mode
        self.save_log_file = save_log_file
        self.save_pred_pkl = save_pred_pkl
        self.final_test_pkl= final_test_pkl
        self.AU_class_names = ["AU1","AU2","AU4","AU6","AU7","AU10","AU12","AU15","AU23","AU24","AU25","AU26"]
        self.EXP_class_names = ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise","Other"]
        self.EMO_class_names = ["Adoration","Amusement","Anxiety","Disgust","Empathic-Pain","Fear","Surprise"]
        self.epoch = 0
    
    def update(self, name, pred, label, loss):
        # print(pred.shape,label.shape)
        for i in range(len(name)):
            self.preds[name[i]] = pred[i]
            self.labels[name[i]] = label[i]
        self.losses.append(loss)

    def dict_to_str(self,dict):
        logs = ""
        for k,v in dict.items():
            logs += k + ":  " +str(v) + " "
        return logs
    
    def write_log(self, avg_loss, res):
        write_json_dict = {'time':time.asctime(), 'Avg losses': avg_loss}
        if self.task == "AU":
            F1s, mean_F1 = res
            for i in range(12):
                write_json_dict[self.AU_class_names[i]+"_F1"] = F1s[i]
            write_json_dict["mean F1"] = mean_F1
        
        elif self.task == "EXP":
            F1s, mean_F1 = res
            print(F1s,mean_F1)
            for i in range(len(F1s)):
                write_json_dict[self.EXP_class_names[i]+"_F1"] = F1s[i]
            write_json_dict["mean F1"] = mean_F1
        elif self.task == "VA":
            cccv,ccca,mean_ccc = res
            write_json_dict["ccc_v"] = cccv
            write_json_dict["ccc_a"] = ccca
            write_json_dict["mean ccc"] = mean_ccc

        elif self.task == "emo":
            pccs, mean_pcc = res
            for i in range(len(pccs)):
                write_json_dict[self.EMO_class_names[i]+"_pcc"] = pccs[i]
            write_json_dict["mean_pcc"] = mean_pcc

        logs = self.dict_to_str(write_json_dict)
        with open(self.save_log_file , mode="a") as f:
            f.write(logs + "\n")

            
    
    def summary(self):
        res = []
        avg_loss = sum(self.losses)/len(self.losses)
        
        final_pred = []
        final_target = []
        tmp = {}
        print("++++++++++===total number of validation data is ", len(self.preds))

        with open(self.final_test_pkl,"rb") as f:
            final_test_items = pickle.load(f)

        for k,v in final_test_items.items():
            if isinstance(v,str):
                tt = v.split(",")
                tt = [float(f) for f in tt]
                v = tt
            v = np.array(v)
            if self.task == "VA":
                VA_value = self.preds[k]
                VA_value[VA_value<-1] = -1
                VA_value[VA_value>1] = 1
                final_pred.append(VA_value)
                final_target.append(v)
                tmp[k] = {"pred":VA_value,"label":v}
            
            if self.task == "EXP":
                pred = self.preds[k]
                label = v
                final_pred.append(pred)
                final_target.append(v)
                tmp[k] = {"pred":pred,"label":v}
            
            if self.task == "AU":
                pred = self.preds[k]
                label = v.split(",")
                label = [int(t) for t in label]
                final_pred.append(pred)
                final_target.append(label)
                tmp[k] = {"pred":pred,"label":label}

           


        # for k,v in self.preds.items():
        #     if self.task == "VA":
        #         v[v<-1] = -1
        #         v[v>1] = 1
        #     elif self.task == "emo":
        #         v[v<0] = 0
        #         v[v>1] = 1
        #     final_pred.append(v)
        #     final_target.append(self.labels[k])
        #     tmp[k] = {"pred":v,"label":self.labels[k]}
        
        
        if self.task == "AU":
            F1s, F1_mean = metrics.compute_AU_F1(final_pred,final_target)
            res = [F1s, F1_mean]
        elif self.task == "EXP":
            F1s, F1_mean = metrics.compute_EXP_F1(final_pred,final_target)
            res = [F1s, F1_mean]
        elif self.task == "VA":
            cccv, ccca = metrics.compute_VA_CCC(final_pred,final_target)
            res = [cccv, ccca, 0.5*(cccv+ccca)]
            print(res)
        elif self.task == "emo":
            pccs, mean_pcc = metrics.compute_emo_PCC(final_pred,final_target)
            res = [pccs, mean_pcc]
            print(res)

        if self.save_log_file!=None:
            self.write_log(avg_loss,res)

        self.preds = {}
        self.labels = {}
        print(self.save_pred_pkl + "/epoch_"+str(self.epoch) + str(res[-1])[:6] + ".pkl")
        with open(self.save_pred_pkl + "/epoch_"+str(self.epoch) + str(res[-1])[:6] + ".pkl", "wb") as f:
            pickle.dump(tmp,f)
        self.epoch += 1
        
        return avg_loss, res





def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    # def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
    #     self._scaler.scale(loss).backward(create_graph=create_graph)
    #     if update_grad:
    #         if clip_grad is not None:
    #             assert parameters is not None
    #             self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
    #             norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
    #         else:
               
    #             self._scaler.unscale_(optimizer)
    #             norm = get_grad_norm_(parameters)
                
    #         self._scaler.step(optimizer)
    #         self._scaler.update()
    #         fla = True
    #         for p in parameters:
    #             if fla == True:
    #                 print(p)
    #                 fla = False
            
    #     else:
    #         norm = None
    #     return norm
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        self._scaler.step(optimizer)
        self._scaler.update()
        


    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x