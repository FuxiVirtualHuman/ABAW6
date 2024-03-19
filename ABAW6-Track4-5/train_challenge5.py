import os
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from data.dataset_challenge5 import DatasetChallenge5, collate_fn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable
import random
from scipy.stats import pearsonr
from models.linear_modal import Vallina_fusion, mixup_data, Vallina_fusion_visual_all
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.distributions import MultivariateNormal as MVN
from torch.nn.modules.loss import _Loss
from torch.nn import TripletMarginLoss
import argparse
from collections import OrderedDict
import torch.nn.init as init

# Exponential Moving Average
def get_model_ema(model, ema_ratio=1e-3):
    def ema_func(avg_param, param, num_avg):
        return (1 - ema_ratio) * avg_param + ema_ratio * param
    return torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_func)

def bmc_loss_md(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    I = torch.eye(pred.shape[-1]).cuda()
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma)).cuda()

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss_md(pred, target, noise_var)
    
class RelPosLoss(_Loss):
    def __init__(self, count_n=100):
        super(RelPosLoss, self).__init__()
        self.count_n = count_n
        self.triplet_loss = TripletMarginLoss(margin=0.1)
        
    def forward(self, pred, target):
        loss = 0
        bs = pred.size(0)
        for i in range(self.count_n):
            idx = torch.randperm(bs)[:3]
            dist01 = torch.dist(target[idx[0]], target[idx[1]])
            dist02 = torch.dist(target[idx[0]], target[idx[2]])
            dist12 = torch.dist(target[idx[1]], target[idx[2]])
            neg_index = idx[torch.argmin(torch.stack([dist12, dist02, dist01]))]
            pos_indexes = (idx != neg_index).nonzero(as_tuple=True)[0]
            loss_i = self.triplet_loss(pred[pos_indexes[0]], pred[pos_indexes[1]], pred[neg_index]) + \
                self.triplet_loss(pred[pos_indexes[1]], pred[pos_indexes[0]], pred[neg_index])
            loss += loss_i
        return loss/self.count_n



def concordance_correlation_coefficient(y_true, y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------s
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    # --------
    # >>> from sklearn.metrics import concordance_correlation_coefficient
    # >>> y_true = [3, -0.5, 2, 7]
    # >>> y_pred = [2.5, 0.0, 2, 8]
    # >>> concordance_correlation_coefficient(y_true, y_pred)
    # 0.97678916827853024
    # """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator



def CCC_Loss(x, y):
    ccc = 2*torch.cov(torch.stack((x, y))) / (x.var() + y.var() + (x.mean() - y.mean())**2)
    return 1 - ccc[0][1]

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
            for i in range(class_num):
                self.alpha[i, :] = 0.25
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
def tensors_to_cuda(data):
    """
    Recursively move all tensors in the input data to CUDA, if CUDA is available.
    
    :param data: A dictionary which may contain other dictionaries or torch.Tensors.
    :return: Same structure as input with all tensors moved to CUDA.
    """
    if torch.cuda.is_available():
        if isinstance(data, dict):
            # Recursively apply to dictionary elements
            return {key: tensors_to_cuda(value) for key, value in data.items()}
        elif isinstance(data, torch.Tensor):
            # Move tensor to CUDA
            return data.to('cuda')
        else:
            # If data is not a dictionary or tensor, return it as is
            return data
    else:
        # If CUDA is not available, return the data unchanged
        return data

def train_ERI(epoch, loader, net, optimizer, best_Exp_score, net3=None):
    lr = optimizer.param_groups[0]['lr']
    print(f"* Epoch {epoch}, foldi:{fold_i}, lr:{lr:.5f}, timestamp:{timestamp}")
    preds, gt = [], []
    loss_sum = 0.0
    net = net.train()
    # print('Total batch',len(loader))
    step_n = len(loader)
    b = '{l_bar}{bar:40}{r_bar}{bar:-10b}'
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format=b, ncols=160)
    logger = []
    time0 = time.time()
    for i, data in pbar:
        if use_mea and i % 10 == 0:
            # model_wa.update_parameters(net)
            average_weights = OrderedDict()
            for key in net.state_dict():
                average_weights[key] = 0.999 * net.state_dict()[key] + 0.001 * model_wa.state_dict()[key]
            model_wa.load_state_dict(average_weights)
            
        data = tensors_to_cuda(data)
        labels = data['labels']
        if use_mixup:
            imgs, labels, _ = mixup_data(imgs, labels)
        
        time_data = time.time() - time0
        optimizer.zero_grad()

        
        res = net(data)
        pred = res['out']

        if use_mixup:
            labels = res['labels_mixup']
        # Exp_loss

        # loss_l1 = criterion_l1(pred, labels)
        loss_l1 = criterion_regression(pred, labels)
        # loss_l1 = criterion_bmc(pred[:,:7], labels)
        
        loss = loss_l1
        
        if use_distill:
            with torch.no_grad():
                res_teacher = net_teacher(data)
                pred_teacher = res_teacher['out']
            loss += criterion_l2(pred, pred_teacher)    
            
        writer.add_scalar(f'loss/train', loss.item(), epoch * len(train_dataloader) + i)
    
        loss.backward()
        optimizer.step()
        # scheduler.step(i)

        loss_sum += loss.item()
        avg_loss = loss_sum / (i+1)
        preds.append(pred.detach().cpu().numpy())
        gt.append(labels.cpu().numpy())
        
        time_train = time.time() - time0 - time_data  
        time0 = time.time()      
        pbar.set_description(f'[Train epoch {epoch}]\t loss:{loss.item():.4f}({avg_loss:.4f}) time_data:{time_data:.1f} time_train:{time_train:.1f}' )
        # pbar.set_description(f'[Train epoch {epoch}(lr:{lr:.5f})]\t loss:{loss.item():.4f}({avg_loss:.4f})' )
        logger.append(f"epoch: {epoch}, step: {i},  Loss: {loss.item():.4f}\n")
    
    scheduler.step(epoch)
    # metrics (pcc)
    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    pcc = [np.round(pearsonr(preds[:,i], gt[:,i])[0], 4) for i in range(preds.shape[1])]
    pcc_avg = np.mean(np.array(pcc))
    
    print(f'[Train epoch {epoch}]\t loss_avg: {avg_loss:.4f}\t pcc:{pcc_avg:.4f}({pcc})')
    logger.append(f'[Train epoch {epoch}]\t loss_avg: {avg_loss:.4f}\t pcc:{pcc_avg:.4f}({pcc})\n')
    with open(os.path.join(log_save_path, f'{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)

    return avg_loss


def test_ERI(epoch, loader, net, best_acc, best_pcc, patience_cur, save_step=10, patience=10, save_res=False):
    # print("train {} epoch".format(epoch))
    preds, gt = [], []
    net = net.eval()
    loss_sum = 0
    b = '{l_bar}{bar:50}{r_bar}{bar:-60b}'
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format=b, ncols=160)
    
    # for results
    vid_id = []

    
    logger = []
    for i, data in pbar:
        data = tensors_to_cuda(data)
        labels = data['labels']

        # print(imgs)
        with torch.no_grad():
            res = net(data)
        pred = res['out']
        # Exp_loss
        # loss_l1 = criterion_l1(pred, labels)
        loss_l1 = criterion_regression(pred, labels)
        # loss_l1 = criterion_bmc(pred[:,:7], labels)

        loss = loss_l1
        writer.add_scalar(f'loss/val', loss.item(), epoch * len(train_dataloader) + i)

        loss_sum += loss.item()
        avg_loss = loss_sum / (i+1)
        
        vid_id += data['vid']
        preds.append(pred.detach().cpu().numpy())
        gt.append(labels.cpu().numpy())

        pbar.set_description(f'[Test epoch {epoch}]\t loss:{loss.item():.4f}({avg_loss:.4f})')
          
    # metrics (pcc)
    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    # set maximum = 1
    if out_dim==14:
        preds_class = np.argmax(preds[:,7:], axis=1)
        # for i in range(len(preds_class)):
            # preds[i, preds_class[i]] = 1 
    else:
        preds_class = np.argmax(preds, axis=1)
    gt_class = np.argmax(gt, axis=1)
    matrix = confusion_matrix(gt_class, preds_class)
    acc = matrix.diagonal()/matrix.sum(axis=1)
    acc = [np.round(acc[i], 4) for i in range(6)]

    pcc = [np.round(pearsonr(preds[:,i], gt[:,i])[0], 4) for i in range(6)]
    pcc_avg = np.mean(np.array(pcc))

    metric = pcc_avg
    if epoch % save_step == 0:
        torch.save({'state_dict': net.state_dict()}, os.path.join(ck_save_path,f'{timestamp}_{epoch}.pt'))
    if metric > best_pcc:
        patience_cur = patience
        best_acc = metric
        best_pcc = pcc_avg
        torch.save({'state_dict': net.state_dict()}, os.path.join(ck_save_path,f'{timestamp}_best.pt'))
        print("Found new best model, saving to disk...")
        if save_res:
            with open(os.path.join(pred_save_path, f'{timestamp}_val.pkl'), 'wb') as f:
                pickle.dump({'vid':vid_id, 'preds':preds, 'gt':gt, 'dist':np.linalg.norm(preds-gt, axis=1)},f)

    else:
        patience_cur -= 1    

    print(f'[Test epoch {epoch}]\t loss_avg: {avg_loss:.4f}(best:{best_acc:.4f})\t pcc:{pcc_avg:.4f}(best:{best_pcc:.4f})({pcc}) acc:{acc}\n')
    logger.append(f'[Test epoch {epoch}]\t loss_avg: {avg_loss:.4f}\t pcc:{pcc_avg:.4f}({pcc})\n')
    with open(os.path.join(log_save_path, f'{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    
    return best_acc, best_pcc


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    # 为其他类型的层添加初始化，例如卷积层
    elif isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:  # 对于权重矩阵
                init.xavier_uniform_(param.data)
            else:  # 对于偏置项
                init.zeros_(param.data)
            
def parse_args():

    parser = argparse.ArgumentParser(description='ABAW6-challenge5.')

    parser.add_argument('--task', type=str, default='reaction', choices=['humor', 'reaction', 'stress'],
                        help='Specify the task (humour, reaction, stress).')
    parser.add_argument('--feature', nargs='+', default=['egemaps', 'bert-4-sentence-level', 'vggface2'], help='Specify the features used (only one).')
    parser.add_argument('--emo_dim', default='physio-arousal',
                        help='Specify the emotion dimension, only relevant for stress (default: arousal).')
    parser.add_argument('--normalize', default=False, action='store_true',
                        help='Specify whether to normalize features (default: False).')
    parser.add_argument('--win_len', type=int, default=200,
                        help='Specify the window length for segmentation (default: 200 frames).')
    parser.add_argument('--hop_len', type=int, default=100,
                        help='Specify the hop length to for segmentation (default: 100 frames).')
    parser.add_argument('--d_rnn', type=int, default=256,
                        help='Specify the number of hidden states in the RNN (default: 64).')
    parser.add_argument('--rnn_n_layers', type=int, default=4,
                        help='Specify the number of layers for the RNN (default: 1).')
    parser.add_argument('--rnn_bi', default=True, action='store_true',
                        help='Specify whether the RNN is bidirectional or not (default: False).')
    parser.add_argument('--d_fc_out', type=int, default=64,
                        help='Specify the number of hidden neurons in the output layer (default: 64).')
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--linear_dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs (default: 100).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Specify the batch size (default: 256).')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Specify initial learning rate (default: 0.0001).')
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Specify number of random seeds to try (default: 5).')
    parser.add_argument('--result_csv', default=None, help='Append the results to this csv (or create it, if it '
                                                           'does not exist yet). Incompatible with --predict')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--reduce_lr_patience', type=int, default=5, help='Patience for reduction of learning rate')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Specify whether to use gpu for training (default: False).')
    parser.add_argument('--cache', default=True, action='store_true',
                        help='Specify whether to cache data as pickle file (default: False).')
    parser.add_argument('--save_path', type=str, default='preds',
                        help='Specify path where to save the predictions (default: preds).')
    parser.add_argument('--predict', action='store_true',
                        help='Specify when no test labels are available; test predictions will be saved '
                             '(default: False). Incompatible with result_csv')
    parser.add_argument('--regularization', type=float, required=False, default=0.0,
                        help='L2-Penalty')
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Specify model which is to be evaluated; no training with this option (default: False).')

    # parser.add_argument('--mask_a_length', type=str, default='50,50')
    # parser.add_argument('--mask_b_length', type=str, default='10,10')
    parser.add_argument('--block_num', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dropout_mmatten', type=float, default=0.5)
    parser.add_argument('--dropout_mtatten', type=float, default=0.2)
    parser.add_argument('--dropout_ff', type=float, default=0.2)
    parser.add_argument('--dropout_subconnect', type=float, default=0.2)
    parser.add_argument('--dropout_position', type=float, default=0.2)
    parser.add_argument('--dropout_embed', type=float, default=0.2)
    parser.add_argument('--dropout_fc', type=float, default=0.2)
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--h_mma', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--embed', type=str, default='temporal')
    parser.add_argument('--levels', type=int, default=5)
    parser.add_argument('--ksize', type=int, default=3)
    parser.add_argument('--ntarget', type=int, default=7)

    parser.add_argument('--fold', required=False,type=int, default=0,
                        help='specify fold for validation, 0 for offical, 1-5 for cross validation')
    parser.add_argument('-seed', required=False,type=int, default=20,
                        help='random seed.')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    args = parse_args()

    import time
    setup_seed(args.seed)
    os.chdir('/data/Workspace/ABAW/code_ABAW6/')
    use_cuda = True
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # pretrained = '20230220-124201_10'

    task = 'ERI' 
    model_name = 'EffMulti'  # ['baseline','mutual','dropout','resnet']
    ck_save_path = f'/project/qiuf/ABAW/checkpoints'
    log_save_path = f'/project/qiuf/ABAW/logs'
    pred_save_path = f'/project/qiuf/ABAW/test'
    tensorboard_save_path = f'/project/qiuf/ABAW/tensorboard'
    os.makedirs(ck_save_path,exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    os.makedirs(pred_save_path, exist_ok=True)
    os.makedirs(tensorboard_save_path, exist_ok=True)

    log_path = f'{tensorboard_save_path}/{timestamp}' 
    if not os.path.exists(log_path): 
        os.mkdir(log_path)
    # 设置TensorBoard日志目录
    writer = SummaryWriter(log_dir=log_path)
    
    print(f'************** NOW IS {task} TASK. TIMESTAMP {timestamp} ******************')
    # training parameters
    bz = 16
    lr = 1e-6 # 5e-5
    patience = 30
    save_step = 5
    img_max = 32  # 32
    audio_max = 320 # 320
    text_max = 128
    out_dim = 6
    lr_visual, lr_audio, lr_text = 1e-6, 1e-4, 1e-4
    # feaname_v, feaname_a, feaname_t = ['mae_emb_large'], ['hubertbase'], ['chatglm3']
    # feaname_v, feaname_a, feaname_t = [], [], ['chatglm3_whipser_large_lora_21st']
    # feaname_v, feaname_a, feaname_t = [], ['wav2vec2', 'hubertbase'], []
    # feaname_v, feaname_a, feaname_t = [], [], ['chatglm3_whisper_large', 'chatglm3_whipser_large_lora_21st']
    # feaname_v, feaname_a, feaname_t = [], [], ['chatglm3_whipser_large_lora_21st', 'chatglm3_whipser_large_lora_14th']
    # feaname_v, feaname_a, feaname_t = [], ['wav2vec2'], []
    feaname_v, feaname_a, feaname_t = ['mae_au_new'], ['wav2vec2'], ['chatglm3_whisper_large', 'chatglm3_whisper_large_lora_21st']
    # feaname_v, feaname_a, feaname_t = ['mae_emb'], [], []
    
    use_attention = False
    use_triplets = False
    save_res = True
    use_dual = False
    use_ccc = False
    use_shift = False
    use_mixup = False
    use_mea = False
    use_distill = False
    distill_model = '20230310-021658' 
    pretrained = ''
    weight_ce = 0.0
    shuffle_ratio = 1
    fold_i = args.fold
    # loss
    criterion_l1 = nn.L1Loss(reduction='mean')
    criterion_l2 = nn.MSELoss(reduction='mean')
    criterion_ce = nn.CrossEntropyLoss(reduction='mean')
    # criterion_huber = nn.HuberLoss(delta=0.2)  
    criterion_bmc = BMCLoss(init_noise_sigma=1.)
    criterion_triplet = RelPosLoss(count_n=100)
    criterion_bce = nn.BCEWithLogitsLoss()    
    criterion_regression = criterion_l2

    # dataset
    data_root = '/project/ABAW6/data/challenge5'
    dataset_ERI = DatasetChallenge5 

    collate = collate_fn(feaname_v=feaname_v, feaname_a=feaname_a, feaname_t=feaname_t)
    train_dataset = dataset_ERI(data_root, 'train', fold_i, img_max=img_max,  audio_max=audio_max, text_max=text_max,
                                feature_a=feaname_a, feature_t=feaname_t, feature_v=feaname_v,shuffle_ratio=shuffle_ratio)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=bz,shuffle=True,num_workers=8,
                                  collate_fn=collate, drop_last=True, pin_memory=True)
    
    val_dataset = dataset_ERI(data_root, 'valid', fold_i, img_max=img_max, audio_max=audio_max, text_max=text_max,
                              feature_a=feaname_a, feature_t=feaname_t, feature_v=feaname_v,shuffle_ratio=0)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=bz,shuffle=False,num_workers=8,
                                collate_fn=collate, drop_last=True, pin_memory=False)
    
    
    # model
    net = Vallina_fusion(out_dim=out_dim, in_dims=val_dataset.feature_dims, feaname_v=feaname_v, feaname_a=feaname_a, feaname_t=feaname_t)
    if use_distill:
        net_teacher = Vallina_fusion(out_dim=out_dim, in_dims=val_dataset.feature_dims, feaname_v=feaname_v, feaname_a=feaname_a, feaname_t=feaname_t)
        ckpt = torch.load(os.path.join(ck_save_path, f'{distill_model}_best.pt'))
        net_teacher.load_state_dict(ckpt['state_dict'])
        net_teacher.cuda().eval()
        print("=> loaded teacher model!")

    net = net.cuda()
    torch.backends.cudnn.benchmark = True
    
    if pretrained:
        checkpoint = torch.load(os.path.join(ck_save_path, f'{pretrained}.pt'))
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded pretrained model {}".format(pretrained))
    else:
        net.apply(init_weights)

    
    # 分别为不同部分的参数创建参数组
    params = [
        {'params': net.visual_encoder.parameters(), 'lr': lr_visual, 'betas':(0.0, 0.99)},
        {'params': net.audio_encoder.parameters(), 'lr': lr_audio, 'betas':(0.0, 0.99)},
        {'params': net.text_encoder.parameters(), 'lr': lr_text ,'betas':(0.0, 0.99)}
    ]

    # 将模型中未明确分组的其他所有参数都设置为另一个学习率
    for name, param in net.named_parameters():
        if "visual_encoder" not in name and "audio_encoder" not in name and "text_encoder" not in name:
            params.append({'params': param, 'lr': lr ,'betas':(0.0, 0.99)})
    
    optimizer = torch.optim.Adam(params)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2, 1e-8)
    
    patience_cur = patience
    step_i = 0
    best_score = float('inf')
    best_pcc = 0
    
    # os.chdir('/data/Workspace/ABAW')
    os.system("git add .")
    os.system(f"git commit -m  training_{task}_{timestamp}")
    os.system("git push")
    # os.chdir('/data/Workspace/ABAW/code_ABAW5/')

    if use_mea:
        # model_wa = get_model_ema(net)
        model_wa = Vallina_fusion(out_dim=out_dim, in_dims=val_dataset.feature_dims, feaname_v=feaname_v, feaname_a=feaname_a, feaname_t=feaname_t)
        model_wa.cuda()
        
    for epoch in range(500000000):
        avg_loss = train_ERI(epoch, train_dataloader, net, optimizer, best_score)
        # net = get_model_ema(net).module        
        if use_mea: 
            net_wa = model_wa
            best_score, best_pcc = test_ERI(epoch, val_dataloader, net_wa, best_score, best_pcc, patience_cur, 
                                    save_step=save_step, patience=patience, save_res=save_res)
        else:
            best_score, best_pcc = test_ERI(epoch, val_dataloader, net, best_score, best_pcc, patience_cur, 
                                                save_step=save_step, patience=patience, save_res=save_res)  