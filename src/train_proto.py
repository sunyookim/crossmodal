import argparse
import math
import numpy as np
import os
import pickle
import random
from sklearn import model_selection
import torch
import torch.nn.functional as F

from decimal import Decimal
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_utils import MetaFolderTwo, split_meta_both, r_at_k, mk_retrieve, np_transform, collate_recipe, collate_recipe_verbose, mk_dataloader_1, mk_dataloader_2, XmodalDataset
from models import *
from losses import *
from tools import *
from datetime import datetime
import sys
import pprint



def train_proto(args, net, loss_fn, data_iter, device):
    train_losses = []
    train_accs = []
    best_acc = 0
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_proto, betas=(0, 0.999))
    for iteration in range(args.iterations):
        # Should be changed after loader is complete.
        x_s, x_q, y_s, y_q = to_device(next(data_iter), device, squeeze=True)
        x = torch.cat([x_s, x_q])
        y = torch.cat([y_s, y_q])
        if net.mode.endswith("a"):
            x = net.audio_enc(x) #(n_way*(n_support+n_query), 128)
        elif net.mode.endswith("i"):
            x = net.image_enc(x) #(n_way*(n_support+n_query), 128)
        elif net.mode.endswith("t"):
            x = net.text_enc(x) #(n_way*(n_support+n_query), 128)
        # s_prototypes = s_emb.reshape(args.classes, args.train_shots, -1)
        # s_prototypes = s_prototypes.sum(1)
        loss, acc = loss_fn(x, target=y,
                            n_support=args.train_shots)
        # dists = euclidean_dist(q_emb, s_prototypes)
        # log_p_y = F.log_softmax(-dists, dim=1)
        # labels_onehot = F.one_hot(y_q, args.classes)
        # loss = (-log_p_y)*labels_onehot.float()
        # print(f"log_p_y.shape = {log_p_y.shape}")
        # print(f"labels_onehot.shape = {labels_onehot.shape}")
        # print(f"loss = {loss}")
        # print(f"loss.shape = {loss.shape}")
        # loss = torch.sum(loss, -1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_accs.append(acc.item())
    return np.mean(train_losses), np.mean(train_accs)

def train_align(args, net, loss_fn, data_iter, device, feat_penalties, margin=0.1):
    align_losses = []
    
    if args.reptile:
        ori_net = net.clone()
        meta_optimizer = torch.optim.Adam(net.parameters(), lr=args.meta_lr, betas=(0, 0.999))
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_align, betas=(0, 0.999))
    for iteration in range(args.iterations):
        feat1, feat2 = [], []
        # x1_s, x1_q, y1_s, y1_q, x2_s, x2_q, y2_s, y2_q = next(data_iter) #image/audio
        x1_s, x1_q, y1_s, y1_q, x2_s, x2_q, y2_s, y2_q = to_device(next(data_iter), device, squeeze=True) #image/audio
        # o1, o2 = net.align(x1_s, x2_s)
        # y_s = y2_s
        if net.mode == 'i2t':
            o1, o2 = net.align_layer(net.image_enc(x1_s)), net.text_enc(x2_s)
            y_s = y2_s
        elif net.mode == 't2i':
            o1, o2 = net.image_enc(x1_s), net.align_layer(net.text_enc(x2_s))
            y_s = y1_s
        elif net.mode == 'i2a':
            o1, o2 = net.align_layer(net.image_enc(x1_s)), net.audio_enc(x2_s)
            y_s = y2_s
            if '3' in args.sharing_layer:
                feat1.append(activation['img_l3'])
                feat2.append(activation['aud_l3'])
            if '4' in args.sharing_layer:
                feat1.append(activation['img_l4'])
                feat2.append(activation['aud_l4'])
        elif net.mode == 'a2i':
            o1, o2 = net.image_enc(x1_s), net.align_layer(net.audio_enc(x2_s))
            y_s = y1_s
            if '3' in args.sharing_layer:
                feat1.append(activation['aud_l3'])
                feat2.append(activation['img_l3'])
            if '4' in args.sharing_layer:
                feat1.append(activation['aud_l4'])
                feat2.append(activation['img_l4'])
        
        loss = loss_fn(o1, o2, y_s, feat1, feat2, feat_penalties, len(args.sharing_layer), margin=margin)
        
        # before = list(net.parameters())[0].clone()
        # before = list(net.audio_enc.parameters())[0].clone()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # after = list(net.parameters())[0].clone()
        # after = list(net.audio_enc.parameters())[0].clone()
        align_losses.append(loss.item())
    if args.reptile:
        ori_net.point_grad_to(net)
        meta_optimizer.step()
        net = ori_net

    return np.mean(align_losses)

def eval_proto(args, net, loss_fn, data_iter, device, best_acc, best_model_path):
    eval_loss = []
    eval_acc = []
    net.eval()
    for iteration in range(args.iterations):
        # Should be changed after loader is complete.
        x_s, x_q, y_s, y_q = to_device(next(data_iter), device)
        x = torch.cat([x_s, x_q])
        y = torch.cat([y_s, y_q])
        if net.mode.startswith("a"):
            x = net.align_layer(net.audio_enc(x)) #(n_way*(n_support+n_query), 128)
        elif net.mode.startswith("i"):
            x = net.align_layer(net.image_enc(x)) #(n_way*(n_support+n_query), 128)
        elif net.mode.startswith("t"):
            x = net.align_layer(net.text_enc(x)) #(n_way*(n_support+n_query), 128)
        loss, acc = loss_fn(x, target=y,
                            n_support=args.train_shots)
        eval_loss.append(loss.item())
        eval_acc.append(acc.item())
    eval_loss, eval_acc = np.mean(eval_loss), np.mean(eval_acc)
    if eval_acc >= best_acc:
        torch.save(model.state_dict(), best_model_path % best_acc)
        best_acc = eval_acc
    postfix = ' (Best)' if eval_acc >= best_acc else ' (Best: {:.3f})'.format(best_acc)
    print('Val Loss: {:.3f}, Val Acc: {:.3f}{}'.format(eval_loss, eval_acc, postfix),flush=True)
    return best_acc, eval_loss, eval_acc

def train_croma(net, loss_fn, optimizer, data_iter, iterations, device):
    support_losses = []
    query_losses = []
    support_correct = 0
    query_correct = 0
    support_samples = 0
    query_samples = 0
    net.train()
    for iteration in range(iterations):
        x_s, x_q, y_s, y_q = to_device(next(data_iter), device, squeeze=True)
        out = net.forward1(x_s)
        loss = loss_fn(out, y_s)
        pred = out.data.max(1, keepdim=True)[1]
        matchings = pred.eq(y_s.data.view_as(pred).type(torch.LongTensor).to(device))
        support_correct += matchings.sum()
        support_samples += len(y_s)
        support_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        out = net.forward1(x_q)
        loss = loss_fn(out, y_q)
        pred = out.data.max(1, keepdim=True)[1]
        matchings = pred.eq(y_q.data.view_as(pred).type(torch.LongTensor).to(device))
        query_correct += matchings.sum()
        query_samples += len(y_q)
        query_losses.append(loss.item())

    sup_acc = float(support_correct)/support_samples
    que_acc = float(query_correct)/query_samples
    return np.mean(support_losses), sup_acc, np.mean(query_losses), que_acc

def eval_croma(net, loss_fn, data_iter, iterations, device, best_acc, best_model_path):
    support_losses = []
    query_losses = []
    support_correct = 0
    query_correct = 0
    support_samples = 0
    query_samples = 0
    net.eval()
    with torch.no_grad():
        for iteration in range(iterations):
            x_s, x_q, y_s, y_q = to_device(next(data_iter), device, squeeze=True)
            out = net.forward2(x_s)
            loss = loss_fn(out, y_s)
            pred = out.data.max(1, keepdim=True)[1]
            matchings = pred.eq(y_s.data.view_as(pred).type(torch.LongTensor).to(device))
            support_correct += matchings.sum()
            support_samples += len(y_s)
            support_losses.append(loss.item())
            
            out = net.forward2(x_q)
            loss = loss_fn(out, y_q)
            pred = out.data.max(1, keepdim=True)[1]
            matchings = pred.eq(y_q.data.view_as(pred).type(torch.LongTensor).to(device))
            query_correct += matchings.sum()
            query_samples += len(y_q)
            query_losses.append(loss.item())

    sup_loss = np.mean(support_losses)
    sup_acc = float(support_correct)/support_samples
    eval_loss = np.mean(query_losses)
    eval_acc = float(query_correct)/query_samples
    if eval_acc >= best_acc:
        torch.save(model.state_dict(), best_model_path % best_acc)
        best_acc = eval_acc
    postfix = ' (Best)' if eval_acc >= best_acc else ' (Best: {:.3f})'.format(best_acc)
    print('Val Support Loss: {:.3f}, Val Support Acc: {:.3f}, Val Query Loss: {:.3f}, '
        'Val Query Acc: {:.3f}{}'.format(sup_loss, sup_acc, eval_loss, eval_acc, postfix),flush=True)
    return best_acc, eval_loss, eval_acc

def eval_croma_same(net, loss_fn, data_iter, iterations, device, best_acc, best_model_path):
    support_losses = []
    query_losses = []
    support_correct = 0
    query_correct = 0
    support_samples = 0
    query_samples = 0
    net.eval()
    with torch.no_grad():
        for iteration in range(iterations):
            x_s, x_q, y_s, y_q = to_device(next(data_iter), device, squeeze=True)
            out = net.forward1(x_s)
            loss = loss_fn(out, y_s)
            pred = out.data.max(1, keepdim=True)[1]
            matchings = pred.eq(y_s.data.view_as(pred).type(torch.LongTensor).to(device))
            support_correct += matchings.sum()
            support_samples += len(y_s)
            support_losses.append(loss.item())
            
            out = net.forward1(x_q)
            loss = loss_fn(out, y_q)
            pred = out.data.max(1, keepdim=True)[1]
            matchings = pred.eq(y_q.data.view_as(pred).type(torch.LongTensor).to(device))
            query_correct += matchings.sum()
            query_samples += len(y_q)
            query_losses.append(loss.item())

    sup_loss = np.mean(support_losses)
    sup_acc = float(support_correct)/support_samples
    eval_loss = np.mean(query_losses)
    eval_acc = float(query_correct)/query_samples
    if eval_acc >= best_acc:
        torch.save(model.state_dict(), best_model_path % best_acc)
        best_acc = eval_acc
    postfix = ' (Best)' if eval_acc >= best_acc else ' (Best: {:.3f})'.format(best_acc)
    print('Val Support Loss: {:.3f}, Val Support Acc: {:.3f}, Val Query Loss: {:.3f}, '
        'Val Query Acc: {:.3f}{}'.format(sup_loss, sup_acc, eval_loss, eval_acc, postfix),flush=True)
    return best_acc, eval_loss, eval_acc

def plot(align_l, proto_l, proto_a, eval_l, eval_a, args):
    
    file_path = os.path.join(model_save_root, f"{args.mode}_acc_loss.png")
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")

    plt.subplot(1, 2, 1)
    # plt.plot(align_l,label="train_align_loss")
    plt.plot(proto_l,label="train_proto_loss")
    plt.plot(eval_l,label="eval_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(proto_a, label="train_accuracy")
    plt.plot(eval_a,label="eval_accuracy")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.savefig(file_path, dpi=300)
## main

if __name__ == "__main__":
    parser = argparse.ArgumentParser('few shot alignment with cos loss')

    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--iseed', type=int, help='idx dict random seed', default=0)
    parser.add_argument('--cuda', default=0, type=int, help='use cuda')
    parser.add_argument('--num-workers', default=4, type=int, help='cuda device')
    parser.add_argument('--margin', default=0.5, type=float, help='margin in loss fn')
    # parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--vis_m', default="nop18", type=str, help='vision model')
    parser.add_argument('--aud_m', default="aud", type=str, help='audio model')
    # parser.add_argument('--data_dir', default="../data/clustered_ESC_CIFAR_TRAIN.pkl", type=str, help='data directory path to read')

    # parser.add_argument('--no-pre', action='store_true', help='dont record untrained model performance', default=True)
    # parser.add_argument('--soft-label', action='store_true', help='supervised alignment', default=False)

    # few shot args
    parser.add_argument('-n', '--classes', default=5, type=int, help='classes in base-task (N-way)')
    # parser.add_argument('--mode', default='a2i', type=str, help='Alignment direction. i=image, t=text, a=audio. ex)i2t, a2i')
    parser.add_argument('--mode', default='i2a', type=str, help='Alignment direction. i=image, t=text, a=audio. ex)i2t, a2i')
    parser.add_argument('--fc-dim', default=128, type=int, help='dimension of embeddings')
    parser.add_argument('--train-shots', default=5, type=int, help='(train) shots per class (K-shot)')
    parser.add_argument('--test-shots', default=10, type=int, help='(test) shots per class (K-shot)')
    parser.add_argument('--meta-iterations', default=1000, type=int, help='number of meta iterations')
    parser.add_argument('--iterations', default=3, type=int, help='number of base iterations')
    parser.add_argument('--test-iterations', default=1, type=int, help='number of base iterations')
    parser.add_argument('--train-align-batch', default=1, type=int, help='minibatch size in alignment metatrain task')
    parser.add_argument('--batch', default=1, type=int, help='minibatch size in base task')
    parser.add_argument('--meta-lr', default=1., type=float, help='meta learning rate')
    parser.add_argument('--lr-proto', default=1e-4, type=float, help='base learning rate')
    parser.add_argument('--lr-align', default=1e-4, type=float, help='base learning rate')
    parser.add_argument('--train-ratio', default=0.7, type=float, help='Percentage of train data in total data')
    parser.add_argument('--val-epoch', default=1, type=int, help='Meta-evaluation every ... base-tasks')
    parser.add_argument('--ntrain-tasks', default=10, type=int, help='Number of train tasks')
    parser.add_argument('--eval-tasks', default=8, type=int, help='Number of eval tasks')
    parser.add_argument('--reptile', default=True, type=bool, help='Train align step with reptile method')
    parser.add_argument('-l', default='tri', type=str, help='loss fn')
    parser.add_argument('--sharing', default=True, type=bool, help='Add soft sharing loss')
    parser.add_argument('--sharing-layer','-sl', type=str, nargs='+', default=[])
    parser.add_argument('--share-penalties', '-sp', nargs='+', default=[0.1, 0.1])
    # parser.add_argument('--fig-filename', type=str, required=True)
    parser.add_argument('--croma', action='store_true', help='use croma training', default=False)

    args = parser.parse_args()


    """
    About mode
    a2i : align audio embeddings to image space
    i2a : align image embeddings to audio space

    t2i : align text embeddings to image space
    i2t : align image embeddings to text space
    """
    # print(args, flush=True)
    # print(pprint.pformat(args), flush=True)
    print("#"*70)
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()), flush=True)
    print("#"*70)
    assert args.mode in ["a2i", "i2a", "i2t", "t2i", "a2a"]
    if len(args.share_penalties) > 2: args.share_penalties = args.share_penalties[2:]
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.croma:
        enc_i = ImageEncoder(args.fc_dim)
        enc_t = TextEncoder(args.fc_dim)
        enc_a = ImageEncoder(args.fc_dim, mode='aud_pre')
        model = CROMA(audio_enc=ImageEncoder(args.fc_dim,mode=args.aud_m),
                        text_enc=TextEncoder(args.fc_dim),image_enc=ImageEncoder(args.fc_dim,mode=args.vis_m), 
                        fc_dim=args.fc_dim, num_classes=args.classes, device=device, mode=args.mode)
        model.to(device)
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    else:
        model = ProtoModel(audio_enc=ImageEncoder(args.fc_dim,mode=args.aud_m),text_enc=TextEncoder(args.fc_dim),image_enc=ImageEncoder(args.fc_dim,mode=args.vis_m), fc_dim=args.fc_dim, num_classes=args.classes, device=device, mode=args.mode)
        model.to(device)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_root = os.path.join(cur_dir, 'checkpoints', timestamp)
    if not os.path.exists(model_save_root):
        os.makedirs(model_save_root)
    best_model_path = os.path.join(model_save_root, "best_model_%.3f.pt")
    last_model_path = os.path.join(model_save_root, "last_model.pt")
    best_acc = 0.0

    #train

    align_losses = []; train_losses = []; train_accs = []
    eval_losses = []; eval_accs = []
    for iteration in range(args.meta_iterations):
        if args.mode in ["a2i", "i2a", "a2a"]:
            vision_dataset = XmodalDataset(task_mode = 'visual', train_mode='train', 
                                        BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224)
            audio_dataset  = XmodalDataset(task_mode='audio', train_mode='train', 
                                        BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224)
            align_dataset  = XmodalDataset(task_mode='align', train_mode='train', 
                                        BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224)

            vision_dataloader = DataLoader(vision_dataset, batch_size=args.batch, shuffle=True)
            audio_dataloader  = DataLoader(audio_dataset, batch_size=args.batch, shuffle=True)
            align_dataloader  = DataLoader(align_dataset, batch_size=args.train_align_batch, shuffle=True)

            vision_iter = iter(vision_dataloader)
            audio_iter  = iter(audio_dataloader)
            align_iter  = iter(align_dataloader)
            
            if args.mode.startswith("a"):
                val_audio_iter  = iter(
                    DataLoader(
                        XmodalDataset(task_mode='audio', train_mode='val', BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224),
                        batch_size=args.batch,
                        shuffle=True
                    )
                )
                # test_audio_iter = iter(
                #     DataLoader(
                #         XmodalDataset(task_mode='audio', train_mode='test',  BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224),
                #         batch_size=args.batch,
                #         shuffle=True
                #     )
                # )
            else:
                val_vision_iter  = iter(
                    DataLoader(
                        XmodalDataset(task_mode='visual', train_mode='val',   BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224),
                        batch_size=args.batch,
                        shuffle=True
                    )
                )
                # test_vision_iter  = iter(
                #     DataLoader(
                #         XmodalDataset(task_mode='visual', train_mode='test',   BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224),
                #         batch_size=args.batch,
                #         shuffle=True
                #     )
                # )

        elif args.mode in ["i2t", "t2i"]:
            vision_dataset = XmodalDataset(task_mode = 'visual', train_mode='train',  
                                        BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224)
            text_dataset  = XmodalDataset(task_mode='text', train_mode='train',  
                                        BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224)
            align_dataset  = XmodalDataset(task_mode='align', train_mode='train',  
                                        BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224)
            
            vision_dataloader = DataLoader(vision_dataset, batch_size=args.batch, shuffle=True)
            text_dataloader  = DataLoader(text_dataset, batch_size=args.batch, shuffle=True)
            align_dataloader  = DataLoader(align_dataset, batch_size=args.train_align_batch, shuffle=True)

            vision_iter = iter(vision_dataloader)
            text_iter  = iter(text_dataloader)
            align_iter  = iter(align_dataloader)

            if args.mode.startswith("i"):
                val_vision_iter  = iter(
                    DataLoader(
                        XmodalDataset(task_mode='visual', train_mode='val',   BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224),
                        batch_size=args.batch,
                        shuffle=True
                    )
                )
                test_vision_iter  = iter(
                    DataLoader(
                        XmodalDataset(task_mode='visual', train_mode='test',   BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224),
                        batch_size=args.batch,
                        shuffle=True
                    )
                )
            else:
                val_text_iter  = iter(
                    DataLoader(
                        XmodalDataset(task_mode='text', train_mode='val',   BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224),
                        batch_size=args.batch,
                        shuffle=True
                    )
                )
                test_text_iter  = iter(
                    DataLoader(
                        XmodalDataset(task_mode='text', train_mode='test',   BS = args.iterations, n_way = args.classes, k_shot = args.train_shots, k_query = args.test_shots, resize = 224),
                        batch_size=args.batch,
                        shuffle=True
                    )
                )
        
        #train alignment
        #set soft params sharing layers
        if args.mode[0] != args.mode[-1]:
            if '3' in args.sharing_layer:
                model.image_enc.model.layer3.register_forward_hook(get_activation('img_l3'))
                model.audio_enc.model.layer3.register_forward_hook(get_activation('aud_l3'))
            if '4' in args.sharing_layer:
                model.image_enc.model.layer4.register_forward_hook(get_activation('img_l4'))
                model.audio_enc.model.layer4.register_forward_hook(get_activation('aud_l4'))
            feat_penalties = [float(p) for p in args.share_penalties]
            
            align_loss = train_align(args, model, tri_loss, align_iter, device, feat_penalties, margin=0.1)
            align_losses.append(align_loss)

        if args.croma:
            net = model.clone()
            cross_entropy = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr_proto, betas=(0, 0.999))
            if args.mode.endswith("i"):
                sloss, sacc, qloss, qacc = train_croma(net, cross_entropy, optimizer, vision_iter, args.iterations, device)
            elif args.mode.endswith("a"):
                sloss, sacc, qloss, qacc = train_croma(net, cross_entropy, optimizer, audio_iter, args.iterations, device)
            elif args.mode.endswith("t"):
                sloss, sacc, qloss, qacc = train_croma(net, cross_entropy, optimizer, text_iter, args.iterations, device)
            print('Support Loss: {:.3f}, Support Acc: {:.3f}, Query Loss: {:.3f}, Query Acc: {:.3f}'.format(sloss, sacc, qloss, qacc), flush=True)
            train_losses.append(qloss); train_accs.append(qacc)
            model.point_grad_to(net)
            meta_optimizer.step()
            
            if iteration % args.val_epoch == 0:
                net = model.clone()
                if args.mode[0] != args.mode[-1]:
                    if args.mode.startswith("i"):
                        best_acc, eval_loss, eval_acc = eval_croma(net, cross_entropy, val_vision_iter, args.iterations, device, best_acc, best_model_path)
                        # best_acc, eval_loss, eval_acc = eval_croma(net, cross_entropy, test_vision_iter, args.test_iterations, device, best_acc, best_model_path)
                    elif args.mode.startswith("a"):
                        best_acc, eval_loss, eval_acc = eval_croma(net, cross_entropy, val_audio_iter, args.iterations, device, best_acc, best_model_path)
                        # best_acc, eval_loss, eval_acc = eval_croma(net, cross_entropy, test_audio_iter, args.test_iterations, device, best_acc, best_model_path)
                    elif args.mode.startswith("t"):
                        best_acc, eval_loss, eval_acc = eval_croma(net, cross_entropy, val_text_iter, args.iterations, device, best_acc, best_model_path)
                        # best_acc, eval_loss, eval_acc = eval_croma(net, cross_entropy, test_text_iter, args.test_iterations, device, best_acc, best_model_path)
                else:
                    if args.mode.startswith("i"):
                        best_acc, eval_loss, eval_acc = eval_croma_same(net, cross_entropy, val_vision_iter, args.iterations, device, best_acc, best_model_path)
                        # best_acc, eval_loss, eval_acc = eval_croma(net, cross_entropy, test_vision_iter, args.test_iterations, device, best_acc, best_model_path)
                    elif args.mode.startswith("a"):
                        best_acc, eval_loss, eval_acc = eval_croma_same(net, cross_entropy, val_audio_iter, args.iterations, device, best_acc, best_model_path)
                        # best_acc, eval_loss, eval_acc = eval_croma(net, cross_entropy, test_audio_iter, args.test_iterations, device, best_acc, best_model_path)
                    elif args.mode.startswith("t"):
                        best_acc, eval_loss, eval_acc = eval_croma_same(net, cross_entropy, val_text_iter, args.iterations, device, best_acc, best_model_path)
                        # best_acc, eval_loss, eval_acc = eval_croma(net, cross_entropy, test_text_iter, args.test_iterations, device, best_acc, best_model_path)
                eval_losses.append(eval_loss)
                eval_accs.append(eval_acc)

        else:
            if args.mode.endswith("i"):
                loss, acc = train_proto(args, model, prototypical_loss, vision_iter, device)
            elif args.mode.endswith("a"):
                loss, acc = train_proto(args, model, prototypical_loss, audio_iter, device)
            elif args.mode.endswith("t"):
                loss, acc = train_proto(args, model, prototypical_loss, text_iter, device)
            print('Train Loss: {:.3f}, Train Acc: {:.3f}'.format(loss, acc), flush=True)
            train_losses.append(loss); train_accs.append(acc)

            if iteration % args.val_epoch == 0:
                if args.mode.startswith("i"):
                    best_acc, eval_loss, eval_acc = eval_proto(args, model, prototypical_loss, val_vision_iter, device, best_acc, best_model_path)
                    # best_acc, eval_loss, eval_acc = eval_proto(args, model, prototypical_loss, test_vision_iter, device, best_acc, best_model_path)
                elif args.mode.startswith("a"):
                    best_acc, eval_loss, eval_acc = eval_proto(args, model, prototypical_loss, val_audio_iter, device, best_acc, best_model_path)
                    # best_acc, eval_loss, eval_acc = eval_proto(args, model, prototypical_loss, test_audio_iter, device, best_acc, best_model_path)
                elif args.mode.startswith("t"):
                    best_acc, eval_loss, eval_acc = eval_proto(args, model, prototypical_loss, val_text_iter, device, best_acc, best_model_path)
                    # best_acc, eval_loss, eval_acc = eval_proto(args, model, prototypical_loss, test_text_iter, device, best_acc, best_model_path)
                eval_losses.append(eval_loss)
                eval_accs.append(eval_acc)

    torch.save(model.state_dict(), last_model_path)
    plot(align_losses, train_losses, train_accs, eval_losses, eval_accs, args)
    

    

    


