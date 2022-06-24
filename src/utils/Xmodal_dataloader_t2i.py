import os
import torch
import random
import pickle
import numpy as np
import pandas as pd
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from preprocess import load_pkl

class AddGaussianNoise(object):
    def __init__(self, std = 0.03):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())*self.std
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={0}, std={self.std})'

class t2i_Dataset(Dataset):
    def __init__(self, task_mode, train_mode, image_dir, text_dir, idxs, val_labs,\
                 BS, n_way, k_shot, k_query, resize, transform = True, max_len = 64):
        '''
        :param mode:
        mode = 'visual'/'text' : output will be a single modality batch of tasks. i.e. X_supoort, X_query
        mode = 'align' : output will comprise even X_support... for the audio modality!
        '''
        self.all_idxs       = idxs
        self.imgs_dir       = image_dir
        self.text_dir       = text_dir
        self.all_classes    = list(self.all_idxs.keys())
        self.val_classes    = val_labs
        self.train_classes  = list(set(self.all_classes) - set(val_labs))
        print("Validation set labels : ", val_labs)
        print("Train set labels : ", self.train_classes)


        self.mode           = task_mode                     # train / align    (meta learning task procedure)
        self.train_mode     = train_mode                    # train / eval     (train/eval/test procedure)
        self.batch_size     = BS                            # Tasks per Batch
        self.n_way          = n_way                         # n_labels per task
        self.k_shot         = k_shot                        # n_sample per labels
        self.k_query        = k_query                       # n_sample per test labels
        self.resize         = resize                        # resizing dimension
        self.txt_max_len    = max_len
        self.transform      = transform
        self.transform_vis  = None
        if self.train_mode == 'train':                      # train / eval&test transform case 부리
            self.transform_vis  = transforms.Compose([transforms.RandomCrop(size=(self.resize, self.resize)),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.RandomVerticalFlip(),
                                                      AddGaussianNoise(std=0.025)])
        else:
            self.transform_vis = transforms.Compose([transforms.Resize(size=(self.resize, self.resize))])

        self.data           = []

        # Modality / purposewise batches for meta learning / aligment task!
        self.visual_support_batch = []
        self.visual_query_batch   = []
        self.text_support_batch   = []
        self.text_query_batch     = []

        self.support_label = []
        self.query_label   = []

        self.generate_batch(BS, self.train_mode)
        #print("query batch shape : ", np.array(self.visual_query_batch).shape)
        #print("support batch shape : ", np.array(self.visual_support_batch).shape)

        assert (self.mode == 'visual') or (self.mode == 'text') or (self.mode == 'align')

    def __len__(self):
        print("Batch size(Total number of tasks per epoch) = ", self.batch_size)
        print("Number of data per task                     = ", self.n_way*self.k_shot)
        return self.batch_size

    def collate_fn(self, indices):
        txt_data = np.zeros(shape = (self.k_shot+self.k_query, self.txt_max_len, 768))
        for idx, id in enumerate(indices):
            #print(self.text_dir+str(id)+'.npy')
            single_txt = np.load(self.text_dir+id+'.npy')
            if (len(single_txt) < self.txt_max_len):
                txt_data[idx] = np.pad(single_txt, ((0, self.txt_max_len-len(single_txt)),(0,0)), mode='constant', constant_values=0.0)
            else:
                txt_data[idx] = single_txt[:self.txt_max_len, :]
            #print(txt_data[idx][:][0])
        return txt_data[:self.k_shot], txt_data[self.k_shot:]

    def get_image(self, indices):
        img_data = np.zeros(shape = (self.k_query+self.k_shot, 3, self.resize, self.resize))
        for idx, id in enumerate(indices):
            single_img = np.load(self.imgs_dir+id+'.npy')
            single_img = torch.tensor(single_img)
            img_data[idx] = self.transform_vis(single_img).numpy()

        return img_data[:self.k_shot], img_data[self.k_shot:]


    def generate_batch(self, BS, mode):
        for i in range(BS):
            selected_clusters = None
            if mode == "train" or mode == "align" : 
                selected_clusters = np.random.choice(self.train_classes, self.n_way, False)     # No class overlap in batch
            elif mode == "eval" : 
                selected_clusters = np.random.choice(self.val_classes, self.n_way, False)
            print("label : ",selected_clusters)
            visual_support, visual_query = np.zeros(shape = (self.n_way*self.k_shot, 3, self.resize, self.resize)), \
                                           np.zeros(shape = (self.n_way*self.k_query, 3, self.resize, self.resize))
            text_support, text_query     = np.zeros(shape = (self.n_way*self.k_shot, self.txt_max_len, 768)), \
                                           np.zeros(shape = (self.n_way*self.k_query, self.txt_max_len, 768))
            support_idx = np.zeros(shape = (self.n_way*self.k_shot), dtype=int)
            query_idx   = np.zeros(shape = (self.n_way*self.k_query), dtype=int)
        
            for idx, lab in enumerate(selected_clusters):
                per_label_idxs = np.random.choice(self.all_idxs[lab], self.k_shot+self.k_query, mode == "eval")
                print(per_label_idxs)

                # Assign supports, queries
                visual_support[idx*self.k_shot : (idx+1)*self.k_shot], visual_query[idx*self.k_query : (idx+1)*self.k_query] =\
                self.get_image(per_label_idxs)
                text_support[idx*self.k_shot : (idx+1)*self.k_shot],   text_query[idx*self.k_query : (idx+1)*self.k_query]   = \
                self.collate_fn(per_label_idxs)

                # Labels to query/support (will be shuffled !)
                support_idx[idx*self.k_shot : (idx+1)*self.k_shot] = idx
                query_idx[idx*self.k_query : (idx+1)*self.k_query] = idx

            print(f"Shape : Vis_s = {visual_support.shape}, txt_q = {np.array(text_query).shape}")

            # Bind supports/queries
            supports = list(zip(support_idx, visual_support, text_support))
            queries  = list(zip(query_idx, visual_query, text_query))
            # Aligned shuffle
            random.shuffle(supports)
            random.shuffle(queries)

            #print(visual_support)
            s_idx, vis_s_data, txt_s_data = zip(*supports)
            q_idx, vis_q_data, txt_q_data = zip(*queries)
            print(s_idx, q_idx)

            self.support_label.append(torch.LongTensor(s_idx))
            self.query_label.append(torch.LongTensor(q_idx))

            self.visual_support_batch.append(torch.tensor(vis_s_data, dtype=torch.float32))
            self.visual_query_batch.append(torch.tensor(vis_q_data, dtype=torch.float32))
            self.text_support_batch.append(torch.tensor(txt_s_data, dtype=torch.float32))
            self.text_query_batch.append(torch.tensor(txt_q_data, dtype=torch.float32))

    def __getitem__(self, idx):
        if self.mode == "visual":
            return self.visual_support_batch[idx], self.visual_query_batch[idx],\
                   self.support_label[idx], self.query_label[idx]
        elif self.mode == "text":
            return self.text_support_batch[idx], self.text_query_batch[idx],\
                   self.support_label[idx], self.query_label[idx]
        elif self.mode == "align":
            return self.visual_support_batch[idx], self.visual_query_batch[idx],\
                   self.text_support_batch[idx], self.text_query_batch[idx],\
                   self.support_label[idx], self.query_label[idx]
        else:
            return ValueError("self.mode have to be either 'train', 'text' or 'align'.")



####################### Global Variables #######################
TOTAL_TASK_NUM = 1
TEXT_MAX_LEN   = 64
N_label_test   = 5
image_dir      = '../../data/recipe/new_imgs/'
text_dir       = '../../data/recipe/text_embs/'
train_idxs     = load_pkl('../../data/recipe/train_items.pkl')
val_idxs       = load_pkl('../../data/recipe/val_items.pkl')
total_idxs     = load_pkl('../../data/recipe/total_items.pkl')
#################################################################

if __name__ == '__main__':
    # Caution : n_ways in test dataset must be fixed to 5. 
    val_indices = np.random.choice(44, N_label_test, replace = False)
    #print(val_indices)
    # val_labs : validation step에서만 사용할 class
    # idxs     : preprocessing에서 걸러진 15792개 data의 {label : idx list들}
    Vision_train = t2i_Dataset(task_mode= 'visual', train_mode='train', 
                               image_dir=image_dir, text_dir=text_dir, idxs=total_idxs, val_labs = val_indices, 
                               BS = TOTAL_TASK_NUM, n_way = 3, k_shot = 5, k_query = 2, resize = 224, max_len = TEXT_MAX_LEN)
    Text_train   = t2i_Dataset(task_mode='text', train_mode='train', 
                               image_dir=image_dir, text_dir=text_dir, idxs=total_idxs, val_labs = val_indices,
                               BS = TOTAL_TASK_NUM, n_way = 3, k_shot = 5, k_query = 2, resize = 224, max_len = TEXT_MAX_LEN)
    Text_eval    = t2i_Dataset(task_mode='text', train_mode='eval', 
                               image_dir=image_dir, text_dir=text_dir, idxs=total_idxs, val_labs = val_indices,
                               BS = TOTAL_TASK_NUM, n_way = 3, k_shot = 5, k_query = 2, resize = 224, max_len = TEXT_MAX_LEN)
    # 이 경우 train mode와 상관없이 두 modality data가 동시에 나옴. 
    Test_align   = t2i_Dataset(task_mode='align', train_mode='train', 
                               image_dir=image_dir, text_dir=text_dir, idxs=total_idxs, val_labs = val_indices,
                               BS = TOTAL_TASK_NUM, n_way = 5, k_shot = 3, k_query = 2, resize = 224, max_len = TEXT_MAX_LEN)

   #dataset.info()
    print("Dataset length : ", len(Vision_train), '\n')

    text_dataloader   = DataLoader(Text_eval, batch_size=1, shuffle=True)
    align_dataloader  = DataLoader(Test_align, batch_size=1, shuffle=True)

    text_iter  = iter(text_dataloader)
    align_iter = iter(align_dataloader)
    
    for i in range(TOTAL_TASK_NUM):
        txt_s, txt_q, txt_s_lab, txt_q_lab = next(text_iter)
        print("Text dataset outputs : ", txt_s.shape, txt_q.shape, txt_s_lab, txt_q_lab)
        vis_s, vis_q, txt_s, txt_q, txt_s_lab, txt_q_lab = next(align_iter)
        print("Align dataloader output : ", vis_s.shape, vis_q.shape, txt_s.shape, txt_q.shape, txt_s_lab, txt_q_lab )
