import os
import torch
import random
import pickle
import numpy as np
import pandas as pd
import torchvision
import torchaudio
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class AddGaussianNoise(object):
    def __init__(self, std = 0.05):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())*self.std
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={0}, std={self.std})'

class XmodalDataset(Dataset):
    def __init__(self, task_mode, train_mode, root_dir, BS, n_way, k_shot, k_query, resize, transform = True):
        '''
        :param mode:
        mode = 'visual'/'audio' : output will be a single modality batch of tasks. i.e. X_supoort, X_query
        mode = 'align' : output will comprise even X_support... for the audio modality!
        '''

        data_file           = open(root_dir, "rb")
        self.Xmodal_data    = pickle.load(data_file)        # Image-Audio paired dataset
        self.cluster_labels = list(self.Xmodal_data.keys()) # Cluster label names as list

        self.mode           = task_mode                     # train / align    (meta learning task procedure)
        self.train_mode     = train_mode                    # train / eval     (train/eval/test procedure)
        self.batch_size     = BS                            # Tasks per Batch
        self.n_way          = n_way                         # n_labels per task
        self.k_shot         = k_shot                        # n_sample per labels
        self.k_query        = k_query                       # n_sample per test labels
        self.resize         = resize                        # resizing dimension
        self.transform      = transform
        self.transform_aud  = None
        if self.train_mode == 'train':                      # train / eval&test transform case 부리
            self.transform_aud  = transforms.Compose([transforms.RandomCrop(size=(self.resize, self.resize)),
                                                      AddGaussianNoise(std=0.05)])
        else:
            self.transform_aud = transforms.Compose([transforms.CenterCrop(size=(self.resize, self.resize))])
#transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        self.transform_vis  = transforms.Compose([transforms.Resize(size=(224,224))])

        self.n_classes      = len(self.Xmodal_data)
        self.data           = []

        # Modality / purposewise batches for meta learning / aligment task!
        self.visual_support_batch  = []
        self.visual_query_batch    = []
        self.audio_support_batch   = []
        self.audio_query_batch     = []

        self.visual_support_label = []
        self.visual_query_label = []
        self.audio_support_label = []
        self.audio_query_label = []

        self.generate_batch(BS)
        #print("query batch shape : ", np.array(self.visual_query_batch).shape)
        #print("support batch shape : ", np.array(self.visual_support_batch).shape)

        assert (self.mode == 'visual') or (self.mode == 'audio') or (self.mode == 'align')

    def info(self):
        print("Keys :          ", self.Xmodal_data.keys())
        print("Internal keys : ", self.Xmodal_data['automobile'].keys())
        sample_img_data   = self.Xmodal_data['automobile']['cifar_data'][0]
        sample_aduio_data = self.Xmodal_data['automobile']['esc_data'][0]
        print("Image shape :   ", sample_img_data.shape)
        print("Audio shape :   ", sample_aduio_data.shape)

        return

    def generate_batch(self, BS):

        for i in range(BS):
            selected_clusters = np.random.choice(self.n_classes, self.n_way, False)     # No class overlap in batch
            # print("label : ",selected_clusters)
            visual_support, visual_query = np.zeros(shape = (self.n_way*self.k_shot, 3072)), np.zeros(shape = (self.n_way*self.k_query, 3072))
            audio_support, audio_query   = np.zeros(shape = (self.n_way*self.k_shot, 224, 501)), np.zeros(shape = (self.n_way*self.k_query, 224, 501))
            support_idx = np.zeros(shape = (self.n_way*self.k_shot), dtype=int)
            query_idx   = np.zeros(shape = (self.n_way*self.k_query), dtype=int)

            for idx, cls in enumerate(selected_clusters):
                # The whole visual-audio data for selected cluster label
                vis_data = np.array(self.Xmodal_data[self.cluster_labels[cls]]['cifar_data'])
                aud_data = np.squeeze(np.array(self.Xmodal_data[self.cluster_labels[cls]]['esc_data']))

                # Select indices for support, query set
                vis_len = len(vis_data)
                aud_len = len(aud_data)
                vis_indices = np.random.choice(vis_len, self.k_shot+self.k_query, False)
                aud_indices = np.random.choice(aud_len, self.k_shot+self.k_query, False)

                visual_support[idx*self.k_shot : (idx+1)*self.k_shot] = vis_data[vis_indices[:self.k_shot]]
                visual_query[idx*self.k_query : (idx+1)*self.k_query] = vis_data[vis_indices[self.k_shot:]]
                audio_support[idx*self.k_shot : (idx+1)*self.k_shot]  = aud_data[aud_indices[:self.k_shot]]
                audio_query[idx*self.k_query : (idx+1)*self.k_query]  = aud_data[aud_indices[self.k_shot:]]

                # Labels to support_idx
                support_idx[idx*self.k_shot : (idx+1)*self.k_shot] = idx
                query_idx[idx*self.k_query : (idx+1)*self.k_query] = idx

            # print(f"Shape : Vis_s = {visual_support.shape}, Aud_q = {np.array(audio_query).shape}")

            visual_support = list(zip(support_idx, visual_support))
            visual_query   = list(zip(query_idx, visual_query))
            audio_support  = list(zip(support_idx, audio_support))
            audio_query    = list(zip(query_idx, audio_query))

            # shuffle except the alignment case : -> 얘도 shuffle해도 상관없지 않나?
            # if (self.mode == 'visual') or (self.mode == 'audio'):
            random.shuffle(visual_support)
            random.shuffle(visual_query)
            random.shuffle(audio_support)
            random.shuffle(audio_query)

            #print(visual_support)
            vis_s_idx, vis_s_data = zip(*visual_support)
            vis_q_idx, vis_q_data = zip(*visual_query)
            aud_s_idx, aud_s_data = zip(*audio_support)
            aud_q_idx, aud_q_data = zip(*audio_query)

            #print(torch.tensor(vis_s_idx, dtype = torch.int8).shape, torch.tensor(vis_s_data, dtype=torch.float32).shape)
            # Append data/labels to indices
            self.visual_support_batch.append(torch.tensor(np.array(vis_s_data), dtype=torch.float32))
            self.visual_query_batch.append(torch.tensor(np.array(vis_q_data), dtype=torch.float32))
            self.audio_support_batch.append(torch.tensor(np.array(aud_s_data), dtype=torch.float32) / 100)
            self.audio_query_batch.append(torch.tensor(np.array(aud_q_data), dtype=torch.float32) / 100)    # divide by max(abs(min), max)

            self.visual_support_label.append(torch.LongTensor(vis_s_idx))
            self.visual_query_label.append(torch.LongTensor(vis_q_idx))
            self.audio_support_label.append(torch.LongTensor(aud_s_idx))
            self.audio_query_label.append(torch.LongTensor(aud_q_idx))


    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        audio_query_item = None
        audio_support_item = None
        visual_query_item = None
        visual_support_item = None

        # Apply transformations // Audio
        if self.transform:
            audio_query_item = torch.zeros((len(self.audio_query_batch[idx]), self.resize, self.resize))
            audio_support_item = torch.zeros((len(self.audio_support_batch[idx]), self.resize, self.resize))

            for (i, data) in enumerate(self.audio_query_batch[idx]):
                audio_query_item[i] = self.transform_aud(torch.unsqueeze(data, 0))
            for (i, data) in enumerate(self.audio_support_batch[idx]):
                audio_support_item[i] = self.transform_aud(torch.unsqueeze(data, 0))
        else:
            audio_query_item = self.audio_query_batch[idx]
            audio_support_item = self.audio_support_batch[idx]

        audio_query_item = torch.unsqueeze(audio_query_item, 1)
        audio_support_item = torch.unsqueeze(audio_support_item, 1)

        # Apply transformations // Visual
        N_items_query, N_items_support = len(self.visual_query_batch[idx]), len(self.visual_support_batch[idx])
        visual_query_item   = self.transform_vis(torch.reshape(self.visual_query_batch[idx], (N_items_query, 3, 32, 32))) / 255.0
        visual_support_item = self.transform_vis(torch.reshape(self.visual_support_batch[idx], (N_items_support, 3, 32, 32))) / 255.0

        # Task cases :
        if self.mode == 'visual':

            return visual_support_item, visual_query_item, \
                   self.visual_support_label[idx], self.visual_query_label[idx]

        elif self.mode == 'audio':

            return audio_support_item, audio_query_item, \
                   self.audio_support_label[idx], self.audio_query_label[idx]

        else:

            return self.visual_support_batch[idx], self.audio_support_batch[idx], \
                   self.visual_support_label[idx], self.visual_query_label[idx], \
                   self.visual_query_batch[idx], self.audio_query_batch[idx], \
                   self.audio_support_label[idx], self.audio_query_label[idx]


    def collate_fn(self):
        # Deprecated
        pass

    def _return_spectrogranm(self, audio_file):
        # Deprecated
        pass


if __name__ == '__main__':
    TOTAL_TASK_NUM = 8

    Vision_dataset = XmodalDataset(task_mode= 'visual', train_mode='train', root_dir='./clustered_ESC_CIFAR.pkl',
                                    BS = TOTAL_TASK_NUM, n_way = 3, k_shot = 5, k_query = 2, resize = 224)
    Audio_dataset  = XmodalDataset(task_mode='audio', train_mode='train', root_dir='./clustered_ESC_CIFAR.pkl',
                                    BS = TOTAL_TASK_NUM, n_way = 5, k_shot = 5, k_query = 2, resize = 224)
   #dataset.info()
    print("Dataset length : ", len(Vision_dataset), '\n')

    vision_dataloader = DataLoader(Vision_dataset, batch_size=1, shuffle=True)
    audio_dataloader  = DataLoader(Audio_dataset, batch_size=1, shuffle=True)

    vision_iter = iter(vision_dataloader)
    audio_iter  = iter(audio_dataloader)

    for i in range(TOTAL_TASK_NUM):
        vis_s, vis_q, vis_s_lab, vis_q_lab = next(vision_iter)
        aud_s, aud_q, aud_s_lab, aud_q_lab = next(audio_iter)
        #print(torch.max(aud_q), torch.min(aud_q), torch.mean(aud_q))

        print(vis_s.shape, vis_q.shape, vis_s_lab, vis_q_lab)
        print(aud_s.shape, aud_q.shape, aud_s_lab, aud_q_lab)
        #print(vis_s[0][0])

        #plt.imshow(np.transpose(vis_s[0][0].numpy()/255, (1,2,0))) ; plt.show()

'''
    for i in range(10):
        print(f'epoch {i}')
        Vision_dataset = XmodalDataset(mode='visual', root_dir='./clustered_ESC_CIFAR.pkl',
                                       BS=TOTAL_TASK_NUM, n_way=3, k_shot=5, k_query=2, resize=224)
        Audio_dataset = XmodalDataset(mode='audio', root_dir='./clustered_ESC_CIFAR.pkl',
                                      BS=TOTAL_TASK_NUM, n_way=5, k_shot=5, k_query=2, resize=224)

        vision_dataloader = DataLoader(Vision_dataset, batch_size=1, shuffle=True)
        audio_dataloader = DataLoader(Audio_dataset, batch_size=1, shuffle=True)
'''