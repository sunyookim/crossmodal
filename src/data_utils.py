import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
import sys
import random
import pickle
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


#os.chdir("/data/projects/Xmodal/src/")

image_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def transform_image(path):
    input_image = Image.open(path)
    input_tensor = image_preprocess(input_image)
    return input_tensor


def np_transform(path):
    return np.load(path)


def save_pkl(obj, path):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class AddGaussianNoise(object):
    def __init__(self, std = 0.05):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())*self.std
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={0}, std={self.std})'

class XmodalDataset(Dataset):
    def __init__(self, task_mode, train_mode, BS, n_way, k_shot, k_query, resize, transform = True):
        '''
        :param mode:
        mode = 'visual'/'audio' : output will be a single modality batch of tasks. i.e. X_supoort, X_query
        mode = 'align' : output will comprise even X_support... for the audio modality!
        '''

        # data_file           = open(root_dir, "rb")
        if train_mode in ["train", "val"]:
            data_file = open("../data/clustered_ESC_CIFAR_TRAIN.pkl", "rb")
        else:
            data_file = open("../data/clustered_ESC_CIFAR_TEST.pkl","rb")
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
            self.transform_aud  = transforms.Compose([transforms.Resize(size=(self.resize, self.resize)),
                                                      AddGaussianNoise(std=0.05)])
        else:
            self.transform_aud = transforms.Compose([transforms.Resize(size=(self.resize, self.resize))])
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
            random.Random(0).shuffle(visual_support)
            random.shuffle(visual_query)
            random.Random(0).shuffle(audio_support)
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

            return visual_support_item, visual_query_item, \
                   self.visual_support_label[idx], self.visual_query_label[idx], \
                   audio_support_item, audio_query_item, \
                   self.audio_support_label[idx], self.audio_query_label[idx]



class t2i_Dataset(Dataset):
    def __init__(self, task_mode, train_mode, BS, n_way, k_shot, k_query, resize, transform = True, max_len = 64):
        '''
        :param mode:
        mode = 'visual'/'text' : output will be a single modality batch of tasks. i.e. X_supoort, X_query
        mode = 'align' : output will comprise even X_support... for the audio modality!
        '''
        self.all_idxs       = load_pkl('../data/recipe/total_items.pkl')
        self.imgs_dir       = '../data/recipe/new_imgs/'
        self.text_dir       = '../data/recipe/text_embs/'
        self.all_classes    = list(self.all_idxs.keys())
        val_labs = np.random.choice(44, n_way, replace = False)
        self.val_classes    = val_labs
        self.train_classes  = list(set(self.all_classes) - set(val_labs))
        # print("Validation set labels : ", val_labs)
        # print("Train set labels : ", self.train_classes)


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
        # print("Batch size(Total number of tasks per epoch) = ", self.batch_size)
        # print("Number of data per task                     = ", self.n_way*self.k_shot)
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
            if mode == "train" or mode == "val" or mode == "align" : 
                selected_clusters = np.random.choice(self.train_classes, self.n_way, False)     # No class overlap in batch
            elif mode == "test" : 
                selected_clusters = np.random.choice(self.val_classes, self.n_way, False)
            # print("label : ",selected_clusters)
            visual_support, visual_query = np.zeros(shape = (self.n_way*self.k_shot, 3, self.resize, self.resize)), \
                                           np.zeros(shape = (self.n_way*self.k_query, 3, self.resize, self.resize))
            text_support, text_query     = np.zeros(shape = (self.n_way*self.k_shot, self.txt_max_len, 768)), \
                                           np.zeros(shape = (self.n_way*self.k_query, self.txt_max_len, 768))
            support_idx = np.zeros(shape = (self.n_way*self.k_shot), dtype=int)
            query_idx   = np.zeros(shape = (self.n_way*self.k_query), dtype=int)
        
            for idx, lab in enumerate(selected_clusters):
                per_label_idxs = np.random.choice(self.all_idxs[lab], self.k_shot+self.k_query, mode == "test")
                # print(per_label_idxs)

                # Assign supports, queries
                visual_support[idx*self.k_shot : (idx+1)*self.k_shot], visual_query[idx*self.k_query : (idx+1)*self.k_query] =\
                self.get_image(per_label_idxs)
                text_support[idx*self.k_shot : (idx+1)*self.k_shot],   text_query[idx*self.k_query : (idx+1)*self.k_query]   = \
                self.collate_fn(per_label_idxs)

                # Labels to query/support (will be shuffled !)
                support_idx[idx*self.k_shot : (idx+1)*self.k_shot] = idx
                query_idx[idx*self.k_query : (idx+1)*self.k_query] = idx

            # print(f"Shape : Vis_s = {visual_support.shape}, txt_q = {np.array(text_query).shape}")

            # Bind supports/queries
            supports = list(zip(support_idx, visual_support, text_support))
            queries  = list(zip(query_idx, visual_query, text_query))
            # Aligned shuffle
            random.shuffle(supports)
            random.shuffle(queries)

            #print(visual_support)
            s_idx, vis_s_data, txt_s_data = zip(*supports)
            q_idx, vis_q_data, txt_q_data = zip(*queries)
            # print(s_idx, q_idx)

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
                   self.support_label[idx], self.query_label[idx],\
                   self.text_support_batch[idx], self.text_query_batch[idx],\
                   self.support_label[idx], self.query_label[idx]
        else:
            return ValueError("self.mode have to be either 'train', 'text' or 'align'.")
