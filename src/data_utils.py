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
from torchvision import transforms

os.chdir("/data/projects/Xmodal/src/")
def r_at_k(embs1, embs2):
    """embs1->embs2 Retrieval

    R@K is Recall@K (high is good). For median rank, low is good.
    Details described in https://arxiv.org/pdf/1411.2539.pdf.

    Args:
        embs1: embeddings, np array with shape (num_data, emb_dim)
            where num_data is either number of dev or test datapoints
        embs2: embeddings, np array with shape shape (num_data, emb_dim)

    Return:
        r1: R@1
        r5: R@5
        r10: R@10
        medr: median rank
    """
    ranks = np.zeros(len(embs1))
    mnorms = np.sqrt(np.sum(embs1**2,axis=1)[None]).T
    embs1 = embs1 / mnorms
    tnorms = np.sqrt(np.sum(embs2**2,axis=1)[None]).T
    embs2 = embs2 / tnorms

    for index in range(len(embs1)):
        im = embs1[index].reshape(1, embs1.shape[1])
        d = np.dot(im, embs2.T).flatten()
        inds = np.argsort(d)[::-1]
        rank = 1e20
        tmp = np.where(inds == index)[0][0]
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return (r1, r5, r10, r50, r100, medr)


def mk_retrieve(embs1, embs2, ys):
    """embs1->embs2 Retrieval

    Args:
        embs1: embeddings, np array with shape (num_data, emb_dim)
            where num_data is either number of dev or test datapoints
        embs2: embeddings, np array with shape shape (num_data, emb_dim)
    """
    all_ranks = []
    ranks = np.zeros(len(embs1))
    mnorms = np.sqrt(np.sum(embs1**2,axis=1)[None]).T
    embs1 = embs1 / mnorms
    tnorms = np.sqrt(np.sum(embs2**2,axis=1)[None]).T
    embs2 = embs2 / tnorms

    for index in range(len(embs1)):
        im = embs1[index:index+1, :]
        d = np.dot(im, embs2.T)
        inds = np.zeros(d.shape)
        inds[0] = np.argsort(d[0])[::-1]
        all_ranks.append(inds[0])
    return all_ranks


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


def collate_recipe(batch):
    '''collate function for recipe task'''
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    max_len_2 = 0
    for (_, _, x2, _) in batch:
        max_len_2 = max(len(x2), max_len_2)
    for (x1, y1, x2, y2) in batch:
        x1s.append(x1)
        y1s.append(y1)
        x2 = np.pad(x2, ((0,max_len_2-len(x2)), (0,0)), mode='constant', constant_values=0.0)
        x2s.append(x2)
        y2s.append(y2)
    return torch.tensor(x1s), torch.tensor(y1s), torch.tensor(x2s), torch.tensor(y2s)


def collate_recipe_verbose(batch):
    '''collate function for recipe task'''
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    idxs = []
    max_len_2 = 0
    for (_, _, x2, _, _) in batch:
        max_len_2 = max(len(x2), max_len_2)
    for (x1, y1, x2, y2, idx) in batch:
        x1s.append(x1)
        y1s.append(y1)
        x2 = np.pad(x2, ((0,max_len_2-len(x2)), (0,0)), mode='constant', constant_values=0.0)
        x2s.append(x2)
        y2s.append(y2)
        idxs.append(idx)
    return torch.tensor(x1s), torch.tensor(y1s), torch.tensor(x2s), torch.tensor(y2s), torch.tensor(idxs)


def collate_text(batch):
    '''collate function for recipe task'''
    x2s = []
    y2s = []
    max_len_2 = 0
    for (x2, _) in batch:
        max_len_2 = max(len(x2), max_len_2)
    for (x2, y2) in batch:
        x2 = np.pad(x2, ((0,max_len_2-len(x2)), (0,0)), mode='constant', constant_values=0.0)
        x2s.append(x2)
        y2s.append(y2)
    return torch.tensor(x2s), torch.tensor(y2s)


class FewShotTwo(Dataset):
    '''
    Dataset for K-shot N-way classification for 2 modalities
    '''
    def __init__(self, x1s, x2s, parent=None, verbose=False):
        self.x1s = x1s
        self.x2s = x2s
        self.parent = parent
        self.verbose = verbose
        if verbose: # assumes x1s & x2s are paths
            self.x1_ids = [x['x'] for x in x1s]
            self.x2_ids = [x['x'] for x in x2s]

    def __len__(self):
        return len(self.x1s)

    def __getitem__(self, idx):
        x1 = self.x1s[idx]['x']
        orig_x2 = self.x2s[idx]['x']
        if self.parent.transform_x1 is not None:
            x1 = self.parent.transform_x1(x1)
        if self.parent.transform_x2 is not None:
            x2 = self.parent.transform_x2(orig_x2)
        else:
            x2 = orig_x2
        y1 = self.x1s[idx]['y']
        y2 = self.x2s[idx]['base_idx']
        # if self.verbose:
        #     print('x: ', orig_x2, ', y: ', y1, ', base_idx: ', y2)
        if self.parent.transform_y1 is not None:
            y1 = self.parent.transform_y1(y1)
        if self.parent.transform_y2 is not None:
            y2 = self.parent.transform_y2(y2)
        if self.verbose:
            return x1, y1, x2, y2, idx
        else:
            return x1, y1, x2, y2


class BothDataset(Dataset):
    def __init__(self, grouped_x1s, grouped_x2s, align_idxs=None, \
            transform_x1=None, transform_x2=None, \
            transform_y1=None, transform_y2=None, is_align=True):
        '''
        grouped_xis organized by shared class label
        assumes grouped x1s & grouped x2s are paired
        '''
        if align_idxs is None:
            non_align_idxs = None
        else:
            non_align_idxs = [[] for _ in grouped_x1s]
            for list_i, l in enumerate(grouped_x1s):
                align_idxs_set = set(list(align_idxs[list_i]))
                for i in range(len(l)):
                    if i not in align_idxs_set:
                        non_align_idxs[list_i].append(i)
        self.transform_x1 = transform_x1
        self.transform_x2 = transform_x2
        self.transform_y1 = transform_y1
        self.transform_y2 = transform_y2

        if align_idxs is not None:
            if is_align:
                grouped_x1s = [[l[i] for i in idxs] for idxs, l in zip(align_idxs, grouped_x1s)]
                grouped_x2s = [[l[i] for i in idxs] for idxs, l in zip(align_idxs, grouped_x2s)]
            else:
                grouped_x1s = [[l[i] for i in idxs] for idxs, l in zip(non_align_idxs, grouped_x1s)]
                grouped_x2s = [[l[i] for i in idxs] for idxs, l in zip(non_align_idxs, grouped_x2s)]

        self.x1s = [item for sublist in grouped_x1s for item in sublist]
        self.x2s = [item for sublist in grouped_x2s for item in sublist]
        grouped_ys = [[i]*len(l) for i, l in enumerate(grouped_x1s)]
        self.ys = [item for sublist in grouped_ys for item in sublist]

    def __len__(self):
        return len(self.x1s)
    
    def __getitem__(self, index):
        x1 = self.x1s[index]
        y1 = self.ys[index]
        x2 = self.x2s[index]
        y2 = self.ys[index]
        if self.transform_x1 is not None:
            x1 = self.transform_x1(x1)
        if self.transform_x2 is not None:
            x2 = self.transform_x2(x2)
        if self.transform_y1 is not None:
            y1 = self.transform_y1(y1)
        if self.transform_y2 is not None:
            y2 = self.transform_y2(y2)
        return x1, y1, x2, y2


class AbstractMetaTwo(object):
    '''
    Note only supports get_random_task and not get_task_split
    '''
    def __init__(self, grouped_x1s, grouped_x2s, align_idxs, \
            transform_x1=None, transform_x2=None, \
            transform_y1=None, transform_y2=None):
        '''
        grouped_xis organized by shared class label
        assumes grouped x1s & grouped x2s are paired
        '''
        self.grouped_x1s = grouped_x1s
        print(len(self.grouped_x1s))
        self.grouped_x2s = grouped_x2s
        self.align_idxs = align_idxs
        if self.align_idxs is None:
            self.non_align_idxs = None
        else:
            self.non_align_idxs = [[] for _ in self.grouped_x1s]
            for list_i, l in enumerate(self.grouped_x1s):
                align_idxs_set = set(list(self.align_idxs[list_i]))
                for i in range(len(l)):
                    if i not in align_idxs_set:
                        self.non_align_idxs[list_i].append(i)
        self.transform_x1 = transform_x1
        self.transform_x2 = transform_x2
        self.transform_y1 = transform_y1
        self.transform_y2 = transform_y2

    def __len__(self):
        return len(self.grouped_x1s)

    def __getitem__(self, idx):
        return self.grouped_x1s[idx], self.grouped_x2s[idx]

    def get_dummy_task(self, N=5, K=1, is_align=True):
        '''for testing only'''
        train_task, __ = self.get_dummy_task_split(N, train_K=K, test_K=0, is_align=is_align)
        return train_task
    
    def get_dummy_task_split(self, N=5, train_K=4, test_K=2, is_align=True, verbose=False):
        '''for testing only'''
        train_samples1 = []
        test_samples1 = []
        train_samples2 = []
        test_samples2 = []
        character_indices = [i for i in range(N)] # what makes it dummy
        for base_idx, idx in enumerate(character_indices):
            x1s = self.grouped_x1s[idx]
            x2s = self.grouped_x2s[idx]
            if self.align_idxs is None:
                curr_idxs1 = np.random.choice(len(x1s), train_K + test_K, replace=False)
            elif is_align:
                curr_idxs1 = np.random.choice(self.align_idxs[idx], train_K + test_K, replace=False)
            else:
                curr_idxs1 = np.random.choice(self.non_align_idxs[idx], train_K + test_K, replace=False)
            for i, x1_idx in enumerate(curr_idxs1):
                x1 = x1s[x1_idx]
                x2 = x2s[x1_idx]
                new_x1 = {'x':x1, 'y':idx, 'base_idx':base_idx}
                new_x2 = {'x':x2, 'y':idx, 'base_idx':base_idx}
                if i < train_K:
                    train_samples1.append(new_x1)
                    train_samples2.append(new_x2)
                else:
                    test_samples1.append(new_x1)
                    test_samples2.append(new_x2)
        train_task = FewShotTwo(train_samples1, train_samples2, parent=self)
        test_task = FewShotTwo(test_samples1, test_samples2, parent=self, verbose=verbose)
        return train_task, test_task

    def get_random_task(self, N=5, K=1, is_align=True):
        train_task, test_task = self.get_random_task_split(N=N, train_K=K, test_K=0, is_align=is_align)
        return train_task

    def get_random_task_split(self, N=5, train_K=4, test_K=2, is_align=True, verbose=False):
        train_samples1 = []
        test_samples1 = []
        train_samples2 = []
        test_samples2 = []
        character_indices = np.random.choice(len(self), N, replace=False)
        for base_idx, idx in enumerate(character_indices):
            x1s = self.grouped_x1s[idx]
            x2s = self.grouped_x2s[idx]
            if self.align_idxs is None:
                curr_idxs1 = np.random.choice(len(x1s), train_K + test_K, replace=False)
            elif is_align:
                curr_idxs1 = np.random.choice(self.align_idxs[idx], train_K + test_K, replace=False)
            else:
                curr_idxs1 = np.random.choice(self.non_align_idxs[idx], train_K + test_K, replace=False)
            for i, x1_idx in enumerate(curr_idxs1):
                x1 = x1s[x1_idx]
                x2 = x2s[x1_idx]
                new_x1 = {'x':x1, 'y':idx, 'base_idx':base_idx}
                new_x2 = {'x':x2, 'y':idx, 'base_idx':base_idx}
                if i < train_K:
                    train_samples1.append(new_x1)
                    train_samples2.append(new_x2)
                else:
                    test_samples1.append(new_x1)
                    test_samples2.append(new_x2)
        train_task = FewShotTwo(train_samples1, train_samples2, parent=self)
        if verbose:
            test_task = FewShotTwo(test_samples1, test_samples2, parent=self, verbose=verbose)
            return train_task, test_task, test_task.x1_ids, test_task.x2_ids
        else:
            test_task = FewShotTwo(test_samples1, test_samples2, parent=self, verbose=verbose)
            return train_task, test_task

    def get_task_split(self, character_indices, all_curr_idxs,
                        new_train_idxs,
                        new_test_idxs,
                        train_K=1, test_K=10, verbose=False):
        train_samples1 = []
        test_samples1 = []
        train_samples2 = []
        test_samples2 = []
        for base_idx, idx in enumerate(character_indices):
            x1s = self.grouped_x1s[idx]
            x2s = self.grouped_x2s[idx]
            
            curr_idxs = all_curr_idxs[base_idx]
            
            for i, x1_idx in enumerate(curr_idxs):
                x1 = x1s[x1_idx]
                x2 = x2s[x1_idx]
                new_x1 = {'x':x1, 'y':idx, 'base_idx':base_idx}
                new_x2 = {'x':x2, 'y':idx, 'base_idx':base_idx}
                if i < train_K:
                    train_samples1.append(new_x1)
                    train_samples2.append(new_x2)
                elif i < train_K+test_K:
                    test_samples1.append(new_x1)
                    test_samples2.append(new_x2)
        train_samples1 = [train_samples1[i] for i in new_train_idxs]
        train_samples2 = [train_samples2[i] for i in new_train_idxs]
        test_samples1 = [test_samples1[i] for i in new_test_idxs]
        test_samples2 = [test_samples2[i] for i in new_test_idxs]
        train_task = FewShotTwo(train_samples1, train_samples2, parent=self)
        test_task = FewShotTwo(test_samples1, test_samples2, parent=self, verbose=verbose)
        return train_task, test_task


def get_recipe_data():
    data_dir = '../data/recipe'
    fid_to_label = load_pkl(os.path.join(data_dir, 'fid_to_label.pkl'))
    fid_to_text = load_pkl(os.path.join(data_dir, 'fid_to_text.pkl'))
    fids = sorted(list(fid_to_label.keys()))
    label_list = sorted(list(set(fid_to_label.values())))
    label_to_int = {}
    for i, label in enumerate(label_list):
        label_to_int[label] = i
    paths1 = [os.path.join(data_dir, 'new_imgs', '%s.npy' % fid) for fid in fids]
    targets = [label_to_int[fid_to_label[fid]] for fid in fids]
    num_labels = len(np.unique(targets))
    paths2 = [os.path.join(data_dir, 'text_embs', '%s.npy' % fid) for fid in fids]
    return num_labels, paths1, paths2, targets


class MetaFolderTwo(AbstractMetaTwo):
    '''dataset for recipe task'''
    def __init__(self, *args, **kwargs):
        num_labels, paths1, paths2, targets = get_recipe_data()
        grouped_x1s = [[] for _ in range(num_labels)]
        for path, target in zip(paths1, targets):
            grouped_x1s[target].append(path)
        grouped_x2s = [[] for _ in range(num_labels)]
        for path, target in zip(paths2, targets):
            grouped_x2s[target].append(path)
        AbstractMetaTwo.__init__(self, grouped_x1s, grouped_x2s, None, *args, **kwargs)


def split_meta_both(all_meta, train=0.8, validation=0.1, test=0.1, seed=0, batch_size=64, num_workers=4, mk_super=False, dept=False, verbose=False, collate_fn=None):
    '''
    NOTE: validation & test args deprecated
    '''
    idx_dir = '../data/recipe/idxs'
    split_idxs_path = os.path.join(idx_dir, 'split_idxs_%d.npy' % seed)

    indices = np.load(split_idxs_path)
    if train >= 0.1:
        n_train = int(train * len(all_meta))
    else:
        n_train = int(0.1 * len(all_meta))
    n_val = int((len(all_meta) - n_train)/2)
    n_test = len(all_meta) - n_train - n_val
    # n_val = int(validation * len(all_meta))
    # n_test = int(test * len(all_meta))

    all_train = [all_meta[i] for i in indices[:n_train]]
    if train < 0.1:
        new_train_frac = train/0.1
        all_train = [l[:int(len(l)*new_train_frac)] for l in all_train]
    all_val = [all_meta[i] for i in indices[-(n_val+n_test):-n_test]]
    all_test = [all_meta[i] for i in indices[-n_test:]]
    print('test', indices[-n_test:])

    train_x1s = [e[0] for e in all_train]
    val_x1s = [e[0] for e in all_val]
    test_x1s = [e[0] for e in all_test]
    train_x2s = [e[1] for e in all_train]
    val_x2s = [e[1] for e in all_val]
    test_x2s = [e[1] for e in all_test]

    train_align_idxs = None
    if not dept:
        if os.path.exists(os.path.join(idx_dir, 'train_align_idxs_%d.npy' % seed)):
            train_align_idxs = load_pkl(os.path.join(idx_dir, 'train_align_idxs_%d.npy' % seed))
        else:
            train_align_idxs = [list(np.random.choice(len(l), int(len(l)/2), replace=False)) for l in train_x1s]
            save_pkl(train_align_idxs, os.path.join(idx_dir, 'train_align_idxs_%d.npy' % seed))

        if verbose:
            num_align = sum([len(l) for l in train_align_idxs])
            num_train = sum([len(l) for l in train_x1s])
            print('num_train_clf: ', num_train-num_align)

    train = AbstractMetaTwo(train_x1s, train_x2s, align_idxs=train_align_idxs,
                transform_x1=all_meta.transform_x1, transform_x2=all_meta.transform_x2, 
                transform_y1=all_meta.transform_y1, transform_y2=all_meta.transform_y2)
    val = AbstractMetaTwo(val_x1s, val_x2s, align_idxs=None,
                transform_x1=all_meta.transform_x1, transform_x2=all_meta.transform_x2, 
                transform_y1=all_meta.transform_y1, transform_y2=all_meta.transform_y2)
    test = AbstractMetaTwo(test_x1s, test_x2s, align_idxs=None,
                transform_x1=all_meta.transform_x1, transform_x2=all_meta.transform_x2, 
                transform_y1=all_meta.transform_y1, transform_y2=all_meta.transform_y2)

    if mk_super:
        dataset = BothDataset(train_x1s, train_x2s, align_idxs=train_align_idxs, \
            transform_x1=all_meta.transform_x1, transform_x2=all_meta.transform_x2, \
            transform_y1=all_meta.transform_y1, transform_y2=all_meta.transform_y2, \
            is_align=True)
        super_train = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)

        return train, val, test, super_train
    else:
        return train, val, test


class ImageDataset(Dataset):
    def __init__(self, phase):
        num_labels, paths1, paths2, targets = get_recipe_data()
        self.num_classes = num_labels
        self.paths1 = paths1
        self.targets = targets
        idxs = [i for i in range(len(self.paths1))]
        random.Random(0).shuffle(idxs)
        self.paths1 = [self.paths1[i] for i in idxs]
        self.targets = [self.targets[i] for i in idxs]
        num_train = int(0.9*len(self.targets))
        if phase == 'train':
            self.paths1 = self.paths1[:num_train]
            self.targets = self.targets[:num_train]
        else:
            self.paths1 = self.paths1[num_train:]
            self.targets = self.targets[num_train:]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return np.load(self.paths1[index]), self.targets[index]


class TextDataset(Dataset):
    def __init__(self, phase):
        num_labels, paths1, paths2, targets = get_recipe_data()
        self.num_classes = num_labels
        self.paths2 = paths2
        self.targets = targets
        idxs = [i for i in range(len(self.paths2))]
        random.Random(0).shuffle(idxs)
        self.paths2 = [self.paths2[i] for i in idxs]
        self.targets = [self.targets[i] for i in idxs]
        num_train = int(0.9*len(self.targets))
        if phase == 'train':
            self.paths2 = self.paths2[:num_train]
            self.targets = self.targets[:num_train]
        else:
            self.paths2 = self.paths2[num_train:]
            self.targets = self.targets[num_train:]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return np.load(self.paths2[index]), self.targets[index]


def mk_dataloader_1(phase, mode='default', batch_size=64, shuffle=True, num_workers=4):
    dataset = ImageDataset(phase)
    dataloader = DataLoader(dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers)
    return dataset.num_classes, dataloader

def mk_dataloader_2(phase, mode='default', batch_size=64, shuffle=True, num_workers=4):
    dataset = TextDataset(phase)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_text,
        shuffle=shuffle, num_workers=num_workers)
    return dataset.num_classes, dataloader


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
    def __init__(self, task_mode, train_mode, BS, n_way, k_shot, k_query, resize, transform = True):
        '''
        :param mode:
        mode = 'visual'/'audio' : output will be a single modality batch of tasks. i.e. X_supoort, X_query
        mode = 'align' : output will comprise even X_support... for the audio modality!
        '''

        # data_file           = open(root_dir, "rb")
        if train_mode in ["train", "val"]:
            data_file = open("/data/projects/Xmodal/data/clustered_ESC_CIFAR_TRAIN.pkl", "rb")
        else:
            data_file = open("/data/projects/Xmodal/data/clustered_ESC_CIFAR_TEST.pkl","rb")
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



    def collate_fn(self):
        # Deprecated
        pass

    def _return_spectrogranm(self, audio_file):
        # Deprecated
        pass
