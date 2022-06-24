import os
import json
import torch
import shutil
import random
import pickle
import numpy as np

from transformers import BertTokenizer, BertModel
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


image_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def transform_image(path):
    input_image = Image.open(path)
    input_tensor = image_preprocess(input_image)
    #print(input_tensor.shape)
    return input_tensor

def save_pkl(obj, path):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    min_data = 100                                                  # Default huddle
    data_dir = '../../data/recipe/'
    fid_to_label_path = os.path.join(data_dir, 'fid_to_label.pkl')  # label data path
    fid_to_text_path = os.path.join(data_dir, 'fid_to_text.pkl')    # text embedding vector path

    if not os.path.exists(fid_to_label_path) and not os.path.exists(fid_to_text_path):
        # Get metadata paths
        metadata_dir = os.path.join(data_dir, 'metadata')
        metadata_files = os.listdir(metadata_dir)
        metadata_files = [f for f in metadata_files if f.endswith('.json')]
        metadata_paths = [os.path.join(metadata_dir, f) for f in metadata_files]

        fid_to_label = {}
        fid_to_text = {}
        label_dict = {}
        fid_with_others = []                                        # detect foreign lang datas

        for path_i, p in enumerate(metadata_paths):
            fid = metadata_files[path_i][4:-5]
            with open(p, 'r', encoding="utf8") as inf:
                metadata = json.load(inf)
            cuisines = metadata['attributes']['cuisine']
            cuisine = '_'.join(sorted(cuisines))
            courses = metadata['attributes']['course']
            course = '_'.join(sorted(courses))
            label = cuisine+'_'+course
            fid_to_label[fid] = label
            fid_to_text[fid] = ', '.join(metadata['ingredientLines'])
            url_name = metadata['source']['sourceSiteUrl']
            ###########
            if url_name:
                if ('.de' in url_name) or ('.fr' in url_name):
                    fid_with_others.append(fid)

            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1
        print("Foreign language datas : ", len(fid_with_others))


        # Select 44 labels with > 100 datapoints and text with english
        new_label_dict = {}
        num_data = 0
        label_counts = []
        for label in label_dict:
            if label_dict[label] >= min_data:
                new_label_dict[label] = label_dict[label]
                num_data += label_dict[label]
                label_counts.append(label_dict[label])
        print('%d Classes' % len(new_label_dict))
        print('%d Datapoints' % num_data)
        mean_labels = np.mean(label_counts)
        std_labels = np.std(label_counts)
        print('original mean labels per class:  %d' % mean_labels)
        print('original stdev labels per class: %d' % std_labels)

        label_set = set(new_label_dict.keys())
        curr_label_counts = {}
        max_labels = mean_labels+std_labels
        print('clipping at %d' % max_labels)       # mean + 1stdev datapoint를 max로 설정.

        fids = list(set(sorted(list(fid_to_label.keys()))) - set(fid_with_others))  # fids with eng datas
        random.Random(0).shuffle(fids)
        #print(len(fids))               #27638 -> 27050
        #print(len(label_set))          #44

        # Filtering sparse datas ; assigning text data to labels
        new_fid_to_label = {}
        new_fid_to_text = {}
        # print("remaining : ", len(set(fids) - set(fid_with_others)))
        for fid in fids:
            label = fid_to_label[fid]
            # if label in label_set and new_label_dict[label] <= max_labels:
            if label in label_set and (label not in curr_label_counts or curr_label_counts[label] <= max_labels):
                if label in curr_label_counts:
                    curr_label_counts[label] += 1
                else:
                    curr_label_counts[label] = 1
                new_fid_to_label[fid] = label
                new_fid_to_text[fid] = fid_to_text[fid]
        new_label_counts = sorted(list(curr_label_counts.values()))

        #print(new_label_counts)
        print('now %d Classes' % len(new_label_counts))
        print('now %d Datapoints' % np.sum(new_label_counts))
        print('new mean labels per class:  %d' % np.mean(new_label_counts))
        print('new stdev labels per class: %d' % np.std(new_label_counts))

        save_pkl(new_fid_to_label, fid_to_label_path)
        save_pkl(new_fid_to_text, fid_to_text_path)

        # Convert text label -> integer labels
        label2int_path = os.path.join(data_dir, 'label2int.pkl')
        if not os.path.exists(label2int_path):
            fid_to_label = load_pkl(fid_to_label_path)
            labels = sorted(list(set(fid_to_label.values())))
            label2int = {label: i for i, label in enumerate(labels)}
            save_pkl(label2int, label2int_path)

        # Get text embeddign vectors with BERT
        text_emb_dir = os.path.join(data_dir, 'text_embs')
        if os.path.exists(text_emb_dir):
            shutil.rmtree(text_emb_dir)
        if not os.path.exists(text_emb_dir):
            os.makedirs(text_emb_dir)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            model = model.cuda()
            model.eval()
            fid_to_label = load_pkl(os.path.join(data_dir, 'fid_to_label.pkl'))
            fid_to_text = load_pkl(os.path.join(data_dir, 'fid_to_text.pkl'))
            with torch.no_grad():
                for fid in tqdm(fid_to_text):
                    text = fid_to_text[fid]
                    text = text.lower()
                    tokenized_text = tokenizer.tokenize(text)
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens[:512]])
                    tokens_tensor = tokens_tensor.cuda()
                    outputs = model(tokens_tensor)
                    encoded_layers = outputs[0][0] # (seq len, 768)
                    emb_path = os.path.join(text_emb_dir, '%s.npy' % fid)
                    if fid in fids:
                        np.save(emb_path, encoded_layers.cpu().numpy())

        # Transform(resize) and store images to .npy files
        new_img_dir = os.path.join(data_dir, 'new_imgs')
        if os.path.exists(new_img_dir):
            shutil.rmtree(new_img_dir)
        if True:
            # if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)
            fid_to_label = load_pkl(os.path.join(data_dir, 'fid_to_label.pkl'))
            #fids_ = sorted(list(fid_to_label.keys()))
            paths = [os.path.join(data_dir, 'images/', 'img%s.jpg' % fid) for fid in fids]
            with torch.no_grad():
                for path in tqdm(paths):
                    # print(path)
                    new_path = path.replace('img', '').replace('jpg', 'npy').replace('images', 'new_imgs')
                    x = transform_image(path)
                    x = x.numpy()
                    np.save(new_path, x)


def generate_item_split(fid_to_lab, lab_to_int, train_ratio = 0.85):
    itemdict = {}
    train_itemdict = {}
    val_itemdict   = {}

    itemdict = {}
    train_itemdict = {}
    val_itemdict   = {}

    for num_lab, txt_lab in fid_to_lab.items():
        if lab_to_int[txt_lab] not in itemdict.keys():          # if itemdict does not have key : 
            itemdict[lab_to_int[txt_lab]]       = []
            train_itemdict[lab_to_int[txt_lab]] = []
            val_itemdict[lab_to_int[txt_lab]]   = []

        itemdict[lab_to_int[txt_lab]].append(num_lab)

        # Generate train/val dataset idx
        temp_int = np.random.uniform()
        if temp_int < train_ratio:
            train_itemdict[lab_to_int[txt_lab]].append(num_lab)
        else:
            val_itemdict[lab_to_int[txt_lab]].append(num_lab)
    
    basedir = '../../data/recipe/'
    train_itempath = os.path.join(basedir, 'train_items.pkl')
    val_itempath   = os.path.join(basedir, 'val_items.pkl')
    total_itempath = os.path.join(basedir, 'total_items.pkl')

    if (not os.path.exists(train_itempath)) and (not os.path.exists(val_itempath)):
        save_pkl(train_itemdict, train_itempath)
        save_pkl(val_itemdict,   val_itempath  )
        save_pkl(itemdict, total_itempath)

    return itemdict, train_itemdict, val_itemdict


if __name__ == '__main__':
    main()

    fid_to_lab = load_pkl('../../data/recipe/fid_to_label.pkl')
    lab_to_int = load_pkl('../../data/recipe/label2int.pkl')

    total_, train_, val_ = generate_item_split(fid_to_lab, lab_to_int)
    count = 0
    for num in total_.keys():
        print(num, "train : ", len(train_[num]), "   val : ", len(val_[num]), "   total : ", len(total_[num]))
        count+=len(total_[num])

    val_items = load_pkl('../../data/recipe/total_items.pkl')
    print(count)

    # test code. if you do not have "train/val_items.pkl" in the base directory, this will return all trues.
    #for i in range(44):
    #    print(val_[i] == val_items[i], '\n')