## Cross-Modal Prototypical Networks for Few-Shot Classification
### Requirements
- python 3.6.13
- pytorch 1.7.0
- numpy 1.19.5
- matplotlib 3.1.2
### Directory structure
    ├── data
    │   ├── recipe
    │   │   └── total_items.pkl
    │   ├── clustered_ESC_CIFAR_TEST.pkl
    │   └── clustered_ESC_CIFAR_TRAIN.pkl
    ├── src
    │   ├── utils
    │   │   ├── preprocess_ESC_CIFAR.ipynb
    │   │   ├── preprocess.py
    │   │   ├── Xmodal_dataloader.py
    │   │   ├── Xmodal_dataloader_t2i.py
    │   │   └── Xmodal_dataloader_v2.py
    │   ├── data_utils.py
    │   ├── losses.py
    │   ├── models.py
    │   ├── tools.py
    │   ├── train_croma.sh
    │   ├── train_proto.py
    │   └── train.sh
    └── requirements.txt
### Train prototypical network
`$ ./train.sh`
### Test prototypical network
`$ python train_proto.py --mode a2i --train_mode test --load_checkpoint checkpoint.pt`
### Train CROMA
`$ ./train_croma.sh`
### Acknowledgment
Our code is based on the paper [**Cross-Modal Generalization: Learning in Low Resource Modalities via Meta-Alignment**](https://arxiv.org/abs/2012.02813)<br>
and the implementation [https://github.com/peter-yh-wu/xmodal](https://github.com/peter-yh-wu/xmodal)

