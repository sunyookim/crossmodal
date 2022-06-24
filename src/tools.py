import pickle
import torch
from torch.autograd import Variable

def print_log(s, log_path):
    print(s)
    with open(log_path, 'a+') as ouf:
        ouf.write("%s\n" % s)


def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def get_optimizer(net, lr, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def Variable_(tensor, *args_, **kwargs):
#     if type(tensor) in (list, tuple):
#         return [Variable_(t, *args_, **kwargs) for t in tensor]
#     if isinstance(tensor, dict):
#         return {key: Variable_(v, *args_, **kwargs) for key, v in tensor.items()}
#     variable = Variable(tensor, *args_, **kwargs)
#     if torch.cuda.is_available():
#         variable = variable.cuda(cuda_device)
#     return variable

def to_device(tensor, device, squeeze=True):
    if type(tensor) in (list, tuple):
        return [to_device(t, device) for t in tensor]
    return tensor.to(device).squeeze(0) if squeeze else tensor.to(device)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
