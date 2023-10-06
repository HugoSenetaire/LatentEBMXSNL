import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfm
import torch.utils.data as data_utils
import torch
import numpy as np
from scipy.io import loadmat
import os
import wget

class dynamicBinarizationDataset(data_utils.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_size = len(dataset)
    
    def __getitem__(self, index):
        x,y = self.dataset.__getitem__(index)
        x = torch.bernoulli(x)
        return x,y
    
    def __len__(self):
        return self.dataset_size


def load_omniglot(args, n_validation=1345, **kwargs):
    # set args
    # args.input_size = [1, 28, 28]
    # args.input_type = 'binary'
    args["dynamic_binarization"] = True

    # start processing
    def reshape_data(data):
        return data.reshape((1, 28, 28))
    
    transforms = tfm.Compose([tfm.Resize(28,), tfm.ToTensor(), reshape_data])
    # wget.download(url="https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat", out=os.path.join(args["root"],"chardata.mat"))
    # omni_raw = np.loadtxt(os.path.join(args["root"],"chardata.mat"))

    # train and test data
    # train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    train = tv.datasets.Omniglot(root=args["root"], download=True, transform=transforms)
    test =  tv.datasets.Omniglot(root=args["root"], background=False, download=True, transform=transforms)
    val = tv.datasets.Omniglot(root=args["root"], background=False, download=True, transform=transforms)
    # x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    # np.random.shuffle(train_data)

    # set train and validation data
    # x_train = train_data[:-n_validation]
    # x_val = train_data[-n_validation:]



    # pytorch data loader
    train = dynamicBinarizationDataset(train)
    train_loader = data_utils.DataLoader(train, batch_size=args["batch_size"], shuffle=True, **kwargs)

    validation = val
    val_loader = data_utils.DataLoader(validation, batch_size=args["batch_size_val"], shuffle=False, **kwargs)

    test = test
    test_loader = data_utils.DataLoader(test, batch_size=args["batch_size_val"], shuffle=True, **kwargs)
    print("Length of train_loader: ", len(train_loader), "Length of train_loader dataset: ", len(train_loader.dataset))
    print("Length of val_loader: ", len(val_loader), "Length of val_loader dataset: ", len(val_loader.dataset))
    print("Length of test_loader: ", len(test_loader), "Length of test_loader dataset: ", len(test_loader.dataset))

    return train_loader, val_loader, test_loader, args



# ======================================================================================================================
def load_caltech101silhouettes(args, **kwargs):
    # set args
    args["dynamic_binarization"] = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 1, 28, 28))
    wget.download(url="https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat", out=os.path.join(args["root"],"caltech101_silhouettes_28_split1.mat"))
    caltech_raw = loadmat(os.path.join(args["root"],"caltech101_silhouettes_28_split1.mat"))
    # train, validation and test data
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args["batch_size"], shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args["batch_size_val"], shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args["batch_size_val"], shuffle=True, **kwargs)
    print("Length of train_loader: ", len(train_loader), "Length of train_loader dataset: ", len(train_loader.dataset))
    print("Length of val_loader: ", len(val_loader), "Length of val_loader dataset: ", len(val_loader.dataset))
    print("Length of test_loader: ", len(test_loader), "Length of test_loader dataset: ", len(test_loader.dataset))
    return train_loader, val_loader, test_loader, args



def get_dataset_and_loader(cfg, device):
    dataset = cfg.dataset.dataset_name
    if cfg.dataset.root_dataset is None:
        cfg.dataset.root_dataset = os.path.join(cfg.machine.root, "datasets")
    if dataset.startswith('SVHN'):
        transform = tfm.Compose([tfm.Resize(cfg.dataset.img_size), tfm.ToTensor(), tfm.Normalize(([0.5]*3), ([0.5]*3)),])
        data_train = t.stack([x[0] for x in tv.datasets.SVHN(download=True, root='{}/svhn'.format(cfg.dataset.root_dataset), transform=transform)]).to(device)
        data_test = t.stack([x[0] for x in tv.datasets.SVHN(split='test', download=True, root='{}/svhn'.format(cfg.dataset.root_dataset), transform=transform)]).to(device)
        data_valid= t.stack([x[0] for x in tv.datasets.SVHN(split='test', download=True, root='{}/svhn'.format(cfg.dataset.root_dataset), transform=transform)]).to(device)
    elif dataset == "MNIST":
        transform = tfm.Compose([tfm.Resize(cfg.dataset.img_size), tfm.ToTensor(), tfm.Normalize((0.5), (0.5),)])
        data_train = t.stack([x[0] for x in tv.datasets.MNIST(download=True, root='{}/mnist'.format(cfg.dataset.root_dataset), transform=transform)]).to(device)
        data_test = t.stack([x[0] for x in tv.datasets.MNIST(train=False, download=True, root='{}/mnist'.format(cfg.dataset.root_dataset), transform=transform)]).to(device)
        data_valid = data_test
    elif dataset == "BINARY_MNIST":
        xtrain = np.loadtxt('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat',dtype=np.float32).reshape(-1,1,28, 28,order='C')
        xvalid = np.loadtxt('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat',dtype=np.float32).reshape(-1,1,28, 28,order='C')
        xtest = np.loadtxt('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat',dtype=np.float32).reshape(-1,1,28, 28,order='C')
        data_train = t.from_numpy(xtrain).to(device)
        data_valid = t.from_numpy(xvalid).to(device)
        data_test = t.from_numpy(xtest).to(device)
    elif dataset == "OMNIGLOT":
        data_train, data_valid, data_test, cfg = load_omniglot(cfg)
        
    elif dataset == "CALTECH101SILHOUETTES":
        data_train, data_valid, data_test, cfg = load_caltech101silhouettes(cfg)
    else :
        raise NotImplementedError("Unknown dataset: {}".format(dataset))
    return data_train, data_valid, data_test



