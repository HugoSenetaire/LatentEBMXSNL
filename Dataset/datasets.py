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


def load_omniglot(cfg, n_validation=1345, **kwargs):
    # set args
    cfg.dataset.dynamic_binarization = True

    # start processing
    def reshape_data(data):
        return data.reshape((1, 28, 28))
    
    transforms = tfm.Compose([tfm.Resize(28,), tfm.ToTensor(), reshape_data])

    train = tv.datasets.Omniglot(root=cfg.dataset.root, download=True, transform=transforms)
    test =  tv.datasets.Omniglot(root=cfg.dataset.root, background=False, download=True, transform=transforms)
    val = tv.datasets.Omniglot(root=cfg.dataset.root, background=False, download=True, transform=transforms)



    # pytorch data loader
    train = dynamicBinarizationDataset(train)
    train_loader = data_utils.DataLoader(train, batch_size=cfg.dataset.batch_size, shuffle=True, **kwargs)

    validation = val
    val_loader = data_utils.DataLoader(validation, batch_size=cfg.dataset.batch_size_val, shuffle=False, **kwargs)

    test = test
    test_loader = data_utils.DataLoader(test, batch_size=cfg.dataset.batch_size_val, shuffle=False, **kwargs)
    print("Length of train_loader: ", len(train_loader), "Length of train_loader dataset: ", len(train_loader.dataset))
    print("Length of val_loader: ", len(val_loader), "Length of val_loader dataset: ", len(val_loader.dataset))
    print("Length of test_loader: ", len(test_loader), "Length of test_loader dataset: ", len(test_loader.dataset))

    return train_loader, val_loader, test_loader, cfg



# ======================================================================================================================
def load_caltech101silhouettes(cfg, **kwargs):
    # set args
    cfg.dataset.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 1, 28, 28))
    wget.download(url="https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat", out=os.path.join(cfg.dataset.root,"caltech101_silhouettes_28_split1.mat"))
    caltech_raw = loadmat(os.path.join(cfg.dataset.root,"caltech101_silhouettes_28_split1.mat"))
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
    train_loader = data_utils.DataLoader(train, batch_size=cfg.dataset.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=cfg.dataset.batch_size_val, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=cfg.dataset.batch_size_val, shuffle=True, **kwargs)
    print("Length of train_loader: ", len(train_loader), "Length of train_loader dataset: ", len(train_loader.dataset))
    print("Length of val_loader: ", len(val_loader), "Length of val_loader dataset: ", len(val_loader.dataset))
    print("Length of test_loader: ", len(test_loader), "Length of test_loader dataset: ", len(test_loader.dataset))
    return train_loader, val_loader, test_loader, cfg

def load_mnist(cfg, **kwargs):
    transform = tfm.Compose([tfm.Resize(cfg.dataset.img_size), tfm.ToTensor(), tfm.Normalize((0.5), (0.5),)])

    ds_train = tv.datasets.MNIST('{}/mnist'.format(cfg.dataset.root_dataset), download=True,
                                             transform=tfm.Compose([
                                             tfm.Resize(cfg.dataset.img_size),
                                             tfm.ToTensor(),
                                             tfm.Normalize((0.5,), (0.5,)),
                               ]))
    ds_test = tv.datasets.MNIST('{}/mnist'.format(cfg.dataset.root_dataset), download=True, train=False,
                                             transform=tfm.Compose([
                                             tfm.Resize(cfg.dataset.img_size),
                                             tfm.ToTensor(),
                                             tfm.Normalize((0.5,), (0.5,)),
                               ]))
    
    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=cfg.dataset.batch_size_val, shuffle=False, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(ds_test, batch_size=cfg.dataset.batch_size_val, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_val, dataloader_test, cfg

def load_binary_mnist(cfg, **kwargs):
    if not os.path.exists(os.path.join(cfg.dataset.root_dataset, "binarized_mnist_train.amat")):
        wget.download(url="http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat", out=os.path.join(cfg.dataset.root_dataset, "binarized_mnist_train.amat"))
    xtrain = np.loadtxt(os.path.join(cfg.dataset.root_dataset, "binarized_mnist_train.amat"),dtype=np.float32).reshape(-1,1,28, 28,order='C')
    
    if not os.path.exists(os.path.join(cfg.dataset.root_dataset, "binarized_mnist_valid.amat")):
        wget.download(url="http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat", out=os.path.join(cfg.dataset.root_dataset, "binarized_mnist_valid.amat"))
    xvalid = np.loadtxt(os.path.join(cfg.dataset.root_dataset, "binarized_mnist_valid.amat"),dtype=np.float32).reshape(-1,1,28, 28,order='C')
    
    if not os.path.exists(os.path.join(cfg.dataset.root_dataset, "binarized_mnist_test.amat")):
        wget.download(url="http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat", out=os.path.join(cfg.dataset.root_dataset, "binarized_mnist_test.amat"))
    xtest = np.loadtxt(os.path.join(cfg.dataset.root_dataset, "binarized_mnist_test.amat"),dtype=np.float32).reshape(-1,1,28, 28,order='C')

    print(xtrain.shape)
    ds_train = torch.utils.data.TensorDataset(torch.from_numpy(xtrain), torch.tensor(np.zeros(xtrain.shape[0])))
    ds_valid = torch.utils.data.TensorDataset(torch.from_numpy(xvalid), torch.tensor(np.zeros(xvalid.shape[0])))
    ds_test = torch.utils.data.TensorDataset(torch.from_numpy(xtest), torch.tensor(np.zeros(xtest.shape[0])))

    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=0)
    dataloader_valid = torch.utils.data.DataLoader(ds_valid, batch_size=cfg.dataset.batch_size_val, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=cfg.dataset.batch_size_val, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_valid, dataloader_test, cfg


def load_celeba(cfg, **kwargs):

    ds_train = tv.datasets.CelebA('{}/celeba'.format(cfg.dataset.root_dataset), split='train', download=True,
                                                transform=tfm.Compose([
                                                    tfm.Resize(cfg.dataset.img_size),
                                                    tfm.CenterCrop(cfg.dataset.img_size),
                                                    tfm.ToTensor(),
                                                    tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    ds_val = tv.datasets.CelebA('{}/celeba'.format(cfg.dataset.root_dataset), split='valid', download=True,
                                                transform=tfm.Compose([
                                                    tfm.Resize(cfg.dataset.img_size),
                                                    tfm.CenterCrop(cfg.dataset.img_size),
                                                    tfm.ToTensor(),
                                                    tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    ds_test = tv.datasets.CelebA('{}/celeba'.format(cfg.dataset.root_dataset), split='test', download=True,
                                                transform=tfm.Compose([
                                                    tfm.Resize(cfg.dataset.img_size),
                                                    tfm.CenterCrop(cfg.dataset.img_size),
                                                    tfm.ToTensor(),
                                                    tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    

    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(ds_val, batch_size=cfg.dataset.batch_size_val, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=cfg.dataset.batch_size_val, shuffle=False, num_workers=0)
    return dataloader_train, dataloader_val, dataloader_test, cfg



def load_svhn(cfg, **kwargs):
    ds_train = tv.datasets.SVHN('{}/svhn'.format(cfg.dataset.root_dataset), download=True,
                                             transform=tfm.Compose([
                                             tfm.Resize(cfg.img_size),
                                             tfm.ToTensor(),
                                             tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    ds_val = tv.datasets.SVHN('{}/svhn'.format(cfg.dataset.root_dataset), download=True, split='test',
                                             transform=tfm.Compose([
                                             tfm.Resize(cfg.img_size),
                                             tfm.ToTensor(),
                                             tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    
    ds_test = tv.datasets.SVHN('{}/svhn'.format(cfg.dataset.root_dataset), download=True, split='test',
                                                transform=tfm.Compose([
                                                tfm.Resize(cfg.img_size),
                                                tfm.ToTensor(),
                                                tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    

    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(ds_val, batch_size=cfg.dataset.batch_size_val, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=cfg.dataset.batch_size_val, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_val, dataloader_test, cfg



def get_dataset_and_loader(cfg, device):
    dataset = cfg.dataset.dataset_name
    if cfg.dataset.root_dataset is None:
        cfg.dataset.root_dataset = os.path.join(cfg.machine.root, "datasets")
    if dataset.startswith('SVHN'):
        data_train, data_val, data_test, cfg = load_svhn(cfg)
    elif dataset == "MNIST":
        data_train, data_val, data_test, cfg = load_mnist(cfg)
    elif dataset == "BINARY_MNIST":
        data_train, data_val, data_test, cfg = load_binary_mnist(cfg)
    elif dataset == "OMNIGLOT":
        data_train, data_val, data_test, cfg = load_omniglot(cfg)
    elif dataset == "CALTECH101SILHOUETTES":
        data_train, data_val, data_test, cfg = load_caltech101silhouettes(cfg)
    elif dataset == "CELEBA":
        data_train, data_val, data_test, cfg = load_celeba(cfg)
    else :
        raise NotImplementedError("Unknown dataset: {}".format(dataset))
    return data_train, data_val, data_test



