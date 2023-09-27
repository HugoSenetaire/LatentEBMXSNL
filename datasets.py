import numpy as np
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tfm


def get_dataset_and_loader(cfg, device):
    dataset = cfg["dataset"]
    if dataset.startswith("SVHN"):
        transform = tfm.Compose(
            [
                tfm.Resize(cfg["img_size"]),
                tfm.ToTensor(),
                tfm.Normalize(([0.5] * 3), ([0.5] * 3)),
            ]
        )
        data_train = t.stack(
            [
                x[0]
                for x in tv.datasets.SVHN(
                    download=True, root=cfg["dataset_dir"] / "svhn", transform=transform
                )
            ]
        ).to(device)
        data_test = t.stack(
            [
                x[0]
                for x in tv.datasets.SVHN(
                    split="test",
                    download=True,
                    root=cfg["dataset_dir"] / "svhn",
                    transform=transform,
                )
            ]
        ).to(device)
        data_valid = t.stack(
            [
                x[0]
                for x in tv.datasets.SVHN(
                    split="test",
                    download=True,
                    root=cfg["dataset_dir"] / "svhn",
                    transform=transform,
                )
            ]
        ).to(device)
    elif dataset == "MNIST":
        transform = tfm.Compose(
            [
                tfm.Resize(cfg["img_size"]),
                tfm.ToTensor(),
                tfm.Normalize(
                    (0.5),
                    (0.5),
                ),
            ]
        )
        data_train = t.stack(
            [
                x[0]
                for x in tv.datasets.MNIST(
                    download=True,
                    root=cfg["dataset_dir"] / "mnist",
                    transform=transform,
                )
            ]
        ).to(device)
        data_test = t.stack(
            [
                x[0]
                for x in tv.datasets.MNIST(
                    split="test",
                    download=True,
                    root=cfg["dataset_dir"] / "mnist",
                    transform=transform,
                )
            ]
        ).to(device)
        data_valid = data_test
    elif dataset == "BINARYMNIST":
        xtrain = np.loadtxt(
            "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat",
            dtype=np.float32,
        ).reshape(-1, 28, 28, order="C")
        xvalid = np.loadtxt(
            "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat",
            dtype=np.float32,
        ).reshape(-1, 28, 28, order="C")
        xtest = np.loadtxt(
            "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat",
            dtype=np.float32,
        ).reshape(-1, 28, 28, order="C")
        data_train = t.from_numpy(xtrain).to(device)
        data_valid = t.from_numpy(xvalid).to(device)
        data_test = t.from_numpy(xtest).to(device)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(dataset))
    return data_train, data_valid, data_test
