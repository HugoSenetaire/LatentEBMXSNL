
import torch as t
import hydra
import sys
import os

from Dataset.datasets import get_dataset_and_loader
from Model.Trainers import get_trainer
from hydra_config import store_main
from hydra import compose, initialize_config_dir
from argparse import ArgumentParser

current_path = os.path.dirname(os.path.realpath(__file__))
current_path = current_path.split('Model')[0]
sys.path.append(current_path)






def clear_hydra():
    hydra.core.global_hydra.GlobalHydra.instance().clear()

def get_config(config_path = "conf_test", config_name = "conf"):
    # Test for all prior distributions
    clear_hydra()
    store_main()
    print(current_path)
    initialize_config_dir(config_dir=os.path.join(current_path, config_path), job_name="test_app")
    # initialize(config_path=config_path, job_name="test_app")
    cfg = compose(config_name=config_name, overrides=[])
    return cfg



def main():
    parser = ArgumentParser()
    parser.add_argument("--config_path", default="conf_binary_mnist_2d")
    parser.add_argument("--config_name", default="conf_snelbo")
    parser.add_argument("--special_name", type = str, default=None)
    parser.add_argument("--overrides", nargs='?', default=[])
    args = parser.parse_args()
    liste_overrides = [str(item) for item in args.overrides.split(",")]
    clear_hydra()
    store_main()
    initialize_config_dir(config_dir=os.path.join(current_path, args.config_path), version_base="1.1", )
    cfg = compose(config_name=args.config_name, overrides=liste_overrides)
    
    device = "cuda:0" if t.cuda.is_available() else "cpu"
    cfg.trainer.device = device

    data_train, data_val, data_test = get_dataset_and_loader(cfg, device)
    total_train = get_trainer(cfg)
    total_train = total_train(cfg, test=False, path_weights=None, load_iter=None, special_name=args.special_name)
    total_train.train(train_dataloader=data_train, val_dataloader=data_val, test_dataloader=data_test)


if __name__ == "__main__":
    main()
