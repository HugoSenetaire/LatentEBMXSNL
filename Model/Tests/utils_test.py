import sys
import os
import torch
current_path = os.path.dirname(os.path.realpath(__file__))
current_path = current_path.split('Model')[0]
sys.path.append(current_path)


from hydra_config import store_main
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf


import hydra
import copy


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

