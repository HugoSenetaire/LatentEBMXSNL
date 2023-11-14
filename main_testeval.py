
import torch as t
import hydra
import sys
import os
import pickle

from Dataset.datasets import get_dataset_and_loader
from Model.Trainers import get_trainer
from Model.Sampler import get_prior_sampler, get_posterior_sampler
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
    parser.add_argument("--nb_iter", default = 20000)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--sampler_name', type=str, default='nuts')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--thinning', type=int, default=30)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--num_chains_test', type=int, default=8)
    parser.add_argument('--multiprocess', type=str, default='Cheating')

    args = parser.parse_args()
    cfg = pickle.load(open(os.path.join(args.weights_path, 'cfg.pkl') , "rb"))
    cfg.trainer.use_reverse = True
    cfg.trainer.forward_posterior = True


    device = "cuda:0" if t.cuda.is_available() else "cpu"
    cfg.trainer.device = device

    data_train, data_val, data_test = get_dataset_and_loader(cfg, device)
    total_train = get_trainer(cfg)
    total_train = total_train(cfg, test=True, path_weights=args.weights_path)
    if hasattr(total_train, "train_for_eval"):
        total_train.train_for_eval(nb_iter = args.nb_iter, train_dataloader=data_train, val_dataloader=data_val, test_dataloader=data_test)

    cfg.sampler_prior.name = args.sampler_name
    cfg.sampler_prior.num_samples = args.num_samples
    cfg.sampler_prior.thinning = args.thinning
    cfg.sampler_prior.warmup_steps = args.warmup_steps
    cfg.sampler_prior.num_chains_test = args.num_chains_test
    cfg.sampler_prior.multiprocess = args.multiprocess
    cfg.sampler_posterior.name = args.sampler_name
    cfg.sampler_posterior.num_samples = args.num_samples
    cfg.sampler_posterior.thinning = args.thinning
    cfg.sampler_posterior.warmup_steps = args.warmup_steps
    cfg.sampler_posterior.num_chains_test = args.num_chains_test
    cfg.sampler_posterior.multiprocess = args.multiprocess
    total_train.sampler_prior = get_prior_sampler(cfg.sampler_prior)
    total_train.sampler_posterior = get_posterior_sampler(cfg.sampler_posterior)

    total_train.eval(data_test, step = args.nb_iter, name='test/')
    total_train.plot_latent(data_test, step = args.nb_iter, name='test/')
if __name__ == "__main__":
    main()
