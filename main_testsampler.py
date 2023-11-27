
import torch as t
import hydra
import sys
import os
import pickle

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
    parser.add_argument("--nb_iter", default = 20000)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--prior_sampler_name', type=str, default='nuts')
    parser.add_argument('--prior_num_samples', type=int, default=8)
    parser.add_argument('--prior_thinning', type=int, default=1)
    parser.add_argument('--prior_warmup_steps', type=int, default=0)
    parser.add_argument('--prior_num_chains_test', type=int, default=8)
    parser.add_argument('--prior_multiprocess', type=str, default='Cheating')

    parser.add_argument('--posterior_sampler_name', type=str, default='nuts')
    parser.add_argument('--posterior_num_samples', type=int, default=8)
    parser.add_argument('--posterior_thinning', type=int, default=2)
    parser.add_argument('--posterior_warmup_steps', type=int, default=0)
    parser.add_argument('--posterior_num_chains_test', type=int, default=8)
    parser.add_argument('--posterior_multiprocess', type=str, default='Cheating')

    args = parser.parse_args()

    cfg = pickle.load(open(os.path.join(args.weights_path, 'cfg.pkl') , "rb"))
    cfg.trainer.use_reverse = True
    cfg.trainer.forward_posterior = True
    cfg.sampler_prior.sampler_name = args.prior_sampler_name
    cfg.sampler_prior.num_samples = args.prior_num_samples
    cfg.sampler_prior.thinning = args.prior_thinning
    cfg.sampler_prior.warmup_steps = args.prior_warmup_steps
    cfg.sampler_prior.num_chains_test = args.prior_num_chains_test
    cfg.sampler_prior.multiprocess = args.prior_multiprocess
    cfg.sampler_posterior.sampler_name = args.posterior_sampler_name
    cfg.sampler_posterior.num_samples = args.posterior_num_samples
    cfg.sampler_posterior.thinning = args.posterior_thinning
    cfg.sampler_posterior.warmup_steps = args.posterior_warmup_steps
    cfg.sampler_posterior.num_chains_test = args.posterior_num_chains_test
    cfg.sampler_posterior.multiprocess = args.posterior_multiprocess
    

    device = "cuda:0" if t.cuda.is_available() else "cpu"
    cfg.trainer.device = device

    data_train, data_val, data_test = get_dataset_and_loader(cfg, device)
    trainer = get_trainer(cfg)
    trainer = trainer(cfg, test=True, path_weights=args.weights_path)
    trainer.get_fixed_x(data_train, data_val, data_test)
    trainer.draw_samples(next(iter(data_test))[0].to(device), step = args.nb_iter,)

if __name__ == "__main__":
    main()
