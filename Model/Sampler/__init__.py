from .sampler import SampleLangevinPosterior, SampleLangevinPrior


def get_prior_sampler(cfg):
    return SampleLangevinPrior(cfg.K, cfg.a, cfg.clamp_min_data, cfg.clamp_max_data, cfg.clamp_min_grad, cfg.clamp_max_grad, cfg.clip_data_norm, cfg.clip_grad_norm)

def get_posterior_sampler(cfg):
    return SampleLangevinPosterior(cfg.K, cfg.a, cfg.clamp_min_data, cfg.clamp_max_data, cfg.clamp_min_grad, cfg.clamp_max_grad, cfg.clip_data_norm, cfg.clip_grad_norm)
