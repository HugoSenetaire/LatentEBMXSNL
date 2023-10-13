from .sampler import SampleLangevinPosterior, SampleLangevinPrior


def get_prior_sampler(cfg):
    return SampleLangevinPrior(cfg.K, cfg.a, cfg.clamp_min, cfg.clamp_max, cfg.clip_grad_norm)

def get_posterior_sampler(cfg):
    return SampleLangevinPosterior(cfg.K, cfg.a, cfg.clamp_min, cfg.clamp_max, cfg.clip_grad_norm)
