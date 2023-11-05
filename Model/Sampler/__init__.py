from .type_steps.basic_langevin import LangevinPosterior, LangevinPrior
from .type_steps.nuts_sampler import NutsPosterior, NutsPrior
from .type_steps.basic_langevin import MALAPosterior, MALAPrior

def get_prior_sampler(cfg):
    if cfg.sampler_name == 'langevin' :
        return LangevinPrior(num_samples=cfg.num_samples,
                            thinning=cfg.thinning,
                            warmup_steps=cfg.warmup_steps,
                            step_size=cfg.step_size,
                            clamp_min_data=cfg.clamp_min_data,
                            clamp_max_data=cfg.clamp_max_data,
                            clamp_min_grad=cfg.clamp_min_grad,
                            clamp_max_grad=cfg.clamp_max_grad,
                            clip_data_norm=cfg.clip_data_norm,
                            clip_grad_norm=cfg.clip_grad_norm,
                            hyperspherical=cfg.hyperspherical,)
    elif cfg.sampler_name == 'nuts':
        return NutsPrior(num_samples=cfg.num_samples,
                        thinning=cfg.thinning,
                        warmup_steps=cfg.warmup_steps,
                        step_size=cfg.step_size,
                        clamp_min_data=cfg.clamp_min_data,
                        clamp_max_data=cfg.clamp_max_data,
                        clamp_min_grad=cfg.clamp_min_grad,
                        clamp_max_grad=cfg.clamp_max_grad,
                        clip_data_norm=cfg.clip_data_norm,
                        clip_grad_norm=cfg.clip_grad_norm,
                        hyperspherical=cfg.hyperspherical,
                        multiprocess=cfg.multiprocess,)
    elif cfg.sampler_name == 'mala':
        return MALAPrior(num_samples=cfg.num_samples,
                        thinning=cfg.thinning,
                        warmup_steps=cfg.warmup_steps,
                        step_size=cfg.step_size,
                        clamp_min_data=cfg.clamp_min_data,
                        clamp_max_data=cfg.clamp_max_data,
                        clamp_min_grad=cfg.clamp_min_grad,
                        clamp_max_grad=cfg.clamp_max_grad,
                        clip_data_norm=cfg.clip_data_norm,
                        clip_grad_norm=cfg.clip_grad_norm,
                        hyperspherical=cfg.hyperspherical,)
    else:
        raise NotImplementedError

def get_posterior_sampler(cfg):
    if cfg.sampler_name == 'langevin':
        return LangevinPosterior(num_samples=cfg.num_samples,
                                thinning=cfg.thinning,
                                warmup_steps=cfg.warmup_steps,
                                step_size=cfg.step_size,
                                clamp_min_data=cfg.clamp_min_data,
                                clamp_max_data=cfg.clamp_max_data,
                                clamp_min_grad=cfg.clamp_min_grad,
                                clamp_max_grad=cfg.clamp_max_grad,
                                clip_data_norm=cfg.clip_data_norm,
                                clip_grad_norm=cfg.clip_grad_norm,
                                hyperspherical=cfg.hyperspherical,)
    elif cfg.sampler_name == 'nuts':
        return NutsPosterior(num_samples=cfg.num_samples,
                            thinning=cfg.thinning,
                            warmup_steps=cfg.warmup_steps,
                            step_size=cfg.step_size,
                            clamp_min_data=cfg.clamp_min_data,
                            clamp_max_data=cfg.clamp_max_data,
                            clamp_min_grad=cfg.clamp_min_grad,
                            clamp_max_grad=cfg.clamp_max_grad,
                            clip_data_norm=cfg.clip_data_norm,
                            clip_grad_norm=cfg.clip_grad_norm,
                            hyperspherical=cfg.hyperspherical,
                            multiprocess=cfg.multiprocess,)
    elif cfg.sampler_name == 'mala':
        return MALAPosterior(num_samples=cfg.num_samples,
                        thinning=cfg.thinning,
                        warmup_steps=cfg.warmup_steps,
                        step_size=cfg.step_size,
                        clamp_min_data=cfg.clamp_min_data,
                        clamp_max_data=cfg.clamp_max_data,
                        clamp_min_grad=cfg.clamp_min_grad,
                        clamp_max_grad=cfg.clamp_max_grad,
                        clip_data_norm=cfg.clip_data_norm,
                        clip_grad_norm=cfg.clip_grad_norm,
                        hyperspherical=cfg.hyperspherical,)
    else:
        raise NotImplementedError
