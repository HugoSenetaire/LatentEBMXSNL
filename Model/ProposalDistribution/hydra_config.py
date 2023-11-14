import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union




@dataclass
class BaseProposalConfig:
    proposal_name: Union[str,None] = MISSING

@dataclass
class NoProposalConfig(BaseProposalConfig):
    proposal_name: Union[str,None] = None

@dataclass
class GaussianProposalConfig(BaseProposalConfig):
    proposal_name: str = "gaussian"
    mu : Optional[float] = 0.
    sigma : Optional[float] = 1.

@dataclass
class StudentProposalConfig(BaseProposalConfig):
    proposal_name: str = "student"
    mu : Optional[float] = 0.
    sigma : Optional[float] = 1.
    df : Optional[float] = 1.


def store_base_proposal(cs: ConfigStore):
    cs.store(group="proposal", name="base_proposal", node=BaseProposalConfig)
    cs.store(group="proposal", name="base_no_proposal", node=NoProposalConfig)
    cs.store(group="proposal", name="base_gaussian_proposal", node=GaussianProposalConfig)
    cs.store(group="proposal", name="base_student_proposal", node=StudentProposalConfig)