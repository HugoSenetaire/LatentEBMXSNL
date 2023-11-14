from .gaussian import GaussianProposal
from .student import StudentProposal


def get_proposal(nz, cfg_proposal):
    if cfg_proposal.proposal_name == 'gaussian':
        return GaussianProposal(nz, cfg_proposal)
    elif cfg_proposal.proposal_name == 'student':
        return StudentProposal(nz, cfg_proposal)
    elif cfg_proposal.proposal_name is None:
        return None
    else:
        raise NotImplementedError