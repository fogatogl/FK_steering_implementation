import torch


def normalize_logw(logW):
    k = logW.shape[0]
    logS = torch.logsumexp(logW, dim=0)
    w = torch.exp(logW - logS)
    log_ell = logS - torch.log(torch.tensor(k, dtype=logW.dtype, device=logW.device))
    return w, log_ell
