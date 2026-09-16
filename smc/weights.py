import torch


def normalize_logw(logW):
    k = logW.shape[0]
    logS = torch.logsumexp(logW, dim=0)
    w = torch.exp(logW - logS)
    log_ell = logS - torch.log(torch.tensor(k, dtype=logW.dtype, device=logW.device))
    return w, log_ell


def ess(w):
    return 1/torch.sum(torch.square(w)).item()


def should_resample(w, threshold=0.5):
    return ess(w) < threshold*w.shape[0]


