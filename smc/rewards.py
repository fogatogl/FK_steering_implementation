import torch
import torch.nn.functional as F
from smc.classifier import load as load_classifier


def reward(x):
    m = torch.mean(x, dim=(2, 3))
    sigma = 0.1962
    r = m[:, 0] - 0.5*(m[:, 1] + m[:, 2])
    return r/sigma


def make_classifier_reward(weights_path, target, device):
    classifier = load_classifier(weights_path, device)

    def score(x):
        with torch.no_grad():
            # DDPM predict_x0 clamps to [-1, 1], but state["x"] at t=0 does not.
            # Clamp explicitly to stay on the classifier's training support.
            x_in = torch.clamp(x, -1.0, 1.0)
            logits = classifier(x_in)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs[:, target]

    return score
