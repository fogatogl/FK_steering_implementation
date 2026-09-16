import torch


def best_of_n(model, reward, n, generator):
    state = model.initial_state(n, generator)
    for t in range(model.T-1, -1, -1):
        state = model.step(state, t, generator)
    r = reward(state["x"])
    i = torch.argmax(r)
    return state["x"][i].cpu(), r[i].item(), n*model.T

