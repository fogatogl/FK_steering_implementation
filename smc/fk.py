import torch
from smc.weights import normalize_logw, should_resample, ess
from smc.resampling import resample_systematic, resample_multinomial


def best_of_n(model, reward, n, generator):
    state = model.initial_state(n, generator)
    for t in model.timesteps:
        state = model.step(state, t, generator)
    r = reward(state["x"])
    w = torch.full((n,), 1.0 / n, device=state["x"].device)
    
    return state["x"], {
        "ess": float(n),
        "weights": w,
        "rewards": r,
    }


def potentials(r_t, gate, lam, potential, t):
    if potential == "difference":
        logG = lam * (r_t - gate)
        new_gate = r_t
        return logG, new_gate

    elif potential == "max":
        if t == 0:
            logG = lam * (r_t - gate)
            new_gate = torch.maximum(gate, r_t)
        else:
            new_gate = torch.maximum(gate, r_t)
            prev_gate = torch.where(torch.isneginf(gate), torch.zeros_like(r_t), gate)
            logG = lam * (new_gate - prev_gate)
        return logG, new_gate

    elif potential == "sum":
        if t == 0:
            logG = lam * (r_t - gate)
            new_gate = gate + r_t
        else:
            logG = lam * r_t
            new_gate = gate + r_t
        return logG, new_gate

    else:
        raise ValueError(f"unknown potential: {potential}")


def fk_steer(
    model,
    reward,
    k,
    lam,
    potential,
    generator,
    resample_threshold=0.5,
    resampler=resample_systematic,
):
    state = model.initial_state(k, generator)
    device = state["x"].device

    logg_list = []
    ess_trace = []
    anc = []

    if potential in ("difference", "sum"):
        gate = torch.zeros(k, device=device)
    elif potential == "max":
        gate = torch.full((k,), -float("inf"), device=device)
    else:
        raise ValueError(f"unknown potential: {potential}")

    logW = torch.zeros(k, device=device)
    w = torch.full((k,), 1.0 / k, device=device)

    n_resamplings = 0
    ess_min = float(k)

    for t in model.timesteps:
        state = model.step(state, t, generator)

        if t > 0:
            r_t = reward(model.predict_x0(state))
        else:
            r_t = reward(state["x"])

        logG, gate = potentials(r_t, gate, lam, potential, t)
        logg_list.append(logG)

        logW = logW + logG
        w, _ = normalize_logw(logW)

        current_ess = ess(w)
        ess_trace.append(current_ess)
        if current_ess < ess_min:
            ess_min = current_ess

        if t > 0 and should_resample(w, resample_threshold):
            n_resamplings += 1
            idx = resampler(w, k, generator)
            anc.append(idx)
            for key in state:
                if isinstance(state[key], torch.Tensor):
                    state[key] = state[key][idx]
            gate = gate[idx]
            logW = torch.zeros(k, device=device)
            w = torch.full((k,), 1.0 / k, device=device)
        else:
            anc.append(torch.arange(k, device=device))

    return state["x"], {
        "ess": current_ess,
        "ess_min": ess_min,
        "ess_trace": torch.tensor(ess_trace, device=device),
        "timesteps": torch.as_tensor(model.timesteps, device=device),
        "n_resamplings": n_resamplings,
        "weights": w,
        "rewards": r_t,
        "logG": torch.stack(logg_list),
        "ancestors": torch.stack(anc),
    }