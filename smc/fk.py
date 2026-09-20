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


def potentials(r_t, gate, lam, lam_prev, potential, last, acc, lam_placement):
    """
    Computes logG and updates the gate across potentials:
      difference: curr = r_t,               prev = gate
      max:        curr = max(gate, r_t),    prev = where(gate == -inf, 0, gate)
      sum:        curr = gate + r_t,        prev = gate

    Under tempering:
      logG = lam * curr - lam_prev * prev   (curr = r_t on last)
    Under terminal correction:
      logG = lam * (curr - prev)            if not last
      logG = lam * r_t - acc                if last
    """
    if potential == "difference":
        curr = r_t
        prev = gate
        new_gate = r_t
    elif potential == "max":
        prev = torch.where(torch.isneginf(gate), torch.zeros_like(r_t), gate)
        curr = r_t if last else torch.maximum(gate, r_t)
        new_gate = torch.maximum(gate, r_t)
    elif potential == "sum":
        curr = r_t if last else gate + r_t
        prev = gate
        new_gate = gate + r_t
    else:
        raise ValueError(f"unknown potential: {potential}")

    if lam_placement == "tempering":
        logG = lam * curr - lam_prev * prev
    elif last and acc is not None:
        logG = lam * r_t - acc
    else:
        logG = lam * (curr - prev)

    return logG, new_gate


def fk_steer(
    model,
    reward,
    k,
    lam,
    potential,
    generator,
    resample_threshold=0.5,
    resampler=resample_systematic,
    schedule=None,
    resample_last=False,
    lam_schedule=None,
    lam_placement="terminal",
):
    if lam_placement not in ("terminal", "tempering"):
        raise ValueError(f"unknown lam_placement: {lam_placement}")

    timesteps = list(model.timesteps)
    n_steps = len(timesteps)
    schedule = set(range(n_steps)) if schedule is None else set(schedule)
    if (n_steps - 1) not in schedule:
        raise ValueError(
            "le pas terminal doit être dans le calendrier : c'est lui qui porte la "
            "correction G_last ; sans lui la cible devient exp(lam * max) au lieu de exp(lam * r_0)"
        )
    state = model.initial_state(k, generator)
    device = state["x"].device

    logg_list, ess_trace, anc = [], [], []

    if potential in ("difference", "sum"):
        gate = torch.zeros(k, device=device)
    elif potential == "max":
        gate = torch.full((k,), -float("inf"), device=device)
    else:
        raise ValueError(f"unknown potential: {potential}")

    # acc is an integral along the lineage and is never reset; logW is a weight and is.
    acc = torch.zeros(k, device=device)
    lam_prev = 0.0

    logW = torch.zeros(k, device=device)
    w = torch.full((k,), 1.0 / k, device=device)
    n_resamplings = 0
    ess_min = float(k)
    current_ess = float(k)

    for i, t in enumerate(timesteps):
        last = (i == n_steps - 1)
        step_lam = lam if lam_schedule is None else float(lam_schedule[i])
        state = model.step(state, t, generator)

        if i not in schedule:
            logg_list.append(torch.zeros(k, device=device))
            ess_trace.append(current_ess)
            anc.append(torch.arange(k, device=device))
            continue

        r_t = reward(state["x"]) if last else reward(model.predict_x0(state))

        logG, gate = potentials(
            r_t=r_t,
            gate=gate,
            lam=step_lam,
            lam_prev=lam_prev,
            potential=potential,
            last=last,
            acc=(acc if lam_schedule is not None else None),
            lam_placement=lam_placement,
        )
        lam_prev = step_lam
        if not last:
            acc = acc + logG

        logg_list.append(logG)

        logW = logW + logG
        w, _ = normalize_logw(logW)

        current_ess = ess(w)
        ess_trace.append(current_ess)
        ess_min = min(ess_min, current_ess)

        may_resample = resample_last or not last
        if may_resample and should_resample(w, resample_threshold):
            n_resamplings += 1
            idx = resampler(w, k, generator)
            anc.append(idx)
            for key in state:
                if isinstance(state[key], torch.Tensor):
                    state[key] = state[key][idx]
            gate = gate[idx]
            acc = acc[idx]
            logW = torch.zeros(k, device=device)
            w = torch.full((k,), 1.0 / k, device=device)
        else:
            anc.append(torch.arange(k, device=device))

    return state["x"], {
        "ess": current_ess,
        "ess_min": ess_min,
        "ess_trace": torch.tensor(ess_trace, device=device),
        "timesteps": torch.as_tensor(timesteps, device=device),
        "n_resamplings": n_resamplings,
        "weights": w,
        "rewards": r_t,
        "logG": torch.stack(logg_list),
        "ancestors": torch.stack(anc),
    }