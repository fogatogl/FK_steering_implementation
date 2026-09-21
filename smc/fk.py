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


def _potential_terms(r_t, gate, lam_prev, potential, last, acc, lam_placement,
                     potential_form):
    """
    Returns (base, offset, new_gate) with  logG = lam * base + offset.

    Every form in use is affine in the step's lambda, which is what lets
    adaptive tempering bisect lambda_t without re-deriving the potential:
      increment / terminal :  base = curr - prev,  offset = 0
                              base = r_t,          offset = -acc        (last, acc given)
      increment / tempering:  base = curr,         offset = -lam_prev * prev
      statistic (paper)    :  base = curr,         offset = 0
                              base = r_t,          offset = -acc        (last)
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

    zero = torch.zeros_like(r_t)

    if potential_form == "statistic" and potential in ("max", "sum"):
        if last:
            if acc is None:
                raise ValueError("statistic form needs acc at the terminal step")
            return r_t, -acc, new_gate
        return curr, zero, new_gate

    if lam_placement == "tempering":
        return curr, -lam_prev * prev, new_gate
    if last and acc is not None:
        return r_t, -acc, new_gate
    return curr - prev, zero, new_gate


def potentials(r_t, gate, lam, lam_prev, potential, last, acc, lam_placement,
               potential_form="increment"):
    """
    Computes logG and updates the gate across potentials:
      difference: curr = r_t,               prev = gate
      max:        curr = max(gate, r_t),    prev = where(gate == -inf, 0, gate)
      sum:        curr = gate + r_t,        prev = gate

    potential_form == "increment" (historical behaviour, smc/ records before 21/09):
      Under tempering:
        logG = lam * curr - lam_prev * prev   (curr = r_t on last)
      Under terminal correction:
        logG = lam * (curr - prev)            if not last
        logG = lam * r_t - acc                if last

    potential_form == "statistic" (paper eq. for G_t and the authors' released code):
        logG = lam * curr                     if not last   (the statistic itself)
        logG = lam * r_t - acc                if last       (G_0 closes the product)
      `difference` is the same under both forms. Requires lam_placement == "terminal".
    """
    base, offset, new_gate = _potential_terms(
        r_t, gate, lam_prev, potential, last, acc, lam_placement, potential_form
    )
    return lam * base + offset, new_gate


def bisect_lambda(logW, base, offset, ess_target, lam_max, lam_default, iters=40):
    """
    Adaptive tempering (Chopin & Papaspiliopoulos ch. 17; Jasra et al. 2011):
    the lambda_t at which the ESS of normalize(logW + lambda_t * base + offset)
    equals ess_target. ESS is decreasing in lambda_t once logW is flat, which it
    is at threshold 1.0 and after every resampling.

    Returns lam_default when base has no across-particle spread: no lambda can
    separate particles whose increments are all equal, which is the `max`
    ratchet's failure and is meant to be visible in lam_trace rather than
    disguised as lam_max.
    """
    if float(base.max() - base.min()) < 1e-12:
        return float(lam_default)

    def ess_at(l):
        w, _ = normalize_logw(logW + l * base + offset)
        return ess(w)

    if ess_at(lam_max) >= ess_target:
        return float(lam_max)
    if ess_at(0.0) <= ess_target:
        return 0.0
    lo, hi = 0.0, float(lam_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ess_at(mid) > ess_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
    potential_form="increment",
    adaptive_lam=False,
    ess_target=None,
    lam_max=None,
):
    """
    lam_schedule : list of per-step lambdas (the ramps of runs 13-15), or None.
    adaptive_lam : bisect lambda_t at every scheduled non-terminal step so the
                   ESS after reweighting equals ess_target (default k/2), capped
                   at lam_max (default 10 * lam). The terminal step keeps lam, so
                   the target exp(lam * r(x_0)) is unchanged; acc carries the
                   correction. Exclusive with lam_schedule.
    """
    if lam_placement not in ("terminal", "tempering"):
        raise ValueError(f"unknown lam_placement: {lam_placement}")
    if potential_form not in ("increment", "statistic"):
        raise ValueError(f"unknown potential_form: {potential_form}")
    if potential_form == "statistic" and lam_placement == "tempering":
        raise ValueError(
            "potential_form='statistic' is only defined with lam_placement='terminal'"
        )
    if adaptive_lam and lam_schedule is not None:
        raise ValueError("adaptive_lam and lam_schedule are exclusive")
    if adaptive_lam:
        ess_target = 0.5 * k if ess_target is None else float(ess_target)
        lam_max = 10.0 * lam if lam_max is None else float(lam_max)

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

    logg_list, ess_trace, anc, lam_trace = [], [], [], []
    # r_t at each scheduled step, per slot, read BEFORE that step's resampling, so slot j
    # of row m is the particle that anc[schedule_idx[m]] then copies or drops. Follow the
    # ancestors to attach it to the r(x_0) of a final particle. This is the paper's
    # fig:reward-corr, whose values its source never prints.
    r_list, r_idx = [], []

    if potential in ("difference", "sum"):
        gate = torch.zeros(k, device=device)
    elif potential == "max":
        gate = torch.full((k,), -float("inf"), device=device)
    else:
        raise ValueError(f"unknown potential: {potential}")

    # acc is an integral along the lineage and is never reset; logW is a weight and is.
    acc = torch.zeros(k, device=device)
    lam_prev = 0.0
    needs_acc = (lam_schedule is not None) or adaptive_lam or (potential_form == "statistic")

    logW = torch.zeros(k, device=device)
    w = torch.full((k,), 1.0 / k, device=device)
    n_resamplings = 0
    ess_min = float(k)
    current_ess = float(k)

    for i, t in enumerate(timesteps):
        last = (i == n_steps - 1)
        state = model.step(state, t, generator)

        if i not in schedule:
            logg_list.append(torch.zeros(k, device=device))
            ess_trace.append(current_ess)
            anc.append(torch.arange(k, device=device))
            lam_trace.append(float("nan"))
            continue

        r_t = reward(state["x"]) if last else reward(model.predict_x0(state))
        r_list.append(r_t.detach().clone())
        r_idx.append(i)

        base, offset, new_gate = _potential_terms(
            r_t=r_t,
            gate=gate,
            lam_prev=lam_prev,
            potential=potential,
            last=last,
            acc=(acc if needs_acc else None),
            lam_placement=lam_placement,
            potential_form=potential_form,
        )

        if last or not adaptive_lam:
            step_lam = lam if lam_schedule is None else float(lam_schedule[i])
        else:
            step_lam = bisect_lambda(logW, base, offset, ess_target, lam_max, lam)

        logG = step_lam * base + offset
        gate = new_gate
        lam_prev = step_lam
        lam_trace.append(step_lam)
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
        "lam_trace": torch.tensor(lam_trace, device=device),
        "potential": potential,
        "potential_form": potential_form,
        "adaptive_lam": adaptive_lam,
        "r_at_schedule": torch.stack(r_list),
        "schedule_idx": torch.as_tensor(r_idx, device=device),
    }