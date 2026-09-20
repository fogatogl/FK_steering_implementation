import pytest
import torch
from smc.fk import fk_steer


class ControlledDummyModel:
    def __init__(self, timesteps):
        self.timesteps = list(timesteps)

    def initial_state(self, k, generator=None):
        return {"x": torch.linspace(1.0, 2.0, steps=k)}

    def step(self, state, t, generator=None):
        return {"x": state["x"] * 1.15}

    def predict_x0(self, state):
        return state["x"] * 1.1


def dummy_reward(x):
    return 0.5 * x


def accumulate_along_lineage(logG_stack, ancestors_stack):
    T, k = logG_stack.shape
    device = logG_stack.device
    curr_slots = torch.arange(k, device=device)
    total_logg = torch.zeros(k, device=device)

    for step in reversed(range(T)):
        curr_slots = ancestors_stack[step, curr_slots]
        total_logg += logG_stack[step, curr_slots]

    return total_logg, curr_slots


@pytest.mark.parametrize(
    "potential,lam",
    [
        ("difference", 3.5),
        ("max", 3.5),
        ("sum", 0.8),
    ],
)
def test_telescoping_with_active_non_collapsed_resampling(potential, lam):
    """Verifies telescoping invariant with lineage mixing on constant lambda."""
    k = 16
    timesteps = [4, 3, 2, 1, 0]
    gen = torch.Generator().manual_seed(0)

    model = ControlledDummyModel(timesteps)
    final_x, out = fk_steer(
        model=model,
        reward=dummy_reward,
        k=k,
        lam=lam,
        potential=potential,
        generator=gen,
        resample_threshold=0.7,
    )

    assert out["n_resamplings"] > 0, f"No resampling occurred for {potential}"

    lineage_logg, initial_ancestors = accumulate_along_lineage(
        out["logG"], out["ancestors"]
    )

    n_surviving = torch.unique(initial_ancestors).numel()
    assert 1 < n_surviving < k, (
        f"Lineage collapse occurred for {potential} (survivors: {n_surviving}/{k})."
    )

    expected = lam * dummy_reward(final_x)
    torch.testing.assert_close(
        lineage_logg,
        expected,
        atol=1e-5,
        rtol=1e-5,
        msg=f"Telescoping invariant violated for {potential}",
    )


def test_invariant_lambda_zero():
    """Verifies that lambda=0 preserves strict neutral dynamics."""
    k = 8
    timesteps = [3, 2, 1, 0]
    gen = torch.Generator().manual_seed(0)

    model = ControlledDummyModel(timesteps)
    _, out = fk_steer(
        model=model,
        reward=dummy_reward,
        k=k,
        lam=0.0,
        potential="difference",
        generator=gen,
        resample_threshold=0.5,
    )

    torch.testing.assert_close(
        out["logG"], torch.zeros_like(out["logG"]), atol=1e-6, rtol=1e-6
    )
    assert out["n_resamplings"] == 0


@pytest.mark.parametrize("potential", ["difference", "max", "sum"])
def test_tempering_constant_equivalence(potential):
    """Verifies tempering with constant schedule matches the default mode exactly."""
    k = 8
    timesteps = [4, 3, 2, 1, 0]
    lam = 0.8 if potential == "sum" else 3.5
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)

    model1 = ControlledDummyModel(timesteps)

    _, out_orig = fk_steer(
        model=model1,
        reward=dummy_reward,
        k=k,
        lam=lam,
        potential=potential,
        generator=gen1,
        resample_threshold=0.7,
        lam_placement="terminal",
    )

    _, out_temp = fk_steer(
        model=model1,
        reward=dummy_reward,
        k=k,
        lam=lam,
        potential=potential,
        generator=gen2,
        resample_threshold=0.7,
        lam_schedule=[lam] * len(timesteps),
        lam_placement="tempering",
    )

    torch.testing.assert_close(
        out_temp["logG"],
        out_orig["logG"],
        atol=1e-6,
        rtol=1e-6,
        msg=f"Tempering at constant lambda did not produce identical logG for {potential}.",
    )


@pytest.mark.parametrize("potential", ["difference", "max", "sum"])
@pytest.mark.parametrize("placement", ["terminal", "tempering"])
@pytest.mark.parametrize(
    "lam_schedule,threshold",
    [
        ([2.0, 4.0, 6.0, 8.0, 10.0], 0.7),
        ([10.0 * ((i + 1) / 5.0) ** 2 for i in range(5)], 0.85),
        ([0.0, 10.0, 10.0, 10.0, 10.0], 0.7),
    ],
)
def test_telescoping_time_dependent_lambda(potential, placement, lam_schedule, threshold):
    """Verifies lineage sum equals lam_T * r(x_0) across potentials and placements."""
    if potential == "sum":
        lam_schedule = [val * 0.2 for val in lam_schedule]

    k = 16
    timesteps = [4, 3, 2, 1, 0]
    gen = torch.Generator().manual_seed(0)

    model = ControlledDummyModel(timesteps)
    final_x, out = fk_steer(
        model=model,
        reward=dummy_reward,
        k=k,
        lam=-999.0,
        potential=potential,
        generator=gen,
        resample_threshold=threshold,
        lam_schedule=lam_schedule,
        lam_placement=placement,
    )

    assert out["n_resamplings"] > 0, (
        f"No resampling for {potential}, placement={placement}, schedule={lam_schedule}"
    )

    lineage_logg, initial_ancestors = accumulate_along_lineage(
        out["logG"], out["ancestors"]
    )

    n_surviving = torch.unique(initial_ancestors).numel()
    assert 1 < n_surviving < k, (
        f"Non-discriminant genealogy for {potential}, {placement} (survivors: {n_surviving}/{k})."
    )

    expected = lam_schedule[-1] * dummy_reward(final_x)
    torch.testing.assert_close(
        lineage_logg,
        expected,
        atol=1e-5,
        rtol=1e-5,
        msg=f"Telescoping invariant failed for {potential}, placement={placement}",
    )