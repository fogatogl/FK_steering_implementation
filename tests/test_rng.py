import pytest
import torch

from smc.rng import check_generator, make_generator


def test_check_generator_rejects_cpu_generator_for_cuda():
    with pytest.raises(ValueError, match="make_generator"):
        check_generator(torch.Generator(), "cuda")


def test_check_generator_accepts_none():
    check_generator(None, "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_cuda_generator_feeds_a_cuda_draw():
    assert torch.randn(4, generator=make_generator(0, "cuda"), device="cuda").is_cuda
