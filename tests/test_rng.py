import pytest
import torch

from smc.rng import check_generator, make_generator


def test_check_generator_refuse_un_generateur_cpu_pour_un_calcul_cuda():
    with pytest.raises(ValueError, match="make_generator"):
        check_generator(torch.Generator(), "cuda")


def test_check_generator_accepte_none():
    check_generator(None, "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")
def test_generateur_cuda_alimente_bien_un_tirage_cuda():
    assert torch.randn(4, generator=make_generator(0, "cuda"), device="cuda").is_cuda
