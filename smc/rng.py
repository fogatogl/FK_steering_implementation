"""Un seul générateur par expérience, créé sur le device du modèle.

PyTorch refuse d'apparier un générateur et un tenseur de devices différents, et
un `torch.Generator()` nu est un générateur CPU : le piège ne se voit qu'une
fois sur GPU. Aucune fonction de `smc/` ne fabrique de générateur, elle le
reçoit — ou reçoit `None` et tire alors sur le RNG global.
"""
import torch


def make_generator(seed, device="cpu"):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def check_generator(generator, device):
    if generator is None:
        return
    attendu = torch.device(device).type
    obtenu = generator.device.type
    if obtenu != attendu:
        raise ValueError(
            f"générateur sur '{obtenu}' pour un calcul sur '{attendu}'. "
            f"Un seul générateur par expérience, créé avec "
            f"make_generator(seed, device='{attendu}')."
        )
