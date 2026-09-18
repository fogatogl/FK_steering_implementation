"""Train a CIFAR-10 classifier on images in [-1, 1].

Checkpoints every epoch to --out and resumes if the file exists: an Onyxia
cut costs one epoch. Weights live outside the repo, like the DDPM.

  python -m experiments.train_classifier --arch small --seed 0
  python -m experiments.train_classifier --arch resnet18 --seed 1 ; then --calibrate
  python -m experiments.train_classifier --arch small --seed 0 --check samples/free_seed12345.pt

`--check`: on a .pt of DDPM images, the histogram of predicted classes (should
be ~uniform on the free model, ~all "cat" on the fine-tuned one) and the mean
of p(class | x), the prior mass of the target.

`--dataset celebahq --attr Eyeglasses --res 64`: binary classifier on one CelebA-HQ
attribute, images decoded once to a uint8 tensor at `--res` and cached next to the
dataset. Positives are rare (glasses ~5 %): class-weighted cross-entropy, and the
positive recall is printed with the accuracy.

`--calibrate`: post-hoc temperature (Guo et al. 2017) fitted on the test set,
written into the model's `temperature` buffer. For the evaluator only: the
guide stays raw, otherwise lambda loses its posterior meaning. Accuracy is
unchanged, the NLL is not — and the test NLL is then optimistic.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from smc.classifier import build, load

DDPM = Path("/home/onyxia/work/ddpm")
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
# CelebA attribute order, as in list_attr_celeba.txt.
CELEBA_ATTRS = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
                "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
                "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
                "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
                "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
                "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
                "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
                "Wearing_Necklace", "Wearing_Necktie", "Young"]


def celebahq_cache(res):
    """(uint8 images (N,3,res,res), attributes (N,40) in {0,1}), decoded once."""
    cache = DDPM / "dataset" / f"celeba_hq_{res}.pt"
    if cache.exists():
        d = torch.load(cache)
        return d["images"], d["attrs"]
    from datasets import load_from_disk
    ds = load_from_disk(str(DDPM / "dataset" / "celeba_hq_256")).with_format("torch")
    images, attrs = [], []
    for i in range(0, len(ds), 500):
        b = ds[i:i + 500]
        x = b["image"]
        if x.shape[-1] == 3:  # HWC -> CHW; the torch format may already give CHW
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        images.append(F.interpolate(x, size=res, mode="area").round().to(torch.uint8))
        attrs.append((torch.as_tensor(b["attributes"]) > 0).to(torch.uint8))
        print(f"decoding {i + len(x)}/{len(ds)}", flush=True)
    images, attrs = torch.cat(images), torch.cat(attrs)
    torch.save({"images": images, "attrs": attrs}, cache)
    return images, attrs


class FlipCrop:
    """Random crop (pad res//8) + horizontal flip on a uint8 batch already on the GPU."""
    def __init__(self, res):
        self.res, self.pad = res, res // 8

    def __call__(self, x):
        n = x.shape[0]
        x = F.pad(x, (self.pad,) * 4, mode="reflect")
        i, j = torch.randint(0, 2 * self.pad + 1, (2,))
        x = x[:, :, i:i + self.res, j:j + self.res]
        flip = torch.rand(n, device=x.device) < 0.5
        return torch.where(flip[:, None, None, None], x.flip(-1), x)


def celebahq_data(attr, res, batch, val=3000):
    images, attrs = celebahq_cache(res)
    y = attrs[:, CELEBA_ATTRS.index(attr)].long()
    x = images.float().div(127.5).sub(1)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(x), generator=g)
    tr, va = perm[val:], perm[:val]
    print(f"{attr}: {y.float().mean():.3f} positives  ({len(tr)} train / {val} val at {res}px)")
    return (DataLoader(TensorDataset(x[tr], y[tr]), batch, shuffle=True, drop_last=True),
            DataLoader(TensorDataset(x[va], y[va]), 512))


def data(batch, augment, workers=4):
    to_minus1_1 = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                   v2.Normalize(mean=[.5] * 3, std=[.5] * 3)]
    tf_train = v2.Compose(([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
                           if augment else []) + to_minus1_1)
    tf_test = v2.Compose(to_minus1_1)
    train = CIFAR10(str(DDPM / "dataset"), train=True, download=True, transform=tf_train)
    test = CIFAR10(str(DDPM / "dataset"), train=False, download=True, transform=tf_test)
    return (DataLoader(train, batch, shuffle=True, num_workers=workers, pin_memory=True,
                       drop_last=True),
            DataLoader(test, 512, num_workers=workers, pin_memory=True))


@torch.no_grad()
def accuracy(model, loader, device):
    """(accuracy, recall of class 1) — the recall only means something for a binary head."""
    model.eval()
    correct = total = tp = pos = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
        tp += ((pred == 1) & (y == 1)).sum().item()
        pos += (y == 1).sum().item()
    return correct / total, tp / max(pos, 1)


@torch.no_grad()
def test_logits(model, loader, device):
    model.eval()
    outputs, targets = [], []
    for x, y in loader:
        outputs.append(model.logits(x.to(device)))
        targets.append(y.to(device))
    return torch.cat(outputs), torch.cat(targets)


def loaders(args, augment):
    if args.dataset == "celebahq":
        return celebahq_data(args.attr, args.res, args.batch)
    return data(args.batch, augment)


def calibrate(args):
    model = load(args.out, args.device)
    _, test = loaders(args, False)
    logits, y = test_logits(model, test, args.device)
    before = F.cross_entropy(logits / model.temperature, y).item()

    log_t = torch.zeros((), device=args.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)

    model.temperature.fill_(log_t.exp().item())
    after = F.cross_entropy(logits / model.temperature, y).item()
    ckpt = torch.load(args.out, map_location=args.device, weights_only=False)
    ckpt["model"] = model.state_dict()
    ckpt["config"]["temperature"] = model.temperature.item()
    torch.save(ckpt, args.out)
    print(f"temperature {model.temperature.item():.3f}  test NLL {before:.4f} -> {after:.4f}  "
          f"(accuracy unchanged: {(logits.argmax(1) == y).float().mean():.4f})")
    print(json.dumps({"out": args.out, "temperature": model.temperature.item()}))


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    binary = args.dataset == "celebahq"
    train_loader, test_loader = loaders(args, not args.no_augment)
    model = build(args.arch, 2 if binary else 10, args.res if binary else 32).to(device)
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    # GPU-side augmentation for the tensor dataset; torchvision does it for CIFAR.
    augment = FlipCrop(args.res) if binary and not args.no_augment else None
    weight = None
    if binary:
        ys = train_loader.dataset.tensors[1]
        freq = torch.bincount(ys, minlength=2).float() / len(ys)
        weight = (0.5 / freq).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                          weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(train_loader))

    out = Path(args.out)
    start = 0
    if out.exists():
        ckpt = torch.load(out, map_location=device, weights_only=False)
        if ckpt["config"]["epochs"] != args.epochs:
            raise SystemExit(f"{out} was trained for {ckpt['config']['epochs']} epochs, "
                             f"--epochs is {args.epochs}: the cosine schedule would restart "
                             f"exhausted. Move the file or pick another --out.")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start = ckpt["epoch"] + 1
        print(f"resuming at epoch {start} from {out}")

    config = {"arch": args.arch, "seed": args.seed, "epochs": args.epochs, "batch": args.batch,
              "lr": args.lr, "wd": args.wd, "label_smoothing": args.label_smoothing,
              "augment": not args.no_augment, "input": "[-1,1]",
              "dataset": args.dataset, "attr": args.attr if binary else None,
              "n_classes": 2 if binary else 10, "input_size": args.res if binary else 32}

    for epoch in range(start, args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if augment is not None:
                x = augment(x)
            loss = F.cross_entropy(model(x), y, weight=weight, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()

        acc, recall = accuracy(model, test_loader, device)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {running / len(train_loader):.3f}  "
              f"test {acc:.4f}" + (f"  recall+ {recall:.3f}" if binary else ""), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "test_acc": acc, "config": config}, out)


@torch.no_grad()
def check(args):
    model = load(args.out, args.device)
    x = torch.load(args.check, map_location=args.device)
    if isinstance(x, dict):
        x = x["x"]
    logits = model(x.clamp(-1, 1))
    p = logits.softmax(1)
    pred = logits.argmax(1)
    print(f"{args.check}: {len(x)} images, classifier {args.out}")
    names = CIFAR_CLASSES if p.shape[1] == 10 else [f"not {args.attr}", args.attr]
    for c, name in enumerate(names):
        n = (pred == c).sum().item()
        print(f"  {name:14s} {n:5d}  {'#' * (100 * n // len(x))}")
    print(f"  mean p({names[args.target]} | x) = {p[:, args.target].mean():.4f}   "
          f"argmax fraction = {(pred == args.target).float().mean():.4f}   "
          f"median log p = {p[:, args.target].log().median():+.3f}  "
          f"min = {p[:, args.target].log().min():+.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "celebahq"])
    p.add_argument("--attr", default="Eyeglasses", choices=CELEBA_ATTRS)
    p.add_argument("--res", type=int, default=64, help="classifier resolution for celebahq")
    p.add_argument("--target", type=int, default=3, help="target class for --check")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--check", default=None, help="a .pt of images: histogram of predictions")
    p.add_argument("--calibrate", action="store_true",
                   help="post-hoc temperature on the test set, written to --out (evaluator only)")
    args = p.parse_args()
    if args.out is None:
        tag = f"{args.attr.lower()}{args.res}_" if args.dataset == "celebahq" else ""
        args.out = str(DDPM / "weights" / f"classifier_{tag}{args.arch}_seed{args.seed}.pt")
    if args.dataset == "celebahq" and args.target == 3:
        args.target = 1
    if args.check:
        check(args)
    elif args.calibrate:
        calibrate(args)
    else:
        train(args)
        print(json.dumps({"out": args.out}))


if __name__ == "__main__":
    main()
