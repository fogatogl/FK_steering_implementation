"""Train a CIFAR-10 classifier on images in [-1, 1].

Checkpoints every epoch to --out and resumes if the file exists: an Onyxia
cut costs one epoch. Weights live outside the repo, like the DDPM.

  python -m experiments.train_classifier --arch small --seed 0
  python -m experiments.train_classifier --arch resnet18 --seed 1 ; then --calibrate
  python -m experiments.train_classifier --arch small --seed 0 --check samples/free_seed12345.pt

`--check`: on a .pt of DDPM images, the histogram of predicted classes (should
be ~uniform on the free model, ~all "cat" on the fine-tuned one) and the mean
of p(class | x), the prior mass of the target.

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
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

from smc.classifier import build, load

DDPM = Path("/home/onyxia/work/ddpm")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


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
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


@torch.no_grad()
def test_logits(model, loader, device):
    model.eval()
    outputs, targets = [], []
    for x, y in loader:
        outputs.append(model.logits(x.to(device)))
        targets.append(y.to(device))
    return torch.cat(outputs), torch.cat(targets)


def calibrate(args):
    model = load(args.out, args.device)
    _, test = data(args.batch, False)
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


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    train_loader, test_loader = data(args.batch, not args.no_augment)
    model = build(args.arch).to(device)
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                          weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(train_loader))

    out = Path(args.out)
    start = 0
    if out.exists():
        ckpt = torch.load(out, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start = ckpt["epoch"] + 1
        print(f"resuming at epoch {start} from {out}")

    config = {"arch": args.arch, "seed": args.seed, "epochs": args.epochs, "batch": args.batch,
              "lr": args.lr, "wd": args.wd, "label_smoothing": args.label_smoothing,
              "augment": not args.no_augment, "input": "[-1,1]"}

    for epoch in range(start, args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()

        acc = accuracy(model, test_loader, device)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {running / len(train_loader):.3f}  "
              f"test {acc:.4f}", flush=True)
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
    for c, name in enumerate(CLASSES):
        n = (pred == c).sum().item()
        print(f"  {name:10s} {n:5d}  {'#' * (100 * n // len(x))}")
    print(f"  mean p({CLASSES[args.target]} | x) = {p[:, args.target].mean():.4f}   "
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
    p.add_argument("--target", type=int, default=3, help="target class for --check")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--check", default=None, help="a .pt of images: histogram of predictions")
    p.add_argument("--calibrate", action="store_true",
                   help="post-hoc temperature on the test set, written to --out (evaluator only)")
    args = p.parse_args()
    if args.out is None:
        args.out = str(DDPM / "weights" / f"classifier_{args.arch}_seed{args.seed}.pt")
    if args.check:
        check(args)
    elif args.calibrate:
        calibrate(args)
    else:
        train(args)
        print(json.dumps({"out": args.out}))


if __name__ == "__main__":
    main()
