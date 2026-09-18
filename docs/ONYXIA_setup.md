# Onyxia — bringing an instance back to a working state

*v2, 16/09/2026 — reread when launching an SSP Cloud service*

> Project values live in `scripts/onyxia_bootstrap.sh`: repo `fogatogl/diffusion-models`,
> MinIO bucket `gfogato`. **The bucket is not `$USERNAME`**: inside the service
> `$USERNAME` is `onyxia`; scripts that defaulted to it targeted an empty bucket.

## 0. Short version

The Onyxia form clones the repo by itself (Git tab). One command remains in the VS Code terminal:

```bash
bash ~/work/diffusion-models/scripts/onyxia_bootstrap.sh && source ~/.bashrc
```

If `~/work/diffusion-models` does not exist — volume lost, or Git tab left empty:

```bash
cd ~/work
git clone https://${GIT_PERSONAL_ACCESS_TOKEN}@github.com/fogatogl/diffusion-models.git
bash ~/work/diffusion-models/scripts/onyxia_bootstrap.sh && source ~/.bashrc
```

The bootstrap is **idempotent**: rerun it as often as you like.

## 1. What survives and what dies

An Onyxia service is an ephemeral container; only the persistent volume survives restarts.

| Path | Survives? | Consequence |
|---|---|---|
| `/home/onyxia/work` | **yes**, if persistence is ticked at launch | venv, repo and heavy artefacts live here |
| `~/.bashrc` | no | venv auto-activation must be rewritten each time |
| `~/.local` | no | the Jupyter kernel must be re-registered each time |
| `~/.gitconfig` | no | commit name / e-mail to reset (unless the Git tab is filled) |
| `~/.cache/huggingface` | no | `HF_HOME=~/work/ddpm/hf` (set by the bootstrap) keeps Hub models on the volume; mirrored by `sync_s3.sh` |
| `~/.mc/config.json` | regenerated at launch | the `s3` alias is ready, token valid **7 days** |
| `apt` packages, `/tmp`, everything else | no | install nothing important there |

**Single rule**: nothing important outside `~/work`, nothing heavy outside S3 and Git.

## 2. Once per Datalab account

1. Create a GitHub token at <https://github.com/settings/tokens> — scope `repo`, short expiry (30 days).
2. Store it in the Datalab account, **External services** section. It becomes `$GIT_PERSONAL_ACCESS_TOKEN` in every service.
3. Check that the **Datalab account e-mail is the GitHub one**, otherwise commits are not attached to the GitHub profile.
4. Renew the token when it expires.
5. Once the service form is filled correctly, **save the configuration** (bookmark icon, top right).

Reference: <https://docs.sspcloud.fr/content/version-control.html>

## 3. At each launch: the form

| Tab | Setting | Why |
|---|---|---|
| **Git** | full repo URL, e.g. `https://github.com/<owner>/<repo>` | cloned automatically into the workspace |
| **Init** | public URL of `onyxia_init.sh` (optional if you use the bootstrap) | run **as root** right after start-up |
| **Resources** | GPU **only when needed** | log-weights, resamplers, ESS are pure CPU: do not hold a shared T4 to run `pytest` |
| **Persistence** | on | otherwise the venv is reinstalled every session |
| **Security** | change the service password | it is displayed in clear otherwise |

The init script runs as root, hence the final `chown -R onyxia:users` in `onyxia_init.sh`.

## 4. The bootstrap (`scripts/onyxia_bootstrap.sh`, versioned)

It lives **in the repo**, not on the volume: since the Git tab clones the repo at start-up, the script is always there even after a lost volume. Read it rather than a copy. It does seven things, in order:

1. clones the repo, or `pull --ff-only` if present;
2. sets the git identity **without overwriting** the one Onyxia already set;
3. creates `~/work/.venvs/ddpm` with `--system-site-packages`, installs `requirements-onyxia.txt` and the repo in editable mode;
4. re-registers the Jupyter kernel in `~/.local`;
5. rewrites `~/.bashrc`: `BUCKET`, `REPO`, venv activation;
6. fetches the **two latest checkpoints** (257 MB) from S3 — `FULL_WEIGHTS=1` for the 2 GB of intermediate checkpoints, `NO_S3=1` to skip;
7. prints the torch version and the visible GPU.

Two traps commented there, which cost time:

- `mc ls s3/` **without a bucket name hangs** until timeout — the `stsonly` policy does not allow `ListBuckets`. Always target `s3/gfogato/...` explicitly.
- `particles` 0.4 declares `numpy<2` while its resamplers run fine on 2.x. Without a pin, pip downgrades the whole venv to numpy 1.26 while the image's torch is built against numpy 2. Hence the single `numpy==2.3.*` pin in `requirements-onyxia.txt`.

### Two dependency files, and why

- `requirements.txt` — **pinned and complete, torch included**. The public one: a stranger must be able to reproduce the figures.
- `requirements-onyxia.txt` — **no torch, no torchvision**. The one the bootstrap installs, because the Datalab GPU image already ships a PyTorch paired with its CUDA, and reinstalling it breaks the GPU.

## 5. Sixty-second check

```bash
which python                                  # -> .../work/.venvs/ddpm/bin/python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
mc ls s3/gfogato/ddpm                         # S3 token alive? (never a bare "mc ls s3/")
git -C ~/work/diffusion-models status -sb
cd ~/work/diffusion-models && pytest -q
```

## 6. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `mc` or `aws s3` hangs silently | command run without a bucket (`mc ls s3/`): `ListBuckets` is denied | target `s3/gfogato/...` explicitly |
| Service **red** in "My services", `mc` refuses | MinIO token expired (7 days) | open a new service, or copy the renewal scripts from <https://datalab.sspcloud.fr/account/storage> |
| `git push` asks for a password | repo cloned without a token in the URL | `git remote set-url origin https://${GIT_PERSONAL_ACCESS_TOKEN}@github.com/fogatogl/diffusion-models.git` |
| Commits not attached to the GitHub profile | Datalab e-mail ≠ GitHub e-mail | fix in the account, then `git config --global user.email` |
| `python` points to `/opt/conda/bin/python` | `~/.bashrc` reset | `source ~/.bashrc`, or rerun the bootstrap |
| "Python (ddpm)" kernel missing in VS Code | `~/.local` not persistent | rerun the bootstrap, reload the VS Code window |
| `torch.cuda.is_available()` is `False` | service launched without GPU | check the Resources tab |
| `pull` refused, "uncommitted local work" | changes not pushed at the previous session | see §7 |
| Everything is gone | persistence not ticked at launch | replay from Git + S3; this is exactly why nothing lives elsewhere |

## 7. End of session

An Onyxia instance can die without warning. **Until it is pushed, it does not exist.**

```bash
cd ~/work/diffusion-models
git add -A && git commit -m "..." && git push        # code
bash scripts/sync_s3.sh push                         # weights, samples, heavy results
```

Then a line in `LEARNING.md` if a bug took more than twenty minutes, and only then delete the service.

## 8. What goes where

| Object | Destination | Never |
|---|---|---|
| Code, tests, scripts, regenerable figures | **Git** | — |
| Checkpoints, dataset, samples, heavy W&B runs | **S3** (`~/work/ddpm/...`) | in Git |
| `results/*.json` (small, needed to regenerate figures) | **Git** | — |
| GitHub token, keys | **Datalab account / Vault** | in Git, nor in a notebook |
| venv | `~/work/.venvs/ddpm` | in Git |

## Sources

- SSP Cloud version control: <https://docs.sspcloud.fr/content/version-control.html>
- MinIO storage: <https://docs.sspcloud.fr/content/storage.html>
- Service configuration: <https://docs.sspcloud.fr/content/services-configuration.html>
