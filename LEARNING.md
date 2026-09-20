# Bugs that cost more than twenty minutes, one line each

- 17/09: `load_model` loaded `ckpt["model"]` (raw weights) while the notebook FID used `ckpt["ema"]`: the whole lambda sweep ran on a different model than the FID anchors. Found by rereading `load_eval_model` in the notebook.
- 20/09: the SD venv had no editable install of the project, so `import smc` failed from `scripts/`. Only `sample_fk` imports `smc`, so the two baselines ran a full night and the FK row would have died on its first prompt.
