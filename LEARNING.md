# Bugs that cost more than twenty minutes, one line each

- 17/09: `load_model` loaded `ckpt["model"]` (raw weights) while the notebook FID used `ckpt["ema"]`: the whole lambda sweep ran on a different model than the FID anchors. Found by rereading `load_eval_model` in the notebook.
