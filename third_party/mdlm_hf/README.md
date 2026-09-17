---
library_name: transformers
license: apache-2.0
language:
- en
datasets:
- Skylion007/openwebtext
metrics:
- perplexity
---

## Using MDLM
To use the pre-trained model for masked language modeling, use the following snippet:
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

# See the `Caduceus` collection page on the hub for list of available models.
tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
model_name = 'kuleshov-group/mdlm-no_flashattn-fp32-owt'
model = AutoModelForMaskedLM.from_pretrained(model_name)
```

For more details, please see our github repository: [MDLM](https://github.com/kuleshov-group/mdlm)

## Model Details
The model, which has a context length of `1024` and is similar in size to GPT2-medium with approximately `130 million` non-embedding parameters,
was trained using a forward diffusion process that generates inputs varying from fully masked to fully unmasked. Its objective is to
reconstruct the original input from these varying levels of masking, outputting logits in the process.
The training regimen comprised one million steps on the OpenWebText corpus, involving the processing of a total of `33 billion` tokens.
Note that this model can run on older GPUs since it uses **regular attention** instead of flash attention and uses `float32` precision.

For more details, please see our paper: [Simple and Effective Masked Diffusion Language Models](http://arxiv.org/abs/2406.07524).



## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->
Please cite our work using the bibtex below:

**BibTeX:**

```
@inproceedings{
    sahoo2024simple,
    title={Simple and Effective Masked Diffusion Language Models},
    author={Subham Sekhar Sahoo and Marianne Arriola and Aaron Gokaslan and Edgar Mariano Marroquin and Alexander M Rush and Yair Schiff and Justin T Chiu and Volodymyr Kuleshov},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=L4uaAR4ArM}
}
```

**APA:**

```
@software{Sahoo_Simple_and_Effective_2024,
author = {Sahoo, Subham Sekhar and Arriola, Marianne and Schiff, Yair and Gokaslan, Aaron and Marroquin, Edgar and Chiu, Justin T and Rush, Alexander and Kuleshov, Volodymyr},
doi = {10.48550/arXiv.2406.07524},
month = jun,
title = {{Simple and Effective Masked Diffusion Language Models}},
version = {arXiv:2406.07524v1},
year = {2024}
}
```

## Model Card Contact
Subham Sekhar Sahoo (subbham@gmail.com)