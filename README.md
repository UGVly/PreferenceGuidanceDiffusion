# Rethinking Preference Alignment for Diffusion Models with Classifier-Free Guidance


## Environment setup
```bash
uv sync
```

## Dataset




## Model Inference

### Inference on Stable Diffusion XL in cPGD method

```bash
bash scripts/infer_sdxl_cpgd.sh
```

### Inference on Stable Diffusion XL in PGD method

```bash
bash scripts/infer_sdxl_pgd.sh
```





## Model Training 

This code is mainly built upon [Diffusion-DPO](https://github.com/SalesforceAIResearch/DiffusionDPO), [Diffusers](https://github.com/huggingface/diffusers).



### Training on Stable Diffusion XL in cPGD method

```bash
bash scripts/train_sdxl_cpgd.sh
```

### Training on Stable Diffusion XL in PGD method

```bash
bash scripts/train_sdxl_pgd.sh
```


