from pgd_diffusers.pipeline_stable_diffusion_xl_pgd import StableDiffusionXLPipeline
from diffusers import UNet2DConditionModel
import torch
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--win_unet_path", type=str, default=None)
    parser.add_argument("--lose_unet_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./tmp.png")
    parser.add_argument("--pgd_scale", type=float, default=10)
    return parser.parse_args()

args = get_args()


pipe = StableDiffusionXLPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")

win_unet  = UNet2DConditionModel.from_pretrained(args.win_unet_path,  subfolder="unet", torch_dtype=torch.float16)
lose_unet = UNet2DConditionModel.from_pretrained(args.lose_unet_path, subfolder="unet", torch_dtype=torch.float16)

pipe.win_unet  = win_unet.to("cuda")
pipe.lose_unet = lose_unet.to("cuda")


pipe.do_cpgd_inference = True
pipe.cpgd_scale = args.cpgd_scale


gs = 7.5
generator = torch.Generator().manual_seed(42)
image = pipe(args.prompt, generator=generator, guidance_scale=gs).images[0]
image.save(args.save_path)