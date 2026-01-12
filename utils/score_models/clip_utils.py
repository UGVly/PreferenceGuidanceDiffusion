import hpsv2
import os 
from PIL import Image
import torch

from .open_clip import create_model_and_transforms, get_tokenizer

# python eval_clip.py --data_root ./laion_aes_500 \
#     --jsonl_name data.jsonl \
#     --model_name openai/clip-vit-large-patch14 \
#     --batch_size 32



class ScoreModel():
    
    def __init__(self, device):
        self.device = device
        self.model, preprocess_train, self.preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False, # not sure what this means but seems to work without
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )


        self.tokenizer = get_tokenizer('ViT-H-14')


    def score(self, prompt, img_path):
        # 确保是列表
        if isinstance(img_path, str):
            img_path = [img_path]

        # 批量加载和预处理
        image_tensors = []
        for one_img_path in img_path:
            if isinstance(one_img_path, str):
                one_img = Image.open(one_img_path)
            elif isinstance(one_img_path, Image.Image):
                one_img = one_img_path
            else:
                raise ValueError(f"Unsupported image type: {type(one_img_path)}")

            tensor = self.preprocess_val(one_img)
            image_tensors.append(tensor)

        # 拼成batch
        image_batch = torch.stack(image_tensors, dim=0).to(device=self.device, non_blocking=True)

        # 同一个prompt重复 batch_size 次
        text_batch = self.tokenizer([prompt] * len(img_path)).to(device=self.device, non_blocking=True)

        with torch.no_grad():
            outputs = self.model(image_batch, text_batch)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]

            # (batch, embed) @ (batch, embed).T = (batch, batch)
            logits_per_image = image_features @ text_features.T

            # 取对角线：每张图与自己的prompt的相似度
            clip_scores = torch.diagonal(logits_per_image).cpu().numpy()

        return clip_scores.tolist()
