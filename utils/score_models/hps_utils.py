import os 
import requests
from clint.textui import progress
from PIL import Image
import torch

from .open_clip import create_model_and_transforms, get_tokenizer



class ScoreModel():
    
    def __init__(self, device):
        self.device = device 
        model, preprocess_train, self.preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        root_path = "/home/jiangzhou/CODE/MyDiffusers/data/cache/reward_models"

        checkpoint_path = os.path.join(root_path, 'HPS_v2_compressed.pt')
        # if not os.path.exists(checkpoint_path):
        #     print('Downloading HPS_v2_compressed.pt ...')
        #     url = 'https://hf-mirror.com/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt'
        #     r = requests.get(url, stream=True)
        #     with open(os.path.join(root_path, 'HPS_v2_compressed.pt'), 'wb') as HPSv2:
        #         total_length = int(r.headers.get('content-length'))
        #         for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
        #             if chunk:
        #                 HPSv2.write(chunk)
        #                 HPSv2.flush()
        #     print('Download HPS_v2_compressed.pt to {} sucessfully.'.format(root_path+'/'))


        # print('Loading model ...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer('ViT-H-14')
        model = model.to(device)
        model.eval()
        self.model = model
        print('Loading model successfully!')

    # def score(self, prompt, img_path):


    #     if not isinstance(img_path, list):
    #         img_path = [img_path]
    #     result = []
    #     for one_img_path in img_path:
    #         # Load your image and prompt
    #         with torch.no_grad():
    #             # Process the image
    #             if isinstance(one_img_path, str):
    #                 one_img_path = Image.open(one_img_path)
    #             image = self.preprocess_val(one_img_path).unsqueeze(0).to(device=self.device, non_blocking=True)
                
    #             # Process the prompt
    #             text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
    #             # Calculate the HPS
               
    #             outputs = self.model(image, text)
    #             image_features, text_features = outputs["image_features"], outputs["text_features"]
    #             logits_per_image = image_features @ text_features.T

    #             hps_score = torch.diagonal(logits_per_image).cpu().numpy()
    #         result.append(hps_score[0])    
    #     return result

    def score(self, prompt, img_path):
        # 逐个处理每张图片
        if not isinstance(img_path, list):
            img_path = [img_path]
        result = []
        for one_img_path in img_path:
            if isinstance(one_img_path, str):
                img_obj = Image.open(one_img_path)
            else:
                img_obj = one_img_path
            image = self.preprocess_val(img_obj).unsqueeze(0).to(device=self.device, non_blocking=True)
            text = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
            with torch.no_grad():
                outputs = self.model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append(hps_score[0])
        # print(result)
        # print("--------------------------------")
        return result
