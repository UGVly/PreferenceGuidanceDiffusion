import torch
import ImageReward as RM
from PIL import Image

class ScoreModel:
    def __init__(self, model_name="ImageReward-v1.0", device="cuda"):
        self.model_name = model_name
        # model_name="ImageReward-v1.0"
        self.model = RM.load(model_name)


    @torch.no_grad()
    def score(self, prompt, img_path):
        if not isinstance(img_path, list):
            img_path = [img_path]
        img_list = [Image.open(path) if isinstance(path, str) else path for path in img_path]
        
        scores = [self.model.score(prompt, img) for img in img_list]
        return scores
    


