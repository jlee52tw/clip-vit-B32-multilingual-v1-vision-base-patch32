import sys
import torch
import requests

import openvino as ov
import numpy as np

from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def precess_input(clip_processor):
    sample_path = Path("data/coco.jpg")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg")

    with sample_path.open("wb") as f:
        f.write(r.content)

    image = Image.open(sample_path)

    input_labels = [
        "cat",
        "dog",
        "wolf",
        "tiger",
        "man",
        "horse",
        "frog",
        "tree",
        "house",
        "computer",
    ]
    text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

    return clip_processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)

class CLIPTextWrapper(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, input_ids, attention_mask):
        text_embeds = self.clip.text_projection(self.clip.text_model(input_ids, attention_mask)[1])

        square_tensor = torch.pow(text_embeds, 2)
        sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
        normed_tensor = torch.pow(sum_tensor, 0.5)
        return text_embeds / normed_tensor

def main():
    # load pre-trained model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    # load preprocessor for model input
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    inputs = precess_input(processor)

    model.config.torchscript = True
    fp16_text_model_path = Path("ov_models/clip-vit-base-patch16_text_static_opt.xml")
    model_t = CLIPTextWrapper(model)
    model_t.eval()

    if not fp16_text_model_path.exists():
        ov_inputs = {
            "input_ids": [1, 10],
            "attention_mask": [1, 10],
        }
        with torch.no_grad():
            ov_model = ov.convert_model(
                model_t, 
                example_input={
                    'input_ids': inputs['input_ids'], 
                    'attention_mask': inputs['attention_mask']
                }, 
                input=ov_inputs
            )
            ov.save_model(ov_model, fp16_text_model_path)

if __name__ == "__main__":
    sys.exit(main())