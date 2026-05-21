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

class CLIPVisionWrapper(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, pixel_values):
        image_embeds = self.clip.visual_projection(self.clip.vision_model(pixel_values)[1])

        square_tensor = torch.pow(image_embeds, 2)
        sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
        normed_tensor = torch.pow(sum_tensor, 0.5)
        return image_embeds / normed_tensor

def main():
    # load pre-trained model
    # openai/clip-vit-base-patch32
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # load preprocessor for model input
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = precess_input(processor)

    model.config.torchscript = True
    fp16_vision_model_path = Path("ov_models/clip-vit-base-patch32_vision_static_fully_opt.xml")
    model_v = CLIPVisionWrapper(model)
    model_v.eval()

    if not fp16_vision_model_path.exists():
        with torch.no_grad():
            ov_inputs = [1, 3, 224, 224]
            ov_model = ov.convert_model(
                model_v, 
                example_input={'pixel_values': inputs['pixel_values']}, 
                input=ov_inputs)
            
            prep = ov.preprocess.PrePostProcessor(ov_model)
            prep.input(0).tensor().set_layout(ov.Layout("NCHW"))
            prep.input(0).preprocess().scale([255, 255, 255])
            prep.input(0).preprocess().mean([0.48145466, 0.4578275, 0.40821073]).scale([0.26862954, 0.26130258, 0.27577711])

            ov_model = prep.build()
            ov.save_model(ov_model, fp16_vision_model_path)



if __name__ == "__main__":
    sys.exit(main())