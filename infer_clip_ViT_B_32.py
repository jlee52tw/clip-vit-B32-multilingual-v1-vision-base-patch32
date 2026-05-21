import sys
import time
import requests

import openvino as ov
import openvino_tokenizers

import numpy as np

from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy.special import softmax

def image_preprocess(image_name):
    image = Image.open(image_name).convert("RGB")

    resize_size = 224

    orig_width, orig_height = image.size 
    if orig_height >= orig_width:
        resized_width = resize_size
        ratio = resized_width / orig_width
        resized_height = int(orig_height * ratio)
    else:
        resized_height = resize_size
        ratio = resized_height / orig_height
        resized_width = int(orig_width * ratio)

    resized_image = image.resize((resized_width, resized_height), resample=Image.Resampling.BICUBIC)

    crop_height = 224
    crop_width = 224

    top = (resized_height - crop_height) // 2
    bottom = top + crop_height
    left = (resized_width - crop_width) // 2
    right = left + crop_width

    cropped_image = np.array(resized_image)[top: bottom, left: right, :]
    input_image = np.transpose(cropped_image, (2, 0, 1))
    
    input_tensor = np.expand_dims(input_image, 0)

    return input_tensor

def text_preprocess(tokenizer, input_text, input_size=None):
    if type(input_text) == str:
        token = tokenizer([input_text])
    elif type(input_text) == list:
        token = tokenizer(input_text)

    if (input_size):
        input_ids = token["input_ids"]
        attention_mask = token["attention_mask"]

        token_length = input_ids.shape[-1]
        if (input_size > token_length):
            appened_data = input_size - token_length
            return {
                "input_ids": np.pad(input_ids, ((0,0),(0,appened_data)), mode='constant', constant_values=input_ids[0][-1]),
                "attention_mask": np.pad(attention_mask, ((0,0),(0,appened_data)), mode='constant', constant_values=0)
            }
    return token

def main():
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

    fp16_vision_model_path = Path("ov_models/clip-vit-base-patch32_vision_static_fully_opt.xml")
    fp16_text_model_path = Path("ov_models/clip-ViT-B-32-multilingual-v1_text_static_opt.xml")
    ov_tokenizer_model_path = Path("ov_tokenizer/openvino_tokenizer.xml")

    core = ov.Core()
    core.set_property(properties={'CACHE_DIR': './cache'}, device_name='NPU')
    core.set_property(properties={'CACHE_DIR': './cache'}, device_name='GPU')

    compiled_model_t = core.compile_model(fp16_text_model_path, 'NPU')
    input_size = compiled_model_t.input(0).shape[1]
    compiled_model_v = core.compile_model(fp16_vision_model_path, 'NPU')
    compiled_model_tokenizer = core.compile_model(ov_tokenizer_model_path, 'CPU')
   
    vision_tensor = image_preprocess(Path("data/coco.jpg"))
    text_tensor = text_preprocess(compiled_model_tokenizer, [f"This is a photo of a {label}" for label in input_labels], input_size=input_size)

    start_t = time.time()

    text_embeds = []
    for i in range(len(input_labels)):
        text_embeds.append(compiled_model_t({'input_ids': np.expand_dims(text_tensor['input_ids'][i], axis=0), 'attention_mask': np.expand_dims(text_tensor['attention_mask'][i], axis=0)})[0])
    image_embeds = compiled_model_v({'pixel_values': vision_tensor})[0]

    logit_scale = 100
    logits_per_text = np.zeros([10])
    for i in range(len(input_labels)):
        logits_per_text[i] = np.matmul(text_embeds[i], np.transpose(image_embeds)) * logit_scale
    
    end_t = time.time()
    print ("Run inference time: ", (end_t - start_t))
    print (logits_per_text)
    print (input_labels[np.argmax(logits_per_text)])


if __name__ == "__main__":
    sys.exit(main())