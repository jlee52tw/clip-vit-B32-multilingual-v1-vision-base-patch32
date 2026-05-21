import sys
import time

import openvino as ov
import numpy as np

from pathlib import Path
from PIL import Image

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

def main():
    fp16_vision_model_path = Path("ov_models/clip-vit-base-patch32_vision_static_fully_opt.xml")

    core = ov.Core()
    core.set_property(properties={'CACHE_DIR': './cache'}, device_name='NPU')
    core.set_property(properties={'CACHE_DIR': './cache'}, device_name='GPU')

    compiled_model_v = core.compile_model(fp16_vision_model_path, 'NPU')

    vision_tensor = image_preprocess(Path("data/coco.jpg"))

    start_t = time.time()

    image_embeds = compiled_model_v({'pixel_values': vision_tensor})[0]
    print (image_embeds)
    print (image_embeds.shape)


if __name__ == "__main__":
    sys.exit(main())