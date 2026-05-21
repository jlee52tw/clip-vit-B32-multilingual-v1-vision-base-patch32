import sys
import time

import openvino as ov
import openvino_tokenizers

import numpy as np

from pathlib import Path

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
    text = "This is a photo of a cat"
    fp16_text_model_path = Path("ov_models/clip-ViT-B-32-multilingual-v1_text_static_opt.xml")
    ov_tokenizer_model_path = Path("ov_tokenizer/openvino_tokenizer.xml")

    core = ov.Core()
    core.set_property(properties={'CACHE_DIR': './cache'}, device_name='NPU')
    core.set_property(properties={'CACHE_DIR': './cache'}, device_name='GPU')

    compiled_model_t = core.compile_model(fp16_text_model_path, 'NPU')
    input_size = compiled_model_t.input(0).shape[1]
    compiled_model_tokenizer = core.compile_model(ov_tokenizer_model_path, 'CPU')
   
    text_tensor = text_preprocess(compiled_model_tokenizer, text, input_size=input_size)

    start_t = time.time()

    text_embeds = compiled_model_t(text_tensor)[0]

    end_t = time.time()
    print ("Run inference time: ", (end_t - start_t))
    print (text_embeds)
    print (text_embeds.shape)

if __name__ == "__main__":
    sys.exit(main())