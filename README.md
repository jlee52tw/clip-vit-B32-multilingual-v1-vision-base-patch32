# applications.ai.openvino.samples.clip

This project is OpenVINO based CLIP for image search.

Please following below steps to get ov models and run inference.  

1. Please download python `3.9`.
2. Following below commands to create virtual envirenment and install required libraries. 
    ```sh
    python -m venv clip_venv
    clip_venv\Scripts\activate
    python -m pip install --upgrade pip wheel setuptools
    pip install -r requirements.txt
    ```
3. Get static text embedding model with offloaded post processing
    ```sh
    python get_static_text_model.py
    ```
    > __Notes__ 
    > In CLIP model, the default setting of image size is `224x224` which is set for OV model and default setting of token length is `77`, now is set to `10` for OV model. 
    > If you want to change the token length for text embedding model, please modify line `64` ~ `77` of file, `get_static_text_model.py`.
    > 
    > * Original code:
    >   ```py
    >   if not fp16_text_model_path.exists():
    >       ov_inputs = {
    >           "input_ids": [1, 10],
    >           "attention_mask": [1, 10],
    >       }
    >       with torch.no_grad():
    >           ov_model = ov.convert_model(
    >               model_t, 
    >               example_input={
    >                   'input_ids': inputs['input_ids'], 
    >                   'attention_mask': inputs['attention_mask']
    >               }, 
    >               input=ov_inputs
    >           )
    >   ```
    > 
    > * Modified:
    >   ```py
    >   token_length = 20 # modify this number.
    >   if not fp16_text_model_path.exists():
    >       ov_inputs = {
    >           "input_ids": [1, token_length],
    >           "attention_mask": [1, token_length],
    >       }
    >       with torch.no_grad():
    >           ov_model = ov.convert_model(
    >               model_t, 
    >               example_input={
    >                   'input_ids': torch.zeros([1, token_length], dtype=torch.int32), 
    >                   'attention_mask': torch.zeros([1, token_length], dtype=torch.int32)
    >               }, 
    >               input=ov_inputs
    >           )
    >   ```
    >

4. Get static vision embedding model with offloaded preprocessing and post processing
    ```sh
    python get_static_vision_model.py
    ```
5. Get openvino tokenizer
    ```sh
    python get_ov_tokenizer.py
    ```
6. Run inference for CLIP
    * Run sample for models include preprocessing and post processing
        ```py
        python infer.py
        ```

    * Get Text embedding
        ```py
        python get_text_embedding.py
        ```
    * Get Vision embedding
        ```py
        python get_vision_embedding.py
        ```
