import sys
import torch
import requests

import openvino as ov
import numpy as np

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer

def process_input(model):
    sample_path = Path("data/coco.jpg")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg")

    with sample_path.open("wb") as f:
        f.write(r.content)

    image = Image.open(sample_path)
    print(f"[DEBUG] Sample image loaded from {sample_path}, size: {image.size}")

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
    print(f"[DEBUG] Created {len(text_descriptions)} text descriptions")
    
    # Tokenize the text using the model's tokenizer
    tokens = model.tokenize(text_descriptions)
    
    # Add debug information about token shapes
    print(f"[DEBUG] Token shapes after tokenization:")
    print(f"[DEBUG] - input_ids shape: {tokens['input_ids'].shape}")
    print(f"[DEBUG] - attention_mask shape: {tokens['attention_mask'].shape}")
    
    # Print sample tokens for comparison
    print(f"[DEBUG] Sample tokens for first text: {tokens['input_ids'][0][:10]}")
    
    # Fix tensor conversion warnings by using detach().clone()
    return {
        'input_ids': tokens['input_ids'].detach().clone() if torch.is_tensor(tokens['input_ids']) else torch.tensor(tokens['input_ids']),
        'attention_mask': tokens['attention_mask'].detach().clone() if torch.is_tensor(tokens['attention_mask']) else torch.tensor(tokens['attention_mask'])
    }

class CLIPTextWrapperForSentenceTransformer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        print(f"[DEBUG] Initialized CLIPTextWrapper with model type: {type(model)}")
    
    def forward(self, input_ids, attention_mask):
        print(f"[DEBUG] Forward pass with input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
        
        # Create unique identifier for this pass to track execution path
        import uuid
        pass_id = str(uuid.uuid4())[:8]
        print(f"[DEBUG] Starting forward pass {pass_id}")
        
        # Create features dictionary as expected by SentenceTransformer
        features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        # Use the encode method directly which handles the pipeline properly
        # and returns normalized embeddings
        with torch.no_grad():
            # Process only the text part by accessing the first module (transformer)
            transformer = self.model._modules['0']
            print(f"[DEBUG] Pass {pass_id}: Using transformer module: {type(transformer)}")
            
            # Get output from transformer
            transformer_output = transformer(features)
            print(f"[DEBUG] Pass {pass_id}: Transformer outputs: {list(transformer_output.keys())}")
            
            # Apply pooling if available (this is usually the second module)
            if '1' in self.model._modules:
                pooling = self.model._modules['1']
                print(f"[DEBUG] Pass {pass_id}: Using pooling module: {type(pooling)}")
                pooled_output = pooling(transformer_output)
                print(f"[DEBUG] Pass {pass_id}: Pooling outputs: {list(pooled_output.keys())}")
            else:
                # If no pooling module, use last_hidden_state's first token (CLS)
                print(f"[DEBUG] Pass {pass_id}: No pooling module found, using first token")
                pooled_output = {'sentence_embedding': transformer_output['token_embeddings'][:, 0]}
            
            # Get the embeddings
            embeddings = pooled_output['sentence_embedding']
            print(f"[DEBUG] Pass {pass_id}: Embedding shape before dense projection: {embeddings.shape}")
            
            # Apply dense projection (module 2) if available
            if '2' in self.model._modules:
                dense = self.model._modules['2']
                print(f"[DEBUG] Pass {pass_id}: Using dense module: {type(dense)}")
                
                # Wrap embedding in a dictionary as expected by Dense module
                features_dict = {'sentence_embedding': embeddings}
                features_dict = dense(features_dict)
                embeddings = features_dict['sentence_embedding']
                print(f"[DEBUG] Pass {pass_id}: Embedding shape after dense projection: {embeddings.shape}")
            
            # CLIP-style normalization (matches the original implementation)
            square_tensor = torch.pow(embeddings, 2)
            sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
            normed_tensor = torch.pow(sum_tensor, 0.5)
            normalized_embeddings = embeddings / normed_tensor
            
            print(f"[DEBUG] Pass {pass_id}: Normalized shape: {normalized_embeddings.shape}")
            print(f"[DEBUG] Completed forward pass {pass_id}")
            
            return normalized_embeddings

def main():
    print("[DEBUG] Starting model conversion process for clip-ViT-B-32-multilingual-v1")
    
    # Load the SentenceTransformer multilingual CLIP model
    model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    print(f"[DEBUG] Loaded model: {model}")
    
    # Process input and get tokenized text
    inputs = process_input(model)
    
    # Create model wrapper
    model_t = CLIPTextWrapperForSentenceTransformer(model)
    model_t.eval()
    print("[DEBUG] Model set to evaluation mode")
    
    # Check a sample embedding to verify it looks right
    print("[DEBUG] Generating a test embedding to verify processing pipeline...")
    with torch.no_grad():
        sample_embedding = model_t(inputs['input_ids'], inputs['attention_mask'])
        print(f"[DEBUG] Sample embedding shape: {sample_embedding.shape}")
        print(f"[DEBUG] Sample embedding norm: {torch.norm(sample_embedding[0]).item():.6f}")
        print(f"[DEBUG] First few values: {sample_embedding[0, :5]}")
    
    fp16_text_model_path = Path("ov_models/clip-ViT-B-32-multilingual-v1_text_static_opt.xml")
    fp16_text_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not fp16_text_model_path.exists():
        # Define static input shapes
        # [1, N] means batch size of 1 and sequence length of N tokens
        # Use a fixed sequence length of 128 instead of the dynamic length
        seq_length = 128  # Changed from 10 to 128 for better handling of longer texts
        print(f"[DEBUG] Using sequence length of {seq_length} for static model")
        
        ov_inputs = {
            "input_ids": [1, seq_length],
            "attention_mask": [1, seq_length],
        }
        print(f"[DEBUG] Static input shapes: {ov_inputs}")
        
        with torch.no_grad():
            # Print model structure to debug
            print("[DEBUG] Model structure:")
            for name, module in model._modules.items():
                print(f"[DEBUG] Module {name}: {type(module)}")
            
            try:
                print("[DEBUG] Starting model conversion with example_input...")
                ov_model = ov.convert_model(
                    model_t, 
                    example_input={
                        'input_ids': inputs['input_ids'], 
                        'attention_mask': inputs['attention_mask']
                    }, 
                    input=ov_inputs
                )
                print("[DEBUG] Model conversion successful")
                ov.save_model(ov_model, fp16_text_model_path)
                print(f"[DEBUG] Model successfully saved to {fp16_text_model_path}")
            except Exception as e:
                print(f"[DEBUG] Error during conversion: {e}")
                # Try alternative conversion method without example_input
                print("[DEBUG] Trying alternative conversion method...")
                ov_model = ov.convert_model(model_t, input=ov_inputs)
                ov.save_model(ov_model, fp16_text_model_path)
                print(f"[DEBUG] Model successfully saved to {fp16_text_model_path}")
    else:
        print(f"[DEBUG] Model already exists at {fp16_text_model_path}")

if __name__ == "__main__":
    sys.exit(main())