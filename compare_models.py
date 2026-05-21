import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import openvino as ov

def load_models():
    """
    Load both the PyTorch CLIP model and the OpenVINO static model
    """
    print("Loading models...")
    
    # Load the original PyTorch model
    print("Loading PyTorch CLIP model...")
    pt_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    
    # Load the OpenVINO converted model
    print("Loading OpenVINO static model...")
    core = ov.Core()
    
    # Add caching to improve performance by avoiding recompilation
    core.set_property(properties={'CACHE_DIR': './cache'}, device_name='NPU')
    core.set_property(properties={'CACHE_DIR': './cache'}, device_name='CPU')
    
    model_path = Path("ov_models/clip-ViT-B-32-multilingual-v1_text_static_opt.xml")
    ov_model = core.read_model(model_path)
    
    # Try to compile for NPU first, fallback to CPU if NPU is not available
    try:
        print("Compiling for NPU...")
        # Add performance hints for better throughput
        compiled_model = core.compile_model(ov_model, "NPU", config={"PERFORMANCE_HINT": "LATENCY"})
        device = "NPU"
    except Exception as e:
        print(f"NPU compilation failed: {e}")
        print("Falling back to CPU...")
        compiled_model = core.compile_model(ov_model, "CPU")
        device = "CPU"
    
    print(f"OpenVINO model compiled for {device}")
    
    return pt_model, compiled_model, device

def get_pt_embedding(model, texts):
    """
    Get embeddings from the PyTorch model
    """
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def get_ov_embedding(model, texts, pt_model):
    """
    Get embeddings from the OpenVINO model (static batch size 1)
    """
    seq_length = 128  # Should match the static shape used during conversion
    all_embeddings = []
    for text in texts:
        # Tokenize single text
        tokens = pt_model.tokenize([text])
        input_ids = tokens['input_ids'].numpy()
        attention_mask = tokens['attention_mask'].numpy()
        # Pad or truncate to static length
        padded_input_ids = np.zeros((1, seq_length), dtype=np.int64)
        padded_attention_mask = np.zeros((1, seq_length), dtype=np.int64)
        actual_length = min(input_ids[0].shape[0], seq_length)
        padded_input_ids[0, :actual_length] = input_ids[0][:actual_length]
        padded_attention_mask[0, :actual_length] = attention_mask[0][:actual_length]
        # Run inference
        outputs = model([padded_input_ids, padded_attention_mask])
        embedding = outputs[0][0]  # shape: (512,)
        all_embeddings.append(embedding)
    # Stack all embeddings into a tensor
    return torch.tensor(np.stack(all_embeddings))

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between embeddings
    """
    return cos_sim(embedding1, embedding2).item()

def simple_test():
    """
    Simple test with a few text inputs
    """
    pt_model, ov_model, device = load_models()
    
    # Test texts
    test_texts = [
        "A photo of a cat",
        "Eine Katze auf dem Dach",  # German: "A cat on the roof"
        "Un gato sentado en una ventana",  # Spanish: "A cat sitting on a window"
        "Un chien courant dans un champ",  # French: "A dog running in a field"
        "A computer on a desk",
        "Eine schöne Landschaft mit Bergen",  # German: "A beautiful landscape with mountains"
        "The sunset over the ocean"
    ]
    
    # Perform warmup passes to reduce initialization overhead
    print("\nPerforming warmup passes...")
    _ = get_pt_embedding(pt_model, ["A warmup text"])
    _ = get_ov_embedding(ov_model, ["A warmup text"], pt_model)
    print("Warmup complete")
    
    # Compute embeddings for each text using both models
    print("\nComputing embeddings and similarities...")
    
    # Time PT model
    start_time = time.time()
    pt_embeddings = get_pt_embedding(pt_model, test_texts)
    pt_time = time.time() - start_time
    print(f"PyTorch CPU model took {pt_time:.4f} seconds")
    
    # Time OV model
    start_time = time.time()
    ov_embeddings = get_ov_embedding(ov_model, test_texts, pt_model)
    ov_time = time.time() - start_time
    print(f"OpenVINO {device} model took {ov_time:.4f} seconds")
    
    speedup = pt_time / ov_time
    print(f"Speedup: {speedup:.2f}x")
    
    # Check embedding shapes
    print(f"\nPyTorch embedding shape: {pt_embeddings.shape}")
    print(f"OpenVINO embedding shape: {ov_embeddings.shape}")
    
    # Compute embedding differences
    l2_diffs = torch.norm(pt_embeddings - ov_embeddings, dim=1)
    cos_sims = torch.diagonal(cos_sim(pt_embeddings, ov_embeddings))
    
    print("\nEmbedding comparison between PT and OV models:")
    for i, text in enumerate(test_texts):
        print(f"Text: '{text}' - L2 diff: {l2_diffs[i]:.6f}, Cosine sim: {cos_sims[i]:.6f}")
    
    print(f"\nAverage L2 difference: {l2_diffs.mean():.6f}")
    print(f"Average cosine similarity: {cos_sims.mean():.6f}")
    
    # Compute similarities between texts
    print("\nText similarities matrix (PyTorch model):")
    pt_sim_matrix = cos_sim(pt_embeddings, pt_embeddings).cpu().numpy()
    
    print("\nText similarities matrix (OpenVINO model):")
    ov_sim_matrix = cos_sim(ov_embeddings, ov_embeddings).cpu().numpy()
    
    # Compute difference between similarity matrices
    sim_matrix_diff = np.abs(pt_sim_matrix - ov_sim_matrix)
    print(f"\nAverage similarity matrix difference: {sim_matrix_diff.mean():.6f}")
    print(f"Max similarity matrix difference: {sim_matrix_diff.max():.6f}")
    
    return test_texts, pt_sim_matrix, ov_sim_matrix

def plot_similarity_matrices(texts, pt_sim_matrix, ov_sim_matrix):
    """
    Plot similarity matrices for visual comparison
    """
    short_texts = [t[:20] + "..." if len(t) > 20 else t for t in texts]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot PyTorch similarity matrix
    im1 = ax1.imshow(pt_sim_matrix, vmin=0, vmax=1, cmap='viridis')
    ax1.set_title("PyTorch Similarity Matrix")
    ax1.set_xticks(range(len(short_texts)))
    ax1.set_yticks(range(len(short_texts)))
    ax1.set_xticklabels(short_texts, rotation=45, ha='right')
    ax1.set_yticklabels(short_texts)
    
    # Plot OpenVINO similarity matrix
    im2 = ax2.imshow(ov_sim_matrix, vmin=0, vmax=1, cmap='viridis')
    ax2.set_title("OpenVINO Similarity Matrix")
    ax2.set_xticks(range(len(short_texts)))
    ax2.set_yticks(range(len(short_texts)))
    ax2.set_xticklabels(short_texts, rotation=45, ha='right')
    ax2.set_yticklabels(short_texts)
    
    # Plot difference matrix
    diff_matrix = np.abs(pt_sim_matrix - ov_sim_matrix)
    im3 = ax3.imshow(diff_matrix, cmap='hot', vmin=0, vmax=diff_matrix.max() * 1.1)
    ax3.set_title("Difference Matrix")
    ax3.set_xticks(range(len(short_texts)))
    ax3.set_yticks(range(len(short_texts)))
    ax3.set_xticklabels(short_texts, rotation=45, ha='right')
    ax3.set_yticklabels(short_texts)
    
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("similarity_comparison.png")
    print("Similarity matrices visualized and saved to 'similarity_comparison.png'")

def benchmark_dataset():
    """
    Benchmark both models on a small dataset
    """
    # Load the models
    pt_model, ov_model, device = load_models()
    
    # Load a test dataset - using a simple set of phrases in different categories
    categories = {
        "animals": [
            "a photo of a dog", "a picture of a cat", "an image of a horse", "a photo of an elephant",
            "un chien noir", "eine schwarze Katze", "un elefante grande", "un caballo rápido"
        ],
        "landscapes": [
            "a beautiful sunset", "mountains with snow", "a beach with palm trees", "a forest with tall trees",
            "un beau coucher de soleil", "montañas con nieve", "ein Strand mit Palmen", "une forêt dense"
        ],
        "objects": [
            "a red car", "a modern laptop", "a wooden table", "a ceramic cup",
            "une voiture rouge", "ein moderner Laptop", "una mesa de madera", "una taza de cerámica"
        ],
        "food": [
            "a delicious pizza", "fresh fruit salad", "chocolate cake", "a glass of wine",
            "une délicieuse pizza", "ensalada de frutas frescas", "ein Schokoladenkuchen", "un vaso de vino"
        ]
    }
    
    # Flatten the dataset into a single list
    all_texts = []
    labels = []
    for category, texts in categories.items():
        all_texts.extend(texts)
        labels.extend([category] * len(texts))
    
    # Perform warmup passes to reduce initialization overhead
    print("\nPerforming warmup passes...")
    _ = get_pt_embedding(pt_model, ["A warmup text"])
    _ = get_ov_embedding(ov_model, ["A warmup text"], pt_model)
    print("Warmup complete")
    
    print(f"\nBenchmarking with dataset: {len(all_texts)} texts across {len(categories)} categories")
    
    # Benchmark PyTorch model
    start_time = time.time()
    pt_embeddings = get_pt_embedding(pt_model, all_texts)
    pt_time = time.time() - start_time
    print(f"PyTorch CPU model took {pt_time:.4f} seconds (avg: {pt_time/len(all_texts):.4f}s per text)")
    
    # Benchmark OpenVINO model
    start_time = time.time()
    ov_embeddings = get_ov_embedding(ov_model, all_texts, pt_model)
    ov_time = time.time() - start_time
    print(f"OpenVINO {device} model took {ov_time:.4f} seconds (avg: {ov_time/len(all_texts):.4f}s per text)")
    
    speedup = pt_time / ov_time
    print(f"Speedup: {speedup:.2f}x")
    
    # Compute embedding differences
    l2_diffs = torch.norm(pt_embeddings - ov_embeddings, dim=1)
    cos_sims = torch.diagonal(cos_sim(pt_embeddings, ov_embeddings))
    
    print(f"\nAverage L2 difference: {l2_diffs.mean():.6f}")
    print(f"Average cosine similarity: {cos_sims.mean():.6f}")
    
    # Evaluate accuracy by checking if both models group texts in the same categories
    print("\nChecking if models group text by categories correctly:")
    
    # Create category centers from PyTorch embeddings
    category_centers = {}
    for category in categories:
        category_texts = [text for i, text in enumerate(all_texts) if labels[i] == category]
        indices = [i for i, l in enumerate(labels) if l == category]
        category_centers[category] = torch.mean(pt_embeddings[indices], dim=0)
    
    # Check if each text is closest to its category center
    pt_correct = 0
    ov_correct = 0
    
    for i, text in enumerate(all_texts):
        true_label = labels[i]
        
        # PyTorch model
        pt_similarities = {cat: cos_sim(pt_embeddings[i].unsqueeze(0), center.unsqueeze(0)).item() 
                         for cat, center in category_centers.items()}
        pt_predicted = max(pt_similarities, key=pt_similarities.get)
        if pt_predicted == true_label:
            pt_correct += 1
        
        # OpenVINO model
        ov_similarities = {cat: cos_sim(ov_embeddings[i].unsqueeze(0), center.unsqueeze(0)).item() 
                         for cat, center in category_centers.items()}
        ov_predicted = max(ov_similarities, key=ov_similarities.get)
        if ov_predicted == true_label:
            ov_correct += 1
    
    pt_accuracy = pt_correct / len(all_texts)
    ov_accuracy = ov_correct / len(all_texts)
    
    print(f"PyTorch model category accuracy: {pt_accuracy:.2%}")
    print(f"OpenVINO model category accuracy: {ov_accuracy:.2%}")
    
    return pt_time, ov_time, pt_accuracy, ov_accuracy

def main():
    print("===== Model Comparison: PyTorch CPU vs OpenVINO NPU =====")
    
    try:
        # Run simple tests with individual texts
        print("\n----- Simple Text Comparison Test -----")
        texts, pt_sim, ov_sim = simple_test()
        plot_similarity_matrices(texts, pt_sim, ov_sim)
        
        # Run dataset benchmark
        print("\n----- Dataset Benchmark -----")
        pt_time, ov_time, pt_acc, ov_acc = benchmark_dataset()
        
        # Print summary
        print("\n===== Summary =====")
        print(f"PyTorch CPU time: {pt_time:.4f}s, Accuracy: {pt_acc:.2%}")
        print(f"OpenVINO NPU time: {ov_time:.4f}s, Accuracy: {ov_acc:.2%}")
        print(f"Speedup: {pt_time/ov_time:.2f}x")
        print(f"Accuracy difference: {abs(pt_acc - ov_acc):.2%}")
        
    except Exception as e:
        print(f"Error in comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
