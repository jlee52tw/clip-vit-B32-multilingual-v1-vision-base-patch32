from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer

hf_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/clip-ViT-B-32-multilingual-v1")
ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)

from pathlib import Path
from openvino import save_model

tokenizer_dir = Path("ov_tokenizer/")
save_model(ov_tokenizer, tokenizer_dir / "openvino_tokenizer.xml")
