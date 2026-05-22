"""列出 text IR 裡每個 SDPA 節點的 attn_mask 來源運算鏈，找出能讓 NPU 內建編譯器走 fast path 的最佳結構。"""
from pathlib import Path
import openvino as ov

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "ov_models" / "clip-ViT-B-32-multilingual-v1_text_static_opt.xml"

core = ov.Core()
m = core.read_model(SRC)

for node in m.get_ordered_ops():
    if node.get_type_name() != "ScaledDotProductAttention":
        continue
    print(f"\n=== SDPA: {node.get_friendly_name()} ===")
    print(f"  Q: {node.input(0).get_source_output().get_element_type()} {node.input(0).get_source_output().get_partial_shape()}")
    print(f"  K: {node.input(1).get_source_output().get_element_type()} {node.input(1).get_source_output().get_partial_shape()}")
    print(f"  V: {node.input(2).get_source_output().get_element_type()} {node.input(2).get_source_output().get_partial_shape()}")
    print(f"  mask: {node.input(3).get_source_output().get_element_type()} {node.input(3).get_source_output().get_partial_shape()}")
    if len(node.inputs()) >= 5:
        print(f"  scale: {node.input(4).get_source_output().get_element_type()} {node.input(4).get_source_output().get_partial_shape()}")
    # walk up 6 levels from mask source
    cur = node.input(3).get_source_output().get_node()
    print(f"  mask source chain:")
    seen = set()
    for depth in range(8):
        if cur.get_friendly_name() in seen:
            break
        seen.add(cur.get_friendly_name())
        ins = [f"{i.get_source_output().get_element_type()}{list(i.get_source_output().get_partial_shape())}" for i in cur.inputs()]
        print(f"    [{depth}] {cur.get_type_name():22s}  in: {ins}")
        if len(cur.inputs()) == 0:
            break
        cur = cur.input(0).get_source_output().get_node()
    break  # 只列第一個就夠了，6 個都是同結構
