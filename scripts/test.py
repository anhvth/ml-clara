import sys
import os
import torch

# Add project root to path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

from openrlhf.models.modeling_clara import CLaRa

# Configuration
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

unirag = CLaRa.from_pretrained(
    "checkpoints/CLaRA-7B-Instruct/compression-128",
    trust_remote_code=True
).to(device)

documents = [
    [
        "Weldenia là một chi thực vật có hoa đơn loài thuộc họ Commelinaceae...",
        "Hagsatera là một chi thực vật có hoa thuộc họ lan (Orchidaceae)...",
        "Alsobia là một chi thực vật có hoa thuộc họ Gesneriaceae..."
    ]
]

questions = [
    "kể tên 1 hoa thuộc họ lan"
]

# Câu trả lời tạo ra:
# "Chi thực vật có nguồn gốc từ Mexico và Guatemala là Weldenia. 
# Phylica là một chi khác có nguồn gốc từ Nam Phi."

# Instruction-tuned usage
out = unirag.generate_from_text(
    questions=questions,
    documents=documents,
    max_new_tokens=64
)

print("Generated answer:", out)
