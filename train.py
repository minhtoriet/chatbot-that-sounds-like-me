import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ====================== CONFIG ======================
device = "xpu"
model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
max_length = 512

script_dir = Path(__file__).parent
json_file = script_dir / "cleaned" / "message-1.json"

print(f"Looking for JSON at: {json_file}")
# ===================================================

# Load model + tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()

# ====================== FIXED DATASET (handles your exact JSON structure) ======================
class ChatDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        print(f"Loading JSON from: {json_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        # Debug so you can see exactly what your file contains
        print(f"Loaded {len(self.data)} examples")
        if self.data:
            first = self.data[0]
            print(f"Type of first example: {type(first)}")
            if isinstance(first, dict):
                print(f"Keys in first example: {list(first.keys())}")
                preview = str(first)[:400] + "..." if len(str(first)) > 400 else str(first)
                print(f"Preview: {preview}")
            else:
                print(f"First example preview: {first}")
        print("-" * 60)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # ── SMART DETECTION FOR YOUR JSON STRUCTURE ──
        if isinstance(example, dict) and "role" in example and "content" in example:
            # Your JSON is a list of single messages like {"role": "...", "content": "..."}
            # We wrap it into a proper chat list (required by apply_chat_template)
            messages = [example]
        
        elif isinstance(example, list):
            # Full conversation already (list of messages)
            messages = example
        
        elif isinstance(example, dict) and "messages" in example:
            # Classic format with "messages" key
            messages = example["messages"]
        
        else:
            # Fallback - try to use the whole thing
            messages = example if isinstance(example, list) else [example]

        # Apply Qwen chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in tokenized.items()}


# Create dataset and dataloader
dataset = ChatDataset(json_file, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ====================== TRAINING ======================
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

print("Starting training on your cleaned messages...")

for epoch in range(1):
    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Epoch 1 | Batch {batch_idx} | Loss: {loss.item():.4f}")

# Save
model.save_pretrained("my-chatbot")
tokenizer.save_pretrained("my-chatbot")
print("Training finished! Model saved to folder: my-chatbot")