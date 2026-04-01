import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ====================== CONFIG ======================
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
max_length = 512

script_dir = Path(__file__).parent
json_file = script_dir / "cleaned" / "message-1.json"

print(f"XPU available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"Device name: {torch.xpu.get_device_name(0)}")
# ===================================================

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="xpu",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ====================== CUSTOM DATASET ======================
class ChatDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        print(f"Loading JSON from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        if not isinstance(self.raw_data, list):
            self.raw_data = [self.raw_data]

        print(f"Loaded {len(self.raw_data)} examples")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        example = self.raw_data[idx]

        if isinstance(example, dict) and "role" in example and "content" in example:
            messages = [
                {"role": "user", "content": "Respond in a natural and friendly way."},
                example
            ]
        else:
            messages = example.get("messages", [example])

        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized


# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Create dataset
chat_dataset = ChatDataset(json_file, tokenizer, max_length)

# ====================== TRAINING ======================
training_args = TrainingArguments(
    output_dir="my-chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    bf16=True,
    fp16=False,
    logging_steps=20,
    save_strategy="epoch",
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=chat_dataset,
    data_collator=data_collator,     # ← This replaces the old tokenizer argument
)

print(f"Starting training on {len(chat_dataset)} examples...")
trainer.train()

# Save model
model.save_pretrained("my-chatbot")
tokenizer.save_pretrained("my-chatbot")
print("Training finished! Model saved to 'my-chatbot' folder")