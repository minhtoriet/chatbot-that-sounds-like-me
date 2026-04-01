import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# ====================== CONFIG ======================
model_path = Path(__file__).parent / "my-chatbot"

print(f"Loading model from: {model_path.resolve()}")

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="xpu",        # Use your Arc B580
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded successfully!\n")

# ====================== CHAT LOOP ======================
print("Chat with your model (type 'exit' or 'quit' to stop)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("fuck off!")
        break

    # Format the conversation
    messages = [
        {"role": "system", "content": "you are a bro. be sarcastic, reply shortly and concisely, you are not an assistant."},
        {"role": "user", "content": user_input}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the output (remove the prompt part)
    if response.startswith(text):
        response = response[len(text):].strip()
    
    print(f"Assistant: {response}\n")