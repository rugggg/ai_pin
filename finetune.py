import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Config
model_id = "Qwen/Qwen2.5-7B"
output_dir = "./qwen-hermes-lora"
dataset_size = 10000  # start small, can increase later

# Load model in 4bit to save VRAM during training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load and slice dataset
dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
dataset = dataset.shuffle(seed=10065).select(range(dataset_size))

def format_sample(sample):
    messages = []
    # include system prompt if present
    if sample.get("system_prompt"):
        messages.append({"role": "system", "content": sample["system_prompt"]})
    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return {"messages": messages}

dataset = dataset.map(format_sample)

# LoRA config
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training config
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch size = 16
    warmup_steps=50,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    # dataset_text_field="text",
    max_seq_length=1024,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=sft_config,
)

trainer.train()
trainer.save_model(output_dir)
print("Done! Model saved to", output_dir)
