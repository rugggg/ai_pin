from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Config
model_id = "Qwen/Qwen2.5-7B"
output_dir = "./qwen-hermes-lora"
dataset_size = 50000

# Load model with Unsloth - handles quantization internally, no need for BitsAndBytesConfig
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# Set Qwen chat template manually
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = "<|im_end|>"
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=10065,
    use_rslora=False,
)

# Load and format dataset
dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
dataset = dataset.shuffle(seed=42).select(range(dataset_size))

def format_sample(sample):
    messages = []
    if sample.get("system_prompt"):
        messages.append({"role": "system", "content": sample["system_prompt"]})
    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    return {"messages": messages}

dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

# Training config
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch size = 16
    warmup_steps=50,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_grad_norm=1.0,
    packing=False,
    max_seq_length=1024,
)
def formatting_func(samples):
    # handle both single sample (unsloth test) and batched
    if isinstance(samples["messages"][0], dict):
        # single sample
        return [tokenizer.apply_chat_template(
            samples["messages"],
            tokenize=False,
            add_generation_prompt=False
        )]
    else:
        # batched
        texts = []
        for messages in samples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return texts

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=sft_config,
)

trainer.train()
trainer.save_model(output_dir)
print("Done! Model saved to", output_dir)
