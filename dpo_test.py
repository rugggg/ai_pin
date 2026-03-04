import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

model_id = "Qwen/Qwen2.5-7B"
adapter_path = "./qwen-hermes-lora"
output_dir = "./qwen-hermes-dpo"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_path)
model.enable_input_require_grads()

#ref_model = AutoModelForCausalLM.from_pretrained(
#    model_id,
#    quantization_config=bnb_config,
#    device_map="auto",
#)
#ref_model = PeftModel.from_pretrained(ref_model, adapter_path)

# Load dataset
dataset = load_dataset("argilla/dpo-mix-7k", split="train")

# Format into DPO structure
def format_sample(sample):
    # extract just the last user message as prompt
    # chosen/rejected already have full conversation including assistant response
    prompt = sample["chosen"][:-1]  # everything except last assistant turn
    chosen = sample["chosen"][-1]["content"]   # last assistant turn
    rejected = sample["rejected"][-1]["content"]
    
    return {
        "prompt": tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True),
        "chosen": chosen,
        "rejected": rejected,
    }

dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

# New LoRA config for DPO - fresh adapters on top of merged SFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
dpo_config = DPOConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,      # down from 2
    gradient_accumulation_steps=8,      # keep effective batch size at 8
    warmup_steps=50,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    max_length=512,                     # down from 1024
    max_prompt_length=256,              # down from 512
    beta=0.1,
    gradient_checkpointing=True,        # big VRAM saver
)

trainer = DPOTrainer(
    model=model,
    # ref_model=ref_model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model(output_dir)
print("Done! DPO model saved to", output_dir)
