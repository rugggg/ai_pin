# Qwen2.5-7B Fine-Tuning on OpenHermes 2.5

Learning project — SFT fine-tuning a Qwen2.5-7B-Instruct model on the OpenHermes 2.5 dataset using QLoRA.

## Environment

- **GPU:** RunPod A40 (48GB VRAM)
- **Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Dataset:** `teknium/OpenHermes-2.5` (10k sample subset)

## Stack

| Library | Role |
|---|---|
| `transformers` | Model loading and tokenization |
| `datasets` | Data loading and preprocessing |
| `peft` | LoRA adapter injection |
| `bitsandbytes` | 4-bit QLoRA quantization |
| `trl` | SFTTrainer training loop |
| `accelerate` | Device management |
| `lm-eval` | MMLU benchmarking |

## Setup

```bash
pip install requirements.txt
```

## Baseline Eval

```bash
lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=bfloat16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 2 \
    --output_path ./baseline_results
```

**Baseline MMLU:** 74.2%

## Training

```bash
python finetune.py
```

Key config:
- LoRA rank 16, alpha 32, target modules: q/k/v/o proj
- 4-bit NF4 quantization, bfloat16 compute
- Effective batch size 16 (batch 2 × grad accum 8)
- Max sequence length 1024
- 1 epoch over 10k samples

## Post-Training Eval

```bash
lm_eval --model hf \
    --model_args pretrained=./qwen-hermes-lora,dtype=bfloat16 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 2 \
    --output_path ./finetuned_results
```

## Next Steps

- [ ] Start from base model (`Qwen/Qwen2.5-7B`) instead of instruct
- [ ] Fix prompt/response loss masking
- [ ] Scale to 50k samples
- [ ] Try DPO with `HuggingFaceH4/ultrafeedback_binarized`
- [ ] Switch to `Qwen2.5-Coder` base for coding specialization
- [ ] Add GRPO stage with execution-based rewards
