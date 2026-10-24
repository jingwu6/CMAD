"""Plain transformers.Trainer + peft LoRA fine-tuning of the strategy
generator G, over (question -> target_strategies) pairs."""

import os
from typing import List

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from cmad.config import LoRAConfig
from cmad.prompts import STRATEGY_GENERATOR_PROMPT


class _StrategySFTDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[dict], tokenizer, max_length: int = 1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Same numbered-list format G produces at generation time (strategies.py).
        target_strategies = ex["target_strategies"]
        m = len(target_strategies)
        prompt = STRATEGY_GENERATOR_PROMPT.format(question=ex["question"], m=m)
        completion = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(target_strategies))
        full_text = prompt + completion + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, add_special_tokens=False
        )
        input_ids = full["input_ids"]
        labels = list(input_ids)
        # Only the completion contributes to the loss.
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def fit_lora_adapter(
    base_model_path: str,
    examples: List[dict],
    output_dir: str,
    config: LoRAConfig = LoRAConfig(),
) -> str:
    """Fine-tune a fresh LoRA adapter over `examples` and save it to
    `output_dir`. Returns output_dir on success.

    Loads its own copy of the base model; the caller is expected to make
    a GPU other than the vLLM server's visible (see run_cmad_full.py)."""
    print(f"[cmad] lora_sft starting: {len(examples)} examples -> {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )
    # low_cpu_mem_usage=False avoids meta-device initialization, which some
    # transformers/accelerate versions fail to move to GPU ("Cannot copy out
    # of meta tensor").
    peft_config = PeftLoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        target_modules=config.target_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    dataset = _StrategySFTDataset(examples, tokenizer)

    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            pad_len = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(b["attention_mask"] + [0] * pad_len)
            labels.append(b["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collate)
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"[cmad] lora_sft done, adapter saved to {output_dir}")
    return output_dir
