#!/usr/bin/env python3
"""
Fine-tune Gemma 3 27B using LoRA (Low-Rank Adaptation) for memory-efficient training.

This script provides a minimal, honest implementation of LoRA fine-tuning for Gemma 3.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str) -> List[Dict]:
    """Load training data from JSON file."""
    logger.info(f"Loading training data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = []
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} training examples")
    return data


def format_training_example(example: Dict) -> str:
    """Format a training example into Gemma's expected format."""
    if "messages" in example:
        # Chat format
        formatted_parts = []
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                formatted_parts.append(f"<|assistant|>\n{content}")
        return "\n".join(formatted_parts)
    elif "prompt" in example and "response" in example:
        # Prompt-response format
        return f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}"
    else:
        logger.warning(f"Unknown training example format: {example.keys()}")
        return ""


def prepare_dataset(data: List[Dict], tokenizer) -> Dataset:
    """Prepare the dataset for training."""
    logger.info("Preparing dataset...")
    
    # Format and tokenize examples
    formatted_texts = []
    for example in data:
        text = format_training_example(example)
        if text:
            formatted_texts.append(text)
    
    # Create dataset
    dataset_dict = {"text": formatted_texts}
    dataset = Dataset.from_dict(dataset_dict)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
    return tokenized_dataset


def main():
    global args
    
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 27B with LoRA")
    parser.add_argument("--model_name_or_path", default="google/gemma-3-27b-it", help="Model name or path")
    parser.add_argument("--data_path", required=True, help="Path to training data JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    parser.add_argument("--tf32", action="store_true", help="Use TF32 for matmul")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    
    args = parser.parse_args()
    
    # Validate dependencies
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import bitsandbytes as bnb
    except ImportError as e:
        logger.error("Missing required dependencies for LoRA fine-tuning")
        logger.error("Install with: pip install peft bitsandbytes accelerate")
        sys.exit(1)
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("Fine-tuning Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    logger.info(f"Batch size: {args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"LoRA alpha: {args.lora_alpha}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data
    training_data = load_training_data(args.data_path)
    if not training_data:
        logger.error("No training data loaded")
        sys.exit(1)
    
    # Prepare dataset
    train_dataset = prepare_dataset(training_data, tokenizer)
    
    # Load model with 4-bit quantization
    logger.info("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        tf32=args.tf32,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        optim="paged_adamw_8bit",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Fine-tuning complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("")
    logger.info("To use the fine-tuned model:")
    logger.info("1. Load base model: model = AutoModelForCausalLM.from_pretrained('google/gemma-3-27b-it')")
    logger.info(f"2. Load LoRA adapter: model = PeftModel.from_pretrained(model, '{args.output_dir}')")


if __name__ == "__main__":
    main()
