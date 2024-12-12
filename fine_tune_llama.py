import os

# Set environment variables before importing any modules
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Use GPU 3
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Step 1: Define paths and configurations
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
ACCESS_TOKEN = "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"  
DATASET_PATH = "fine_tuning_dataset.json"
OUTPUT_DIR = "./results/llama2-finetuned"

# Step 2: Load the dataset and split into train, validation, and test
def load_data(dataset_path):
    print("Loading dataset and splitting into train, validation, and test sets...")
    dataset = load_dataset("json", data_files=dataset_path)
    # Split the dataset into train and temp (which will be split into validation and test)
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    # Further split the temp set into validation and test sets
    test_valid = dataset["test"].train_test_split(test_size=0.5, seed=42)
    dataset["validation"] = test_valid["train"]
    dataset["test"] = test_valid["test"]
    return dataset

# Step 3: Tokenize the data
def tokenize_data(dataset, tokenizer):
    print("Tokenizing data...")
    def preprocess_function(examples):
        instructions = [
            "### Instruction: " + str(inst) + "\n### Input: " + str(inp) + "\n### Response: " + str(out)
            for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
        ]
        max_length = 512
        tokenized = tokenizer(
            instructions,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # Add labels field matching input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_datasets = {}
    for split in dataset.keys():
        print(f"Tokenizing {split} dataset...")
        tokenized_dataset = dataset[split].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset[split].column_names,
        )
        tokenized_datasets[split] = tokenized_dataset
    return tokenized_datasets

# Step 4: Configure QLoRA and Fine-tuning
def fine_tune():
    print("Loading tokenizer...")
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME, use_auth_token=ACCESS_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    print("Configuring 4-bit quantization...")
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base model with quantization...")
    # Load model with quantization
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},  # Explicitly set to the first (and only) visible GPU
        use_auth_token=ACCESS_TOKEN,
        use_cache=False,
    )

    print("Model device map:", model.hf_device_map)  # Optional: Verify device map

    print("Preparing model for k-bit training...")
    # Prepare for 4-bit training
    model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA...")
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Wrapping model with PEFT...")
    # Wrap model with PEFT
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize dataset
    dataset = load_data(DATASET_PATH)
    tokenized_datasets = tokenize_data(dataset, tokenizer)

    print("Defining training arguments...")
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,  # Increased to reduce memory usage
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="steps",  # Evaluate every `eval_steps`
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,  # Enable gradient checkpointing
    )

    print("Setting up data collator...")
    # Use the correct data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    print("Initializing Trainer...")
    # Initialize Trainer with validation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    # Start training
    trainer.train()

    print("Saving model and tokenizer...")
    # Save model and tokenizer
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Evaluating model on test set...")
    # Evaluate on the test set
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Test Results:", test_results)

if __name__ == "__main__":
    fine_tune()
