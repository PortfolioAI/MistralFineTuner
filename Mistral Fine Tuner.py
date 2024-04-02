import logging
import os
import shutil
import tkinter as tk
from tkinter import filedialog

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TextDataset,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FineTuner:
    def __init__(self):
        self.warmup_steps = 1
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 32
        self.max_steps = 1000
        self.learning_rate = 2e-4
        self.logging_steps = 1
        self.save_steps = 1
        self.eval_steps = 500
        self.precision = torch.float16
        self.lora_alpha = 16
        self.r = 8
        self.trust_remote_code = False

    def load_tokenizer(self, model_directory):
        tokenizer = AutoTokenizer.from_pretrained(model_directory, trust_remote_code=self.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "right"
        return tokenizer

    def select_directory(self, title="Select a Directory"):
        folder_selected = filedialog.askdirectory(title=title)
        if not folder_selected:
            raise ValueError("No directory selected.")
        return folder_selected

    def select_file(self, title="Select a File"):
        file_selected = filedialog.askopenfilename(title=title)
        if not file_selected:
            raise ValueError("No file selected.")
        return file_selected

    def load_data_and_model(self, text_file, model_directory):
        try:
            logging.info("Loading model from directory: %s", model_directory)
            model = AutoModelForCausalLM.from_pretrained(
                model_directory, torch_dtype=self.precision, trust_remote_code=self.trust_remote_code
            ).to("cuda")
            logging.info("Model loaded successfully.")
            logging.info("Loading tokenizer.")
            tokenizer = self.load_tokenizer(model_directory)
            logging.info("Tokenizer loaded successfully.")
            logging.info("Attempting to load text file: %s", text_file)
            dataset = load_dataset("text", data_files=text_file)
            logging.info("Text file loaded into dataset.")
            logging.info("Tokenizing dataset.")

            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            logging.info("Dataset tokenized successfully.")
            logging.info("Enabling gradient checkpointing.")
            model.gradient_checkpointing_enable()
            logging.info("Preparing model for k-bit training.")
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ],
                bias="none",
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            logging.info("PEFT model configuration complete.")
            logging.info("Model and data loaded and prepared successfully.")
            return model, tokenizer, tokenized_dataset["train"]
        except Exception as e:
            logging.error("An error occurred: %s", e)
            raise

    def load_validation_dataset(self, validation_file):
        if validation_file:
            logging.info("Attempting to load validation file: %s", validation_file)
            try:
                validation_dataset = load_dataset("text", data_files=validation_file)
                logging.info("Validation file loaded into dataset.")
                logging.info("Tokenizing validation dataset.")

                def tokenize_function(examples):
                    return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

                tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True)
                logging.info("Validation dataset tokenized successfully.")
                return tokenized_validation_dataset["train"]
            except Exception as e:
                logging.error("Error loading or tokenizing validation dataset: %s", e)
                raise ValueError("Validation dataset must be in the same format as the training dataset.")
        else:
            logging.info("No validation file provided. Skipping validation.")
            return None

    def train_model(self, model, tokenizer, train_dataset, validation_dataset, output_directory, model_directory):
        training_args = TrainingArguments(
            output_dir=output_directory,
            warmup_steps=self.warmup_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            logging_steps=1,
            save_strategy="steps",
            save_steps=self.save_steps,
            evaluation_strategy="steps" if validation_dataset else "no",
            eval_steps=self.eval_steps if validation_dataset else None,
            do_eval=False,
            fp16=True if self.precision == torch.float16 else False,
            logging_dir="./logs",
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            dataset_text_field="text",
            max_seq_length=4096,
        )

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            preds = preds.flatten()
            labels = labels.flatten()
            return {"accuracy": (preds == labels).mean()}

        if validation_dataset:
            trainer.compute_metrics = compute_metrics

        trainer.train()

        if validation_dataset:
            eval_results = trainer.evaluate()
            logging.info("Validation metrics:")
            for metric, value in eval_results.items():
                logging.info(f"{metric}: {value:.4f}")

        trainer.save_model(output_directory)
        tokenizer.save_pretrained(output_directory)
        shutil.copyfile(os.path.join(model_directory, "config.json"), os.path.join(output_directory, "config.json"))

    def adjust_training_parameters(self):
        while True:
            print("\nTraining Parameters:")
            print(f"1. Warmup Steps (Current: {self.warmup_steps})")
            print(f"2. Per Device Train Batch Size (Current: {self.per_device_train_batch_size})")
            print(f"3. Gradient Accumulation Steps (Current: {self.gradient_accumulation_steps})")
            print(f"4. Max Steps (Current: {self.max_steps})")
            print(f"5. Learning Rate (Current: {self.learning_rate})")
            print(f"6. Logging Steps (Current: {self.logging_steps})")
            print(f"7. Save Steps (Current: {self.save_steps})")
            print(f"8. Eval Steps (Current: {self.eval_steps})")
            print(f"9. Change Lora Alpha (Current: {self.lora_alpha})")
            print(f"10. Adjust Dimension Count (Current: {self.r})")
            print(f"11. Back to main menu")
            choice = input("Select the parameter number you want to adjust: ")
            if choice == "1":
                self.warmup_steps = int(input("Enter new Warmup Steps: "))
            elif choice == "2":
                self.per_device_train_batch_size = int(input("Enter new Batch Size (per device): "))
            elif choice == "3":
                self.gradient_accumulation_steps = int(input("Enter new Gradient Accumulation Steps: "))
            elif choice == "4":
                self.max_steps = int(input("Enter new Max Steps: "))
            elif choice == "5":
                self.learning_rate = float(input("Enter new Learning Rate: "))
            elif choice == "6":
                self.logging_steps = int(input("Enter new Logging Steps: "))
            elif choice == "7":
                self.save_steps = int(input("Enter new Save Steps: "))
            elif choice == "8":
                self.eval_steps = int(input("Enter new Eval Steps: "))
            elif choice == "9":
                self.lora_alpha = int(input("Enter new Lora Alpha: "))
            elif choice == "10":
                self.r = int(input("Enter new Dimension Count: "))
            elif choice == "11":
                break

    def toggle_precision(self):
        if self.precision == torch.float32:
            self.precision = torch.float16
            print("Model precision set to FP16.")
        else:
            self.precision = torch.float32
            print("Model precision set to FP32.")

    def main_menu(self):
        while True:
            print("\nMain Menu:")
            print("1. Load text file and initialize model")
            print("2. Train the model")
            print("3. Adjust Training Parameters")
            print(
                "4. Toggle Model Precision (Current: FP16)"
                if self.precision == torch.float16
                else "4. Toggle Model Precision (Current: FP32)"
            )
            print(
                "5. Toggle Trust Remote Code (Current: Disabled)"
                if not self.trust_remote_code
                else "5. Toggle Trust Remote Code (Current: Enabled)"
            )
            print("6. Exit")
            choice = input("Select an option: ")
            if choice == "1":
                text_file = self.select_file("Select a Text File")
                validation_file = self.select_file("Select a Validation File (optional, must be in the same format as the training dataset)")
                model_directory = self.select_directory("Select Model Directory")
                if text_file and model_directory:
                    model, tokenizer, train_dataset = self.load_data_and_model(text_file, model_directory)
                    self.tokenizer = tokenizer  # Store the tokenizer as an instance variable
                    validation_dataset = self.load_validation_dataset(validation_file) if validation_file else None
            elif choice == "2":
                if "model" in locals() and "tokenizer" in locals() and "train_dataset" in locals():
                    output_directory = self.select_directory("Select Output Directory")
                    if output_directory:
                        self.train_model(model, tokenizer, train_dataset, validation_dataset, output_directory, model_directory)
                else:
                    print("Load the text file and initialize the model first.")
            elif choice == "3":
                self.adjust_training_parameters()
            elif choice == "4":
                self.toggle_precision()
            elif choice == "5":
                self.trust_remote_code = not self.trust_remote_code
                print("Trust Remote Code enabled." if self.trust_remote_code else "Trust Remote Code disabled.")
            elif choice == "6":
                break


if __name__ == "__main__":
    fine_tuner = FineTuner()
    fine_tuner.main_menu()
