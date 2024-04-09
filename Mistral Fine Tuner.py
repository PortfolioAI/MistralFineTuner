import logging
import os
import shutil
import tkinter as tk
from tkinter import filedialog
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TextDataset,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FineTuner:
    def __init__(self):
        self.warmup_steps = 1
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 1
        self.max_steps = 1000
        self.learning_rate = 2e-4
        self.logging_steps = 1
        self.save_steps = 1
        self.precision = torch.float16
        self.lora_alpha = 16
        self.r = 8
        self.trust_remote_code = False
        self.target_loss = None

    def load_tokenizer(self, model_directory):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_directory, trust_remote_code=self.trust_remote_code)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.padding_side = "right"
            return tokenizer
        except Exception as e:
            logging.error("Error loading tokenizer: %s", e)
            raise

    def select_directory(self, title="Select a Directory"):
        try:
            folder_selected = filedialog.askdirectory(title=title)
            if not folder_selected:
                raise ValueError("No directory selected.")
            return folder_selected
        except Exception as e:
            logging.error("Error selecting directory: %s", e)
            raise

    def select_file(self, title="Select a File"):
        try:
            file_selected = filedialog.askopenfilename(title=title)
            if not file_selected:
                raise ValueError("No file selected.")
            return file_selected
        except Exception as e:
            logging.error("Error selecting file: %s", e)
            raise

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
            logging.error("An error occurred while loading data and model: %s", e)
            raise

    class StopTrainingCallback(TrainerCallback):
        def __init__(self, target_loss, window_size=3):
            self.target_loss = target_loss
            self.window_size = window_size
            self.loss_history = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.target_loss is not None and logs and "loss" in logs:
                self.loss_history.append(logs["loss"])
                if len(self.loss_history) >= self.window_size:
                    avg_loss = sum(self.loss_history[-self.window_size:]) / self.window_size
                    if avg_loss <= self.target_loss:
                        control.should_training_stop = True

    def train_model(self, model, tokenizer, train_dataset, output_directory, model_directory):
        try:
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
                evaluation_strategy="no",
                eval_steps=None,
                fp16=True if self.precision == torch.float16 else False,
                logging_dir="./logs",
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                dataset_text_field="text",
                max_seq_length=4096,
            )
            stop_training_callback = self.StopTrainingCallback(self.target_loss)
            trainer.add_callback(stop_training_callback)
            trainer.train()
            trainer.save_model(output_directory)
            tokenizer.save_pretrained(output_directory)
            shutil.copyfile(os.path.join(model_directory, "config.json"), os.path.join(output_directory, "config.json"))
            hyperparameters = {
                "warmup_steps": self.warmup_steps,
                "per_device_train_batch_size": self.per_device_train_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_steps": self.max_steps,
                "learning_rate": self.learning_rate,
                "logging_steps": self.logging_steps,
                "save_steps": self.save_steps,
                "precision": str(self.precision),
                "lora_alpha": self.lora_alpha,
                "r": self.r,
                "trust_remote_code": self.trust_remote_code,
                "target_loss": self.target_loss,
            }
            with open(os.path.join(output_directory, "hyperparameters.txt"), "w") as file:
                for key, value in hyperparameters.items():
                    file.write(f"{key}: {value}\n")
            logging.info("Merging PEFT model with the base model...")
            device_arg = {'device_map': 'auto'}
            base_model = AutoModelForCausalLM.from_pretrained(
                model_directory,
                return_dict=True,
                torch_dtype=torch.float16,
                trust_remote_code=self.trust_remote_code,
                **device_arg
            )
            model = PeftModel.from_pretrained(base_model, output_directory, **device_arg)
            model = model.merge_and_unload()
            model.to("cuda")
            model.save_pretrained(output_directory)
            logging.info("Merged model saved to %s", output_directory)

            del model
            del base_model
            torch.cuda.empty_cache()
            logging.info("Model unloaded from GPU memory")

        except Exception as e:
            logging.error("An error occurred during model training: %s", e)
            raise

    def adjust_training_parameters(self):
        while True:
            try:
                print("\nTraining Parameters:")
                print(f"1. Warmup Steps (Current: {self.warmup_steps})")
                print(f"2. Per Device Train Batch Size (Current: {self.per_device_train_batch_size})")
                print(f"3. Gradient Accumulation Steps (Current: {self.gradient_accumulation_steps})")
                print(f"4. Max Steps (Current: {self.max_steps})")
                print(f"5. Learning Rate (Current: {self.learning_rate})")
                print(f"6. Logging Steps (Current: {self.logging_steps})")
                print(f"7. Save Steps (Current: {self.save_steps})")
                print(f"8. Change Lora Alpha (Current: {self.lora_alpha})")
                print(f"9. Adjust Dimension Count (Current: {self.r})")
                print(f"10. Set Target Loss (Current: {self.target_loss})")
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
                    self.lora_alpha = int(input("Enter new Lora Alpha: "))
                elif choice == "9":
                    self.r = int(input("Enter new Dimension Count: "))
                elif choice == "10":
                    target_loss = input("Enter the target loss value (or leave empty to disable): ")
                    self.target_loss = float(target_loss) if target_loss else None
                elif choice == "11":
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError as e:
                logging.error("Invalid input: %s", e)

    def toggle_precision(self):
        if self.precision == torch.float32:
            self.precision = torch.float16
            print("Model precision set to FP16.")
        else:
            self.precision = torch.float32
            print("Model precision set to FP32.")

    def main_menu(self):
        while True:
            try:
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
                    model_directory = self.select_directory("Select Model Directory")
                    if text_file and model_directory:
                        model, tokenizer, train_dataset = self.load_data_and_model(text_file, model_directory)
                elif choice == "2":
                    if "model" in locals() and "tokenizer" in locals() and "train_dataset" in locals():
                        output_directory = self.select_directory("Select Output Directory")
                        if output_directory:
                            self.train_model(model, tokenizer, train_dataset, output_directory, model_directory)
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
                else:
                    print("Invalid choice. Please try again.")
            except Exception as e:
                logging.error("An error occurred: %s", e)

if __name__ == "__main__":
    fine_tuner = FineTuner()
    fine_tuner.main_menu()
