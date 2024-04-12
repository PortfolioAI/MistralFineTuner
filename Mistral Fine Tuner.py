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
from colorama import init, Fore, Style, Back
from tabulate import tabulate

init()
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
            model = self.load_model(model_directory)
            tokenizer = self.load_tokenizer(model_directory)
            dataset = self.load_dataset(text_file, tokenizer)
            model = self.prepare_model(model)
            return model, tokenizer, dataset
        except Exception as e:
            logging.error("An error occurred while loading data and model: %s", e)
            raise

    def load_model(self, model_directory):
        logging.info("Loading model from directory: %s", model_directory)
        model = AutoModelForCausalLM.from_pretrained(
            model_directory, torch_dtype=self.precision, trust_remote_code=self.trust_remote_code
        ).to("cuda")
        logging.info("Model loaded successfully.")
        return model

    def load_dataset(self, text_file, tokenizer):
        logging.info("Attempting to load text file: %s", text_file)
        dataset = load_dataset("text", data_files=text_file)
        logging.info("Text file loaded into dataset.")
        logging.info("Tokenizing dataset.")
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        logging.info("Dataset tokenized successfully.")
        return tokenized_dataset["train"]

    def prepare_model(self, model):
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
        return model

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
            training_args = self.get_training_arguments(output_directory)
            trainer = self.create_trainer(model, tokenizer, train_dataset, training_args)
            trainer.train()
            self.save_model(model, tokenizer, output_directory, model_directory)
        except Exception as e:
            logging.error("An error occurred during model training: %s", e)
            raise

    def get_training_arguments(self, output_directory):
        return TrainingArguments(
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

    def create_trainer(self, model, tokenizer, train_dataset, training_args):
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
        return trainer

    def save_model(self, model, tokenizer, output_directory, model_directory):
        model.save_pretrained(output_directory)
        tokenizer.save_pretrained(output_directory)
        shutil.copyfile(os.path.join(model_directory, "config.json"), os.path.join(output_directory, "config.json"))
        self.save_hyperparameters(output_directory)
        self.merge_and_save_model(model, output_directory, model_directory)

    def save_hyperparameters(self, output_directory):
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

    def merge_and_save_model(self, model, output_directory, model_directory):
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

    def evaluate_model(self, model_directory, evaluation_dataset):
        try:
            model = self.load_model(model_directory)
            tokenizer = self.load_tokenizer(model_directory)
            print(Fore.YELLOW + Style.BRIGHT + "Tokenizing evaluation dataset..." + Style.RESET_ALL)
            tokenized_evaluation_dataset = evaluation_dataset.map(lambda x: self.tokenize_function(x, tokenizer), batched=True)
            tokenized_evaluation_dataset = tokenized_evaluation_dataset.with_format("torch")
            print(Fore.GREEN + Style.BRIGHT + "Evaluation dataset tokenized successfully." + Style.RESET_ALL)
            print(Fore.YELLOW + Style.BRIGHT + "Evaluating the model..." + Style.RESET_ALL)
            training_args = TrainingArguments(
                output_dir="./eval_results",
                do_train=False,
                do_eval=True,
                per_device_eval_batch_size=1,
                eval_accumulation_steps=1,
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                eval_dataset=tokenized_evaluation_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
                dataset_text_field="text",
                max_seq_length=4096,
            )
            eval_results = trainer.evaluate()
            print(Fore.GREEN + Style.BRIGHT + "Model evaluation completed." + Style.RESET_ALL)
            del model
            del tokenizer
            torch.cuda.empty_cache()
            return eval_results
        except Exception as e:
            logging.error(Fore.RED + Style.BRIGHT + "Error evaluating the model: %s" + Style.RESET_ALL, e)
            raise ValueError("Failed to evaluate the model.")

    def tokenize_function(self, examples, tokenizer):
        try:
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        except Exception as e:
            logging.error(Fore.RED + Style.BRIGHT + "Error tokenizing examples: %s" + Style.RESET_ALL, e)
            raise ValueError("Failed to tokenize examples.")

    def main_menu(self):
        evaluation_dataset = None
        model_directories = []
        while True:
            print(Fore.CYAN + Back.BLACK + Style.BRIGHT + "\nMain Menu:" + Style.RESET_ALL)
            print(Fore.CYAN + "1. Load text file and initialize model" + Style.RESET_ALL)
            print(Fore.CYAN + "2. Train the model" + Style.RESET_ALL)
            print(Fore.CYAN + "3. Adjust Training Parameters" + Style.RESET_ALL)
            print(Fore.CYAN + "4. Toggle Model Precision (Current: FP16)" + Style.RESET_ALL
                if self.precision == torch.float16
                else "4. Toggle Model Precision (Current: FP32)" + Style.RESET_ALL)
            print(Fore.CYAN + "5. Toggle Trust Remote Code (Current: Disabled)" + Style.RESET_ALL
                if not self.trust_remote_code
                else "5. Toggle Trust Remote Code (Current: Enabled)" + Style.RESET_ALL)
            print(Fore.CYAN + "6. Select Evaluation File" + Style.RESET_ALL)
            print(Fore.CYAN + "7. Add Model Directory for Evaluation" + Style.RESET_ALL)
            if len(model_directories) > 1:
                print(Fore.CYAN + "8. Run Batch Evaluation" + Style.RESET_ALL)
            else:
                print(Fore.CYAN + "8. Run Evaluation" + Style.RESET_ALL)
            print(Fore.CYAN + "9. Exit" + Style.RESET_ALL)
            choice = input(Fore.YELLOW + Style.BRIGHT + "Select an option: " + Style.RESET_ALL)
            try:
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
                        print(Fore.RED + Style.BRIGHT + "Load the text file and initialize the model first." + Style.RESET_ALL)
                elif choice == "3":
                    self.adjust_training_parameters()
                elif choice == "4":
                    self.toggle_precision()
                elif choice == "5":
                    self.trust_remote_code = not self.trust_remote_code
                    print(Fore.GREEN + Style.BRIGHT + "Trust Remote Code enabled." + Style.RESET_ALL if self.trust_remote_code else Fore.RED + Style.BRIGHT + "Trust Remote Code disabled." + Style.RESET_ALL)
                elif choice == "6":
                    evaluation_file = self.select_file("Select Evaluation File")
                    evaluation_dataset = load_dataset("text", data_files=evaluation_file)["train"]
                    print(Fore.GREEN + Style.BRIGHT + "Evaluation file loaded successfully." + Style.RESET_ALL)
                elif choice == "7":
                    model_directory = self.select_directory("Select Model Directory")
                    model_directories.append(model_directory)
                    print(Fore.GREEN + Style.BRIGHT + f"Model directory '{model_directory}' added for evaluation." + Style.RESET_ALL)
                elif choice == "8":
                    if evaluation_dataset is None:
                        print(Fore.RED + Style.BRIGHT + "Please select an evaluation file first." + Style.RESET_ALL)
                    elif not model_directories:
                        print(Fore.RED + Style.BRIGHT + "Please add at least one model directory for evaluation." + Style.RESET_ALL)
                    else:
                        results = []
                        for model_directory in model_directories:
                            print(Fore.YELLOW + Style.BRIGHT + f"Evaluating model from directory: {model_directory}" + Style.RESET_ALL)
                            eval_results = self.evaluate_model(model_directory, evaluation_dataset)
                            results.append([model_directory] + [f"{value:.4f}" for value in eval_results.values()])
                        headers = ["Model"] + list(eval_results.keys())
                        min_eval_loss = min(float(result[1]) for result in results)
                        table_data = []
                        for result in results:
                            if float(result[1]) == min_eval_loss:
                                result[1] = Fore.GREEN + result[1] + Style.RESET_ALL
                            table_data.append(result)
                        table = tabulate(table_data, headers, tablefmt="grid")
                        print(Fore.BLUE + Back.WHITE + Style.BRIGHT + "\nEvaluation Results:" + Style.RESET_ALL)
                        print(table)
                elif choice == "9":
                    print(Fore.GREEN + Style.BRIGHT + "Exiting the program. Goodbye!" + Style.RESET_ALL)
                    break
                else:
                    print(Fore.RED + Style.BRIGHT + "Invalid choice. Please try again." + Style.RESET_ALL)
            except ValueError as e:
                print(Fore.RED + Style.BRIGHT + str(e) + Style.RESET_ALL)
            except Exception as e:
                logging.error(Fore.RED + Style.BRIGHT + "An unexpected error occurred: %s" + Style.RESET_ALL, e)
                print(Fore.RED + Style.BRIGHT + "An unexpected error occurred. Please check the logs for more details." + Style.RESET_ALL)

if __name__ == "__main__":
    print(Fore.MAGENTA + Back.WHITE + Style.BRIGHT + "Welcome to the Model Fine-Tuning and Evaluation Program!" + Style.RESET_ALL)
    try:
        fine_tuner = FineTuner()
        fine_tuner.main_menu()
    except Exception as e:
        logging.error(Fore.RED + Style.BRIGHT + "An unexpected error occurred: %s" + Style.RESET_ALL, e)
        print(Fore.RED + Style.BRIGHT + "An unexpected error occurred. Please check the logs for more details." + Style.RESET_ALL)
