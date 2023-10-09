import tkinter as tk
from tkinter import filedialog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def select_directory(title="Select a Directory"):
    folder_selected = filedialog.askdirectory(title=title)
    if not folder_selected:
        raise ValueError("No directory selected.")
    return folder_selected

def select_file(title="Select a File"):
    file_selected = filedialog.askopenfilename(title=title)
    if not file_selected:
        raise ValueError("No file selected.")
    return file_selected

def load_data_and_model(text_file, model_directory):
    with open(text_file, 'r') as f:
        content = f.read()
    if not content:
        raise ValueError("Text file is empty.")

    model = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=precision).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=text_file, block_size=128)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
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

    return model, tokenizer, train_dataset

class CustomSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs):
        outputs = super().training_step(model, inputs)

        loss_value = outputs.loss if isinstance(outputs, dict) else outputs

        if loss_value and loss_value.item() <= stop_loss_value:
            print(f"Loss reached {stop_loss_value} or below. Saving and stopping training.")
            self.save_model()  # Save the model
            self.tokenizer.save_pretrained(self.args.output_dir)  # Save the tokenizer
            raise ValueError(f"Training halted due to loss reaching {stop_loss_value} or below")
        return outputs

def train_model(model, tokenizer, train_dataset, output_directory):
    training_args = TrainingArguments(
        output_dir=output_directory,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="steps",
        save_steps=save_steps,
        evaluation_strategy="no",
        eval_steps=50,
        do_eval=False,
        fp16=True if precision == torch.float16 else False,
        logging_dir="./logs",
    )

    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        dataset_text_field="text",
        max_seq_length=2048
    )

    try:
        trainer.train()
    except ValueError as e:
        print(e)
    trainer.save_model(output_directory)
    tokenizer.save_pretrained(output_directory)

def adjust_training_parameters():
    global warmup_steps, per_device_train_batch_size, gradient_accumulation_steps, max_steps, learning_rate, logging_steps, save_steps, stop_loss_value
    while True:
        print("\nTraining Parameters:")
        print(f"1. Warmup Steps (Current: {warmup_steps})")
        print(f"2. Per Device Train Batch Size (Current: {per_device_train_batch_size})")
        print(f"3. Gradient Accumulation Steps (Current: {gradient_accumulation_steps})")
        print(f"4. Max Steps (Current: {max_steps})")
        print(f"5. Learning Rate (Current: {learning_rate})")
        print(f"6. Logging Steps (Current: {logging_steps})")
        print(f"7. Save Steps (Current: {save_steps})")
        print(f"8. Set the loss value to stop training (Current: {stop_loss_value})")
        print(f"9. Back to main menu")
        choice = input("Select the parameter number you want to adjust: ")
        if choice == "1":
            warmup_steps = int(input("Enter new Warmup Steps: "))
        elif choice == "2":
            per_device_train_batch_size = int(input("Enter new Batch Size (per device): "))
        elif choice == "3":
            gradient_accumulation_steps = int(input("Enter new Gradient Accumulation Steps: "))
        elif choice == "4":
            max_steps = int(input("Enter new Max Steps: "))
        elif choice == "5":
            learning_rate = float(input("Enter new Learning Rate: "))
        elif choice == "6":
            logging_steps = int(input("Enter new Logging Steps: "))
        elif choice == "7":
            save_steps = int(input("Enter new Save Steps: "))
        elif choice == "8":
            stop_loss_value = float(input("Enter the loss value at which training should stop: "))
        elif choice == "9":
            break

def toggle_precision():
    global precision
    if precision == torch.float32:
        precision = torch.float16
        print("Model precision set to FP16.")
    else:
        precision = torch.float32
        print("Model precision set to FP32.")

def main_menu():
    global warmup_steps, per_device_train_batch_size, gradient_accumulation_steps, max_steps, learning_rate, logging_steps, save_steps, precision, stop_loss_value
    warmup_steps = 5
    per_device_train_batch_size = 64
    gradient_accumulation_steps = 1
    max_steps = 1000
    learning_rate = 2.5e-5
    logging_steps = 50
    save_steps = 50
    precision = torch.float16
    stop_loss_value = 1.66

    while True:
        print("\nMain Menu:")
        print("1. Load text file and initialize model")
        print("2. Train the model")
        print("3. Adjust Training Parameters")
        print("4. Toggle Model Precision (Current: FP16)" if precision == torch.float16 else "4. Toggle Model Precision (Current: FP32)")
        print("5. Exit")
        choice = input("Select an option: ")
        if choice == "1":
            text_file = select_file("Select a Text File")
            model_directory = select_directory("Select Model Directory")
            if text_file and model_directory:
                model, tokenizer, train_dataset = load_data_and_model(text_file, model_directory)
        elif choice == "2":
            if 'model' in locals() and 'tokenizer' in locals() and 'train_dataset' in locals():
                output_directory = select_directory("Select Output Directory")
                if output_directory:
                    train_model(model, tokenizer, train_dataset, output_directory)
            else:
                print("Load the text file and initialize the model first.")
        elif choice == "3":
            adjust_training_parameters()
        elif choice == "4":
            toggle_precision()
        elif choice == "5":
            break

if __name__ == "__main__":
    main_menu()
