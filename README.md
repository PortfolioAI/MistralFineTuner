# Mistral Fine Tuner README

## Welcome
Forked from bdytx5's project, this streamlined codebase offers a one-script solution for fine-tuning language models like Mistral and beyond.

## Overview
Designed for simplicity, this version focuses on local machine fine-tuning, offering an efficient, user-friendly experience. Key updates include enhanced data processing, robust logging, and error handling improvements.

## Program Updates
- **Data Processing**: Replaced TextDataset with `load_dataset` for efficient data handling.
- **Logging**: Comprehensive logging integrated for better debugging and traceability.
- **Error Handling**: Improved error handling during model and data preparation.
- **Training Functionality**: `train_model` now handles model directory and config file management.

## Usage
- **Dataset Selection**: GUI prompt for text file selection.
- **Model Initialization**: Start with the model from a chosen directory.
- **Training**: Adjust parameters like learning rate, batch size, etc., and control training with a loss threshold.
- **Precision Control**: Toggle between FP16 and FP32.

## Steps to Run
1. Execute `python your_script_name.py`.
2. Follow the main menu: load datasets, initialize, train, adjust parameters, toggle precision.

## Important Notes
- **Storage**: Ensure sufficient storage for checkpoints.
- **Dataset Format**: Format with sequences of a specific length, ending with an assistant's response.

## Getting Started
1. Set up Python environment.
2. Clone/download `your_script_name.py`.
3. Format your dataset.
4. Run the script, follow instructions.

## Support and Contribution
Open an issue for support. Contributions via pull requests are welcome.

Happy fine-tuning!