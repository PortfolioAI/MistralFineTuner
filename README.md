# Mistral Fine Tuner

Welcome to the Mistral Fine Tuner repository! This codebase is a simplified version forked from the original project by [bdytx5](https://github.com/bdytx5). We've distilled it down to a one-script solution for fine-tuning the Mistral model.

## Overview

While the original repository was designed with a broader suite of features, this version concentrates on providing a streamlined experience for tuning Mistral on a local machine.

## Usage

1. **Dataset Selection**: Utilize a GUI prompt to select a text file for training.
2. **Model Initialization**: Initialize the Mistral model from a chosen directory.
3. **Training**: Embark on the training journey with the capability to adjust parameters such as learning rate, batch size, dimension count (`r`), Lora Alpha, and set a specific loss threshold to halt training prematurely.
4. **Precision Control**: Toggle between FP16 and FP32 model precision based on your requirements.

## Steps to Run

- Fire up the script with the command: `python your_script_name.py`.
- Follow the intuitive main menu to load datasets, kickstart model initialization, train the model, fine-tune parameters, and switch precision settings.

## Important Notes

- Before commencing, ensure you've allocated enough storage for model checkpoints.
- Your dataset should be structured to have sequences constrained to a specified length. Each sequence ought to conclude with an assistant's response, prefaced by the maximum prior context.
