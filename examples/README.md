# GradES Example Notebooks

This directory contains example Jupyter notebooks demonstrating how to use GradES for fine-tuning language and vision-language models.

## Vision-Language Model (VLM) Examples

These examples are based on the dataset and preprocessing from the [Unsloth Qwen2.5 VL notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_5_VL_(7B)-Vision.ipynb).

### 1. `huggingface_vlm_lora_grades.ipynb`
- **Framework**: Hugging Face `transformers`
- **Fine-tuning Method**: LoRA
- **Description**: Demonstrates how to fine-tune a VLM with LoRA using the standard Hugging Face `Trainer` and integrate GradES for early stopping.

### 2. `unsloth_vlm_fft_grades.ipynb`
- **Framework**: Unsloth
- **Fine-tuning Method**: Full Fine-Tuning (FFT)
- **Description**: Shows how to perform Full Fine-Tuning on a VLM with Unsloth's `FastVisionModel` and `SFTTrainer`, accelerated with GradES.

## Language Model (LLM) Examples

### 3. `unsloth_lora_grades.ipynb`
- **Framework**: Unsloth
- **Fine-tuning Method**: LoRA
- **Description**: A standard example of fine-tuning a language model with LoRA and Unsloth, with GradES added to optimize the training process.

### 4. `unsloth_fft_grades.ipynb`
- **Framework**: Unsloth
- **Fine-tuning Method**: Full Fine-Tuning (FFT)
- **Description**: An example of performing Full Fine-Tuning on a language model with Unsloth, integrated with GradES for computational savings.
