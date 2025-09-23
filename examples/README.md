# 📚 GradES Examples

This directory contains example notebooks and scripts demonstrating how to use GradES with different frameworks and training setups.

## 🦙 Unsloth Integration

### LoRA Fine-tuning
- **File**: `unsloth_lora_grades.ipynb`
- **Description**: Demonstrates GradES with Unsloth for LoRA fine-tuning
- **Features**: 40-50% speedup with LoRA training
- **Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IXZZZ9/GradES/blob/main/examples/unsloth_lora_grades.ipynb)

### Full Fine-tuning (FFT)
- **File**: `unsloth_fft_grades.ipynb`
- **Description**: Demonstrates GradES with Unsloth for full parameter fine-tuning
- **Features**: Maximum performance with efficiency gains
- **Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IXZZZ9/GradES/blob/main/examples/unsloth_fft_grades.ipynb)

## 🚀 Quick Start

### Option 1: Use with Unsloth Notebook (Recommended)
1. Open the [Unsloth Llama3.1 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)
2. Add this cell after imports:
   ```python
   !pip install grades
   ```
3. Replace the trainer setup with:
   ```python
   from grades import GradEarlyStoppingCallback, GradEarlyStoppingConfig

   # For LoRA
   config = GradEarlyStoppingConfig(tau=1e-10, alpha=0.1, enable_wandb_logging=True)

   # For FFT
   config = GradEarlyStoppingConfig(tau=1e-10, alpha=0.90, enable_wandb_logging=True)

   callback = GradEarlyStoppingCallback(config)

   # Add callbacks=[callback] to your SFTTrainer
   ```

### Option 2: Use Our Example Notebooks
1. Click the Colab badges above to open our example notebooks
2. Run all cells to see GradES in action
3. Modify hyperparameters as needed

## ⚙️ Configuration Guidelines

### LoRA Training
```python
config = GradEarlyStoppingConfig(
    tau=1e-10,      # Very low threshold for LoRA
    alpha=0.1,      # Allow early freezing (10% progress)
    enable_wandb_logging=True
)
```

### Full Fine-tuning
```python
config = GradEarlyStoppingConfig(
    tau=1e-10,      # Very low threshold for FFT
    alpha=0.90,     # Late freezing (90% progress)
    enable_wandb_logging=True
)
```

### Vision-Language Models
```python
from grades import VLMGradEarlyStoppingCallback, VLMGradEarlyStoppingConfig

config = VLMGradEarlyStoppingConfig(
    tau=1e-4,       # Standard threshold for VLMs
    alpha=0.3,      # Moderate freezing point
    enable_wandb_logging=True
)
```

## 🖼️ Coming Soon

- **Hugging Face VLM + LoRA**: Vision-language model fine-tuning with LoRA
- **Hugging Face VLM + FFT**: Full fine-tuning for vision-language models
- **Multi-GPU Training**: Distributed training examples
- **Custom Datasets**: Examples with different dataset formats

## 📊 Expected Results

- **Computational Savings**: 40-50% reduction in training time
- **Memory Efficiency**: Reduced GPU memory usage through component freezing
- **Performance**: Maintained or improved model quality
- **Monitoring**: Real-time tracking via WandB integration

## 🤝 Contributing

Found an issue or want to add an example? Please submit a PR or open an issue!

## 📖 Paper Reference

```bibtex
@article{wen2024grades,
  title={GradES: Significantly Faster Training in Transformers with Gradient-Based Early Stopping},
  author={Wen, Qifu and Zeng, Xi and Zhou, Zihan and Liu, Shuaijun and Hosseinzadeh, Mehdi and Rawassizadeh, Reza},
  journal={arXiv preprint arXiv:2509.01842},
  year={2024}
}
```