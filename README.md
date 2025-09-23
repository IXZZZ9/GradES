# 🎯 GradES: Gradient-based Early Stopping

[![PyPI version](https://badge.fury.io/py/grades.svg)](https://badge.fury.io/py/grades)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2509.01842-b31b1b.svg)](https://arxiv.org/abs/2509.01842)

Official implementation of **GradES** - a gradient-based selective training method that dynamically freezes converged modules during fine-tuning to achieve **40-50% computational savings** without sacrificing model performance.

## 📄 Paper
**GradES: Significantly Faster Training in Transformers with Gradient-Based Early Stopping**
*Qifu Wen, Xi Zeng, Zihan Zhou, Shuaijun Liu, Mehdi Hosseinzadeh, Reza Rawassizadeh*
📖 [arXiv:2509.01842](https://arxiv.org/abs/2509.01842)

## 🚀 Quick Installation

### From PyPI
```bash
pip install grades
```

### From Source
```bash
git clone https://github.com/IXZZZ9/GradES.git
cd GradES
pip install -e .
```

### For Development
```bash
git clone https://github.com/IXZZZ9/GradES.git
cd GradES
pip install -e ".[dev,wandb,examples]"
```

## 💡 Quick Start

### Basic Usage with Transformers

```python
from grades import GradEarlyStoppingCallback, GradEarlyStoppingConfig
from transformers import Trainer, TrainingArguments

# Configure GradES
config = GradEarlyStoppingConfig(
    tau=1e-4,           # Convergence threshold
    alpha=0.3,          # Minimum training progress before freezing
    enable_wandb_logging=True
)

# Create callback
callback = GradEarlyStoppingCallback(config)

# Use with any Transformers Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[callback]
)

trainer.train()
```

### Integration with Unsloth (Recommended)

GradES seamlessly integrates with [Unsloth](https://github.com/unslothai/unsloth) for ultra-fast LLM fine-tuning:

#### 🔥 LoRA Fine-tuning
```python
from grades import GradEarlyStoppingCallback, GradEarlyStoppingConfig
from trl import SFTTrainer, SFTConfig

# GradES configuration for LoRA
config = GradEarlyStoppingConfig(
    tau=1e-10,
    alpha=0.1,
    enable_wandb_logging=True,
)
callback = GradEarlyStoppingCallback(config)

# Unsloth SFTTrainer with GradES
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,
    callbacks=[callback],
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="wandb",
        gradient_checkpointing=True,
        bf16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    ),
)
```

#### 🚀 Full Fine-tuning (FFT)
```python
from grades import GradEarlyStoppingCallback, GradEarlyStoppingConfig

# GradES configuration for FFT
config = GradEarlyStoppingConfig(
    tau=1e-10,
    alpha=0.90,  # Higher alpha for FFT
    enable_wandb_logging=True,
)
callback = GradEarlyStoppingCallback(config)

# Set full_finetuning=True in FastLanguageModel.from_pretrained
# Remove FastLanguageModel.get_peft_model LoRA setup

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,
    callbacks=[callback],
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_ratio=0.05,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="wandb",
        gradient_checkpointing=True,
        bf16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    ),
)
```

### Vision-Language Models (VLMs)
```python
from grades import VLMGradEarlyStoppingCallback, VLMGradEarlyStoppingConfig

# Configure for VLMs
vlm_config = VLMGradEarlyStoppingConfig(
    tau=1e-4,
    alpha=0.3,
    enable_wandb_logging=True
)

vlm_callback = VLMGradEarlyStoppingCallback(vlm_config)
```

## 🎯 Try it Now!

### Google Colab Integration

Ready-to-use notebooks with minimal setup:

#### 🦙 LLM Fine-tuning with Unsloth
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)

**Quick Setup:**
1. Open the [Unsloth Llama3.1 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)
2. Add this cell after imports:
   ```python
   !pip install grades
   ```
3. Replace the trainer setup with the GradES examples above
4. Run and enjoy 40-50% faster training! 🚀

#### 🖼️ VLM Fine-tuning (Coming Soon)
- Hugging Face VLM + LoRA notebook
- Hugging Face VLM + FFT notebook

## 📊 Key Results

- ✅ **40-50% computational savings** compared to standard fine-tuning
- ✅ **Maintains or improves** model performance across multiple benchmarks
- ✅ **Tested on**: Qwen3, Phi4, Llama-3.1, and Mistral models (0.6B to 14B parameters)
- ✅ **Compatible with**: LoRA, Full Fine-tuning, and Vision-Language Models
- ✅ **Framework support**: Transformers, TRL, Unsloth

## ⚙️ Configuration Options

### GradEarlyStoppingConfig
```python
config = GradEarlyStoppingConfig(
    tau=1e-4,                    # Convergence threshold
    alpha=0.3,                   # Min training progress before freezing
    max_frozen_ratio=1.0,        # Max fraction of components to freeze
    compute_interval=1,          # Steps between gradient computations
    history_maxlen=1000,         # Gradient history buffer size
    enable_wandb_logging=False,  # WandB logging
    log_interval=10,             # Logging frequency
    save_stats=True,             # Save component statistics
    output_dir="./grades_output" # Output directory
)
```

## 🏗️ Package Structure

```
grades/
├── __init__.py                 # Main exports
├── gradient_early_stopping.py # LLM early stopping
└── vlm_early_stopping.py      # VLM early stopping
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📖 Citation

If you find GradES useful in your research, please cite:

```bibtex
@article{wen2024grades,
  title={GradES: Significantly Faster Training in Transformers with Gradient-Based Early Stopping},
  author={Wen, Qifu and Zeng, Xi and Zhou, Zihan and Liu, Shuaijun and Hosseinzadeh, Mehdi and Rawassizadeh, Reza},
  journal={arXiv preprint arXiv:2509.01842},
  year={2024}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- 📖 **Paper**: [arXiv:2509.01842](https://arxiv.org/abs/2509.01842)
- 🐙 **GitHub**: [IXZZZ9/GradES](https://github.com/IXZZZ9/GradES)
- 📦 **PyPI**: [grades](https://pypi.org/project/grades/)
- 🤗 **Hugging Face**: [Coming Soon]
- 🦙 **Unsloth Integration**: [Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)

---

**Made with ❤️ by the GradES Team**
