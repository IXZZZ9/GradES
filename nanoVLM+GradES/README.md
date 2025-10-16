# nanoVLM with GradES Integration

This project provides an implementation of `nanoVLM` integrated with **GradES** (Gradient-based Early Stopping) for efficient fine-tuning. By dynamically freezing converged model components, GradES can significantly reduce training time and computational cost.

This guide explains how to set up the project and run a training session with GradES enabled.

## 1. Initial Setup

First, clone the original `nanoVLM` repository from Hugging Face:

```bash
git clone https://github.com/huggingface/nanoVLM.git
```

This will create a `nanoVLM` directory with the original project structure.

## 2. Apply GradES Modifications

Next, move the files from this `nanoVLM+GradES` directory into your cloned `nanoVLM` repository, replacing the existing files.

- **`train.py`**: This is the modified training script that incorporates the GradES callback.
- **`models/config.py`**: This configuration file includes a new `GradESConfig` section to manage the early stopping parameters.
- **`models/gradient_early_stopping.py`**: This is the core implementation of the GradES callback, tailored for nanoVLM.

You can do this with the following commands:

```bash
# Assuming you are in the directory containing both nanoVLM and nanoVLM+GradES
mv nanoVLM+GradES/train.py nanoVLM/train.py
mv nanoVLM+GradES/models/config.py nanoVLM/models/config.py
mv nanoVLM+GradES/models/gradient_early_stopping.py nanoVLM/models/gradient_early_stopping.py
```

## 3. Environment Setup

Navigate into the `nanoVLM` directory and set up your Python environment. We recommend using `uv`:

```bash
cd nanoVLM
uv init --bare --python 3.12
uv sync --python 3.12
source .venv/bin/activate
uv add torch numpy torchvision pillow datasets huggingface-hub transformers wandb matplotlib
```

## 4. Running Training with GradES

The modified `train.py` script accepts command-line arguments to configure GradES. To enable GradES, use the `--enable_grades` flag.

Here is an example command to start training with GradES, using the hyperparameters from the official paper for reproducibility:

```bash
python train.py \
    --enable_grades \
    --vision_tau 0.30 \
    --language_tau 6.00 \
    --alpha 0.28 \
    --compute_interval 20 \
    --log_wandb
```

These hyperparameters correspond to the `Training` configuration for nanoVLM in Table 10 of the GradES paper.

### GradES Configuration Arguments:

- `--enable_grades`: (Flag) Activates the GradES callback during training.
- `--vision_tau` (float): Convergence threshold (τ) for ViT components.
- `--language_tau` (float): Convergence threshold (τ) for LLM components.
- `--alpha` (float): The grace period, representing the minimum fraction of training to complete before freezing begins.
- `--compute_interval` (int): How often (in steps) to calculate weight changes.

## 5. Monitoring and Results

When training with GradES, you will see output in your console indicating when components are frozen. At the end of the training run, a convergence plot will be saved to the `grades_output` directory, showing the weight change trends and freezing progress.

If `log_wandb` is enabled, you can monitor the freezing process in real-time on your Weights & Biases dashboard.

---
*This implementation of GradES is based on the research paper: "GradES: Significantly Faster Training in Transformers with Gradient-Based Early Stopping" (arXiv:2509.01842).*
