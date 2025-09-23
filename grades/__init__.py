"""
GradES: Gradient-based Early Stopping for Efficient Fine-tuning

A PyTorch library implementing gradient-based early stopping that monitors gradient
magnitudes during backpropagation and freezes individual transformer components when
their gradients fall below convergence thresholds.

Authors: Qifu Wen, Xi Zeng, Zihan Zhou, Shuaijun Liu, Mehdi Hosseinzadeh, Reza Rawassizadeh
Paper: GradES: Significantly Faster Training in Transformers with Gradient-Based Early Stopping
Link: https://arxiv.org/abs/2509.01842
License: MIT License
"""

from .gradient_early_stopping import (
    GradEarlyStoppingCallback,
    GradEarlyStoppingConfig,
    ComponentStats
)

from .vlm_early_stopping import (
    VLMGradEarlyStoppingCallback,
    VLMGradEarlyStoppingConfig,
    VLMComponentStats
)

__version__ = "1.0.0"
__author__ = "Qifu Wen, Xi Zeng, Zihan Zhou, Shuaijun Liu, Mehdi Hosseinzadeh, Reza Rawassizadeh"
__email__ = "qifu.wen@example.com"
__description__ = "Gradient-based Early Stopping for Efficient Fine-tuning of Large Language Models"
__url__ = "https://github.com/IXZZZ9/GradES"

__all__ = [
    "GradEarlyStoppingCallback",
    "GradEarlyStoppingConfig",
    "ComponentStats",
    "VLMGradEarlyStoppingCallback",
    "VLMGradEarlyStoppingConfig",
    "VLMComponentStats"
]