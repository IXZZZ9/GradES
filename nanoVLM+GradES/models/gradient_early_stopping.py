"""
GradES: Gradient-based Early Stopping for nanoVLM
=================================================

This module provides an implementation of the GradES methodology, optimized for
the nanoVLM architecture. It monitors weight changes (using L1 norm) during
fine-tuning and dynamically freezes components (ViT and LLM blocks) when they
converge. This leads to significant computational savings without sacrificing
model performance.

This implementation is designed to be a standalone component for easy integration
with the nanoVLM training pipeline.

Authors: Qifu Wen, Xi Zeng, Zihan Zhou, Shuaijun Liu, Mehdi Hosseinzadeh, Ningxin Su, Reza Rawassizadeh
Paper: GradES: Significantly Faster Training in Transformers with Gradient-Based Early Stopping
Link: https://arxiv.org/abs/2509.01842
"""

import torch
import torch.nn as nn
from typing import Dict, List
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class GradEarlyStoppingConfig:
    """
    Configuration for GradES tailored for nanoVLM.

    Args:
        vision_tau (float): Convergence threshold for Vision Transformer (ViT) components.
        language_tau (float): Convergence threshold for LLM components.
        alpha (float): Minimum fraction of total training steps before freezing can begin.
        max_frozen_ratio (float): Early stopping threshold. Training stops when this
                                  fraction of components is frozen.
        compute_interval (int): Frequency (in steps) for computing weight changes.
        use_cuda_acceleration (bool): If True, uses GPU for weight change calculations.
        enable_wandb_logging (bool): Placeholder for future wandb integration.
        output_dir (str): Directory to save plots and statistics.
    """
    vision_tau: float = 1e-4
    language_tau: float = 1e-5
    alpha: float = 0.3
    max_frozen_ratio: float = 0.8
    compute_interval: int = 10
    use_cuda_acceleration: bool = True
    enable_wandb_logging: bool = False # Placeholder
    output_dir: str = "grades_output"


class ComponentTracker:
    """
    Tracks the training progress of a single model component (e.g., an attention block).
    """
    def __init__(self, name: str, component_type: str, device: torch.device):
        """
        Initializes the tracker for a component.

        Args:
            name (str): A unique name for the component (e.g., 'vit_block_0_attn_qkv_proj').
            component_type (str): The type of component, either 'vision' or 'language'.
            device (torch.device): The device (CPU or CUDA) for computation.
        """
        self.name = name
        self.component_type = component_type
        self.device = device
        self.is_frozen = False
        self.frozen_at_step = None
        self.weight_cache = None
        self.current_change_norm = 0.0
        self.param_count = 0
        self.total_steps = 0
        self.change_history = []  # List of (step, l1_norm_change)

    def calculate_weight_change(self, module: nn.Module, step: int):
        """
        Calculates the L1 norm of the weight change since the last update.

        Args:
            module (nn.Module): The neural network module to track.
            step (int): The current training step.
        """
        if self.is_frozen:
            return

        with torch.no_grad():
            # Collect all trainable parameters into a single flat tensor.
            current_weights = [p.data.flatten() for p in module.parameters() if p.requires_grad]
            if not current_weights:
                return
            
            current_tensor = torch.cat(current_weights).to(self.device, non_blocking=True)

            # Initialize the cache on the first call.
            if self.weight_cache is None:
                self.weight_cache = current_tensor.clone()
                self.param_count = current_tensor.numel()
                self.total_steps = step
                return

            # Calculate the L1 norm of the difference.
            change_norm = torch.norm(current_tensor - self.weight_cache, p=1).item()

            # Update cache and history.
            self.weight_cache.copy_(current_tensor)
            self.current_change_norm = change_norm
            self.total_steps = step
            self.change_history.append((step, change_norm))

    def should_freeze(self, threshold: float, min_steps: int) -> bool:
        """
        Determines if the component has converged and should be frozen.

        Args:
            threshold (float): The convergence threshold (tau).
            min_steps (int): The minimum number of steps before freezing (alpha * total_steps).

        Returns:
            bool: True if the component should be frozen, False otherwise.
        """
        if self.is_frozen or self.total_steps < min_steps:
            return False
        return self.current_change_norm < threshold

    def freeze(self, step: int):
        """
        Marks the component as frozen and releases its weight cache from memory.
        """
        self.is_frozen = True
        self.frozen_at_step = step
        self.weight_cache = None  # Free memory
        self.current_change_norm = 0.0


class GradEarlyStoppingCallback:
    """
    Implements the GradES callback for the nanoVLM training loop.
    """
    def __init__(self, config: GradEarlyStoppingConfig):
        """
        Initializes the callback.

        Args:
            config (GradEarlyStoppingConfig): The configuration for GradES.
        """
        self.config = config
        self.trackers: Dict[str, ComponentTracker] = {}
        self.step = 0
        self.frozen_count = 0
        self.total_components = 0
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda_acceleration else 'cpu')
        self.start_time = None
        self.freeze_events = []

    def register_model(self, model: nn.Module):
        """
        Registers the nanoVLM model and initializes trackers for all target components.

        Args:
            model (nn.Module): The nanoVLM model.
        """
        self.start_time = time.time()
        self.model = model

        # Register Vision Transformer (ViT) components
        vit_count = self._register_vit_components(model.vision_encoder)
        
        # Register Language Model (LLM) components
        llm_count = self._register_llm_components(model.decoder)
        
        self.total_components = len(self.trackers)
        print(f"[GradES] Registered {self.total_components} components ({vit_count} ViT, {llm_count} LLM) "
              f"using {'GPU' if self.device.type == 'cuda' else 'CPU'}")

    def _register_vit_components(self, vision_encoder: nn.Module) -> int:
        """Helper to register all target modules within the ViT."""
        count = 0
        if hasattr(vision_encoder, 'blocks'):
            for i, block in enumerate(vision_encoder.blocks):
                if hasattr(block, 'attn'):
                    for name in ['qkv_proj', 'out_proj']:
                        if hasattr(block.attn, name):
                            self._register_component(f"vision_L{i}_attn_{name}", getattr(block.attn, name), 'vision')
                            count += 1
                if hasattr(block, 'mlp'):
                    for name in ['fc1', 'fc2']:
                        if hasattr(block.mlp, name):
                            self._register_component(f"vision_L{i}_mlp_{name}", getattr(block.mlp, name), 'vision')
                            count += 1
        return count

    def _register_llm_components(self, decoder: nn.Module) -> int:
        """Helper to register all target modules within the LLM."""
        count = 0
        if hasattr(decoder, 'blocks'):
            for i, block in enumerate(decoder.blocks):
                if hasattr(block, 'attn'):
                    for name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                        if hasattr(block.attn, name):
                            self._register_component(f"language_L{i}_attn_{name}", getattr(block.attn, name), 'language')
                            count += 1
                if hasattr(block, 'mlp'):
                    for name in ['gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(block.mlp, name):
                            self._register_component(f"language_L{i}_mlp_{name}", getattr(block.mlp, name), 'language')
                            count += 1
        return count

    def _register_component(self, name: str, module: nn.Module, component_type: str):
        """Creates and registers a tracker for a single module if it has trainable parameters."""
        if any(p.requires_grad for p in module.parameters()):
            tracker = ComponentTracker(name, component_type, self.device)
            tracker.module_ref = module  # Store a reference to the module
            self.trackers[name] = tracker

    def on_step_end(self, total_steps: int) -> bool:
        """
        Should be called at the end of each training step. It updates weight changes,
        freezes converged components, and checks for early stopping.

        Args:
            total_steps (int): The total number of training steps planned.

        Returns:
            bool: True if training should stop early, False otherwise.
        """
        self.step += 1

        # Only compute changes at the specified interval.
        if self.step % self.config.compute_interval == 0:
            self._calculate_weight_changes()
            self._check_and_freeze_components(total_steps)

        # Check for early stopping condition.
        frozen_ratio = self.frozen_count / max(1, self.total_components)
        return frozen_ratio >= self.config.max_frozen_ratio

    def _calculate_weight_changes(self):
        """Iterates through active components and calculates their weight changes."""
        for tracker in self.trackers.values():
            if not tracker.is_frozen:
                tracker.calculate_weight_change(tracker.module_ref, self.step)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def _check_and_freeze_components(self, total_steps: int):
        """Checks for convergence and freezes components that meet the criteria."""
        min_steps_before_freeze = int(total_steps * self.config.alpha)
        
        for name, tracker in self.trackers.items():
            threshold = self.config.vision_tau if tracker.component_type == 'vision' else self.config.language_tau
            
            if tracker.should_freeze(threshold, min_steps_before_freeze):
                self._freeze_component(name)

    def _freeze_component(self, name: str):
        """Freezes a component by disabling gradients for its parameters."""
        tracker = self.trackers[name]
        if tracker.is_frozen:
            return

        # Disable gradients for all parameters in the module.
        for param in tracker.module_ref.parameters():
            param.requires_grad = False
        
        tracker.freeze(self.step)
        self.frozen_count += 1
        
        self.freeze_events.append({
            'step': self.step,
            'component': name,
            'type': tracker.component_type,
            'param_count': tracker.param_count
        })
        
        print(f"[GradES] Step {self.step}: Froze {name} ({self.frozen_count}/{self.total_components} total)")

    def on_train_end(self):
        """
        Should be called at the end of training to generate plots and clean up.
        """
        print("[GradES] Training finished. Generating convergence plots...")
        self.plot_convergence_history()
        self.cleanup()

    def cleanup(self):
        """Releases memory used by the callback."""
        for tracker in self.trackers.values():
            tracker.freeze(-1) # Releases weight cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def plot_convergence_history(self):
        """
        Generates and saves plots visualizing the convergence and freezing history.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), constrained_layout=True)
        fig.suptitle('GradES Convergence History for nanoVLM', fontsize=16)
        
        # Plot 1: Average Weight Change Trend
        ax1 = axes[0]
        vit_step_changes = defaultdict(list)
        llm_step_changes = defaultdict(list)
        
        for tracker in self.trackers.values():
            for step, change in tracker.change_history:
                if tracker.component_type == 'vision':
                    vit_step_changes[step].append(change)
                else:
                    llm_step_changes[step].append(change)

        if vit_step_changes:
            steps = sorted(vit_step_changes.keys())
            means = [np.mean(vit_step_changes[s]) for s in steps]
            ax1.plot(steps, means, 'b-', label='Vision Mean Change')
        
        if llm_step_changes:
            steps = sorted(llm_step_changes.keys())
            means = [np.mean(llm_step_changes[s]) for s in steps]
            ax1.plot(steps, means, 'orange', label='Language Mean Change')

        ax1.axhline(y=self.config.vision_tau, color='blue', linestyle='--', label=f'Vision Tau ({self.config.vision_tau:.1e})')
        ax1.axhline(y=self.config.language_tau, color='orange', linestyle='--', label=f'Language Tau ({self.config.language_tau:.1e})')
        ax1.set_title('Average Component Weight Change')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('L1 Norm of Weight Change')
        ax1.legend()
        ax1.grid(True, alpha=0.5)
        ax1.set_yscale('log')

        # Plot 2: Freezing Progress
        ax2 = axes[1]
        if self.freeze_events:
            steps = [e['step'] for e in self.freeze_events]
            vit_frozen = np.cumsum([1 if e['type'] == 'vision' else 0 for e in self.freeze_events])
            llm_frozen = np.cumsum([1 if e['type'] == 'language' else 0 for e in self.freeze_events])
            
            ax2.plot(steps, vit_frozen, 'b-o', markersize=4, label='Vision Components Frozen')
            ax2.plot(steps, llm_frozen, 'orange', marker='s', markersize=4, label='Language Components Frozen')

        ax2.set_title('Component Freezing Progress')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Cumulative Number of Frozen Components')
        ax2.legend()
        ax2.grid(True, alpha=0.5)

        plot_file = output_dir / "grades_convergence_history.png"
        plt.savefig(plot_file, dpi=300)
        plt.close(fig)
        print(f"[GradES] Convergence plot saved to: {plot_file}")
