"""
Gradient-based Early Stopping Callback for Transformers

A TrainerCallback implementing gradient-based early stopping that monitors gradient magnitudes 
during backpropagation and freezes individual transformer components (attention/FFN matrices) when 
their gradients fall below convergence threshold τ, eliminating costly validation passes while 
allowing slow-converging parameters to continue learning.

Authors: Qifu Wen, Xi Zeng, Zihan Zhou, Shuaijun Liu, Mehdi Hosseinzadeh, Reza Rawassizadeh
Paper: GradES: Significantly Faster Training in Transformers with Gradient-Based Early Stopping
Link: https://arxiv.org/abs/2509.01842
License: MIT License
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

__version__ = "1.0.0"
__all__ = ["GradEarlyStoppingCallback", "GradEarlyStoppingConfig", "ComponentStats"]

logger = logging.getLogger(__name__)

# Define constants
DEFAULT_HISTORY_MAXLEN = 1000  # Maximum change history entries per component
LORA_DETECTION_LAYERS = 3      # Number of layers to check for LoRA detection
ATTENTION_COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_COMPONENTS = ["gate_proj", "up_proj", "down_proj"]

@dataclass
class GradEarlyStoppingConfig:
    """
    Configuration for the GradEarlyStoppingCallback.

    Args:
        tau (`float`, *optional*, defaults to 1e-4):
            Weight change threshold below which a component is considered converged and frozen.
        alpha (`float`, *optional*, defaults to 0.3):
            Minimum fraction of total training steps before freezing is allowed.
        max_frozen_ratio (`float`, *optional*, defaults to 1.0):
            Maximum fraction of components that can be frozen before early stopping.
        compute_interval (`int`, *optional*, defaults to 1):
            Number of steps between weight change computations.
        target_components (`List[str]`, *optional*):
            List of component names to monitor. Defaults to common transformer components.
        auto_detect_mode (`bool`, *optional*, defaults to True):
            Whether to automatically detect LoRA vs full parameter fine-tuning.
        use_cuda_acceleration (`bool`, *optional*, defaults to True):
            Whether to use CUDA acceleration for weight change computations.
        save_freezing_history (`bool`, *optional*, defaults to False):
            Whether to save detailed freezing history to file.
        enable_wandb_logging (`bool`, *optional*, defaults to False):
            Whether to log component change metrics to wandb during training.
        output_dir (`str`, *optional*):
            Directory to save freezing history and statistics. If None, uses trainer's output_dir.
    """

    tau: float = 1e-4
    alpha: float = 0.3
    max_frozen_ratio: float = 1.0
    compute_interval: int = 1
    target_components: List[str] = field(default_factory=lambda:
        ATTENTION_COMPONENTS + MLP_COMPONENTS
    )
    auto_detect_mode: bool = True
    use_cuda_acceleration: bool = True
    save_freezing_history: bool = False
    enable_wandb_logging: bool = False
    output_dir: Optional[str] = None


@dataclass
class ComponentStats:
    """Statistics for tracking component weight changes."""

    name: str
    component_type: str  # 'lora' or 'full'
    frozen: bool = False
    frozen_at_step: Optional[int] = None
    change_history: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_MAXLEN))
    weight_cache: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None
    total_change_accumulated: float = 0.0
    param_count: int = 0

    def add_change(self, change: float, step: int):
        """Records a weight change for a given step."""
        self.change_history.append((step, change))
        self.total_change_accumulated += change


class GradEarlyStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that dynamically freezes model parameters during training
    based on weight change convergence.

    This callback monitors the weight changes of specified components during training and
    freezes them when their changes fall below a threshold, indicating convergence.
    It supports both LoRA adapter fine-tuning and full parameter fine-tuning.

    Args:
        config (`GradEarlyStoppingConfig`, *optional*):
            Configuration for the dynamic freezing behavior. If None, uses default config.

    Example:
        ```python
        from transformers import Trainer

        # Create callback with custom configuration
        freeze_config = GradEarlyStoppingConfig(
            tau=1e-5,
            alpha=0.2,
            compute_interval=50
        )
        freeze_callback = GradEarlyStoppingCallback(config=freeze_config)

        # Add to trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            callbacks=[freeze_callback],
            ...
        )
        ```

    Logging Policy:
        - logger.info: Major events (initialization, freezing, early stopping)
        - logger.warning: Recoverable issues (OOM, missing wandb)
        - logger.error: Critical errors that prevent normal operation
        - logger.debug: Detailed debugging info and non-critical errors

    Note:
        This callback works best when used with gradient accumulation and mixed precision training.
        For optimal performance, ensure your TrainingArguments includes appropriate settings for
        these features.
    """

    DEFAULT_HISTORY_MAXLEN = DEFAULT_HISTORY_MAXLEN
    LORA_DETECTION_LAYERS = LORA_DETECTION_LAYERS
    ATTENTION_COMPONENTS = ATTENTION_COMPONENTS
    MLP_COMPONENTS = MLP_COMPONENTS

    def __init__(self, config: Optional[GradEarlyStoppingConfig] = None):
        self.config = config or GradEarlyStoppingConfig()

        # Validate configuration
        if not 0 < self.config.tau:
            raise ValueError(f"tau must be greater than 0, got {self.config.tau}")
        if not 0 <= self.config.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.config.alpha}")
        if not 0 < self.config.max_frozen_ratio <= 1:
            raise ValueError(f"max_frozen_ratio must be between 0 and 1, got {self.config.max_frozen_ratio}")
        if self.config.compute_interval < 1:
            raise ValueError(f"compute_interval must be >= 1, got {self.config.compute_interval}")

        # Core tracking structures
        self.component_stats: Dict[str, ComponentStats] = {}
        self.frozen_components: Set[str] = set()

        # Training mode detection
        self.mode: Optional[str] = None  # 'lora' or 'full'
        self.model_layers_cache = None

        # Device management
        self.device = None
        self.cuda_available = False

        # Global statistics
        self.start_time: Optional[float] = None
        self.current_training_step: int = 0
        self.total_training_steps: int = 0
        self.all_components_frozen_at_step: Optional[int] = None
        self.frozen_events: List[Dict] = []

        # Wandb logging
        self.wandb_available: bool = False

        # For state export
        self.initialized = False

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                       control: TrainerControl, model=None, **kwargs):
        """Initialize the callback and detect training mode."""

        # Record start time
        self.start_time = time.time()

        # Set output directory
        if self.config.output_dir is None:
            self.config.output_dir = args.output_dir

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                  self.config.use_cuda_acceleration else "cpu")
        self.cuda_available = torch.cuda.is_available() and self.config.use_cuda_acceleration

        # Initialize total training steps
        self.total_training_steps = state.max_steps

        # Check wandb availability
        if self.config.enable_wandb_logging:
            try:
                import wandb
                self.wandb_available = wandb.run is not None
                if self.wandb_available:
                    logger.info("Wandb integration enabled for layer-wise component tracking")
                    # Update config with GradES-specific parameters
                    wandb.config.update({
                        "grades_mode": self.mode,
                        "grades_tau": self.config.tau,
                        "grades_alpha": self.config.alpha,
                        "grades_max_frozen_ratio": self.config.max_frozen_ratio,
                        "grades_compute_interval": self.config.compute_interval,
                    })
            except ImportError:
                logger.warning("Wandb not available. Install with: pip install wandb")

        # Detect training mode
        if self.config.auto_detect_mode:
            self.mode = self._detect_training_mode(model)
            logger.info(f"GradEarlyStoppingCallback detected {self.mode.upper()} parameter fine-tuning")
        else:
            self.mode = 'full'  # Default to full if not auto-detecting
            logger.info("GradEarlyStoppingCallback using full parameter mode (auto_detect disabled)")

        # Initialize component tracking
        initialized_count = self._initialize_component_tracking(model)

        if initialized_count == 0:
            logger.warning(
                "GradEarlyStoppingCallback found no components to track. "
                "Check that your model has trainable parameters in the target components."
            )
            self.initialized = False
            return

        self.initialized = True

        # Log initialization summary
        total_params = sum(stats.param_count for stats in self.component_stats.values())
        logger.info(
            f"GradEarlyStoppingCallback initialized: "
            f"tracking {initialized_count} components with {total_params:,} parameters"
        )

        if self.cuda_available:
            logger.info("CUDA acceleration enabled for weight change computations")

    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        """Monitor components and freeze converged ones."""

        if not self.initialized:
            return control

        current_step = state.global_step
        self.current_training_step = current_step

        # Calculate alpha-based threshold: components can only be frozen after
        # completing alpha fraction of total training steps to ensure adequate learning
        min_steps_before_freeze = int(self.total_training_steps * self.config.alpha)

        # Only compute changes at specified intervals
        if current_step > 0 and current_step % self.config.compute_interval == 0:

            # Get active (non-frozen) components
            active_components = [k for k, v in self.component_stats.items() if not v.frozen]

            if not active_components:
                # All components are frozen
                if self._should_stop_training():
                    control.should_training_stop = True
                return control

            # Process each active component
            components_to_freeze = []
            component_changes = defaultdict(list)

            for component_key in active_components:
                component = self._get_component_by_key(model, component_key)
                if component is None:
                    continue

                try:
                    # Calculate weight change
                    stats = self.component_stats[component_key]
                    change = self._calculate_component_change(component, stats)
                    stats.add_change(change, current_step)

                    # Track changes by component type for wandb
                    parts = component_key.split('_')
                    comp_type = f"{parts[2]}_{parts[3]}"  # e.g., "attn_q_proj", "mlp_gate_proj"
                    component_changes[comp_type].append(change)

                    # Check if component should be frozen
                    if (current_step >= min_steps_before_freeze and 
                        change < self.config.tau):
                        components_to_freeze.append(component_key)

                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM while processing {component_key}, clearing cache")
                    if self.cuda_available:
                        torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    logger.error(f"Error processing component {component_key}: {e}")
                    continue

            # Freeze converged components
            if components_to_freeze:
                for component_key in components_to_freeze:
                    self._freeze_component(model, component_key, current_step)

                # Clear CUDA cache after freezing
                if self.cuda_available:
                    torch.cuda.empty_cache()

                logger.info(
                    f"Step {current_step}: Frozen {len(components_to_freeze)} components. "
                    f"Total frozen: {len(self.frozen_components)}/{len(self.component_stats)}"
                )

            # Log to wandb if available
            if self.wandb_available:
                self._log_to_wandb(current_step, component_changes)

            # Check if all components are now frozen
            if self.all_components_frozen_at_step is None and len(self.frozen_components) == len(self.component_stats):
                self.all_components_frozen_at_step = current_step
                logger.info(f"All components frozen at step {current_step}")
                if self._should_stop_training():
                    control.should_training_stop = True

        # Check early stopping based on frozen ratio
        if self._should_stop_training():
            frozen_ratio = len(self.frozen_components) / len(self.component_stats) if self.component_stats else 0
            logger.info(
                f"Early stopping triggered at step {current_step}: "
                f"{frozen_ratio:.1%} components frozen (threshold: {self.config.max_frozen_ratio:.1%})"
            )
            control.should_training_stop = True

        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, model=None, **kwargs):
        """Save final statistics and clean up."""

        if not self.initialized:
            return

        total_time = time.time() - self.start_time if self.start_time else 0

        logger.info(
            f"Final state: {len(self.frozen_components)}/{len(self.component_stats)} components frozen"
        )

        # Save statistics if requested
        if self.config.save_freezing_history:
            self._save_freezing_history(total_time)

        # Clean up memory
        self._cleanup_memory()

    def state(self) -> dict:
        """
        Export callback state for checkpointing and resumption.

        Returns:
            dict: Serializable state containing configuration and runtime statistics
                  including frozen components, training progress, and freezing events.
        """
        return {
            "config": {
                "tau": self.config.tau,
                "alpha": self.config.alpha,
                "max_frozen_ratio": self.config.max_frozen_ratio,
                "compute_interval": self.config.compute_interval,
                "target_components": self.config.target_components,
                "auto_detect_mode": self.config.auto_detect_mode,
            },
            "attributes": {
                "mode": self.mode,
                "frozen_components": list(self.frozen_components),
                "all_components_frozen_at_step": self.all_components_frozen_at_step,
                "frozen_events": self.frozen_events,
                "current_training_step": self.current_training_step,
                "initialized": self.initialized,
            }
        }

    # ============= Private Helper Methods =============

    def _detect_training_mode(self, model) -> str:
        """Detect whether using LoRA or full parameter fine-tuning."""
        layers = self._get_model_layers(model)
        if not layers:
            return 'full'

        # Check first few layers for LoRA adapters
        for layer in layers[:self.LORA_DETECTION_LAYERS]:
            if hasattr(layer, 'self_attn'):
                for component_name in self.config.target_components[:4]:
                    if hasattr(layer.self_attn, component_name):
                        component = getattr(layer.self_attn, component_name)
                        if self._has_lora(component):
                            return 'lora'
        return 'full'

    def _has_lora(self, component) -> bool:
        """Check if a component has LoRA adapters."""
        return (hasattr(component, 'lora_A') and 
                hasattr(component, 'lora_B') and 
                len(getattr(component, 'lora_A', {})) > 0)

    def _initialize_component_tracking(self, model) -> int:
        """Initialize tracking for all target components."""
        layers = self._get_model_layers(model)
        if not layers:
            return 0

        initialized_count = 0

        for i, layer in enumerate(layers):
            # Check attention components
            if hasattr(layer, 'self_attn'):
                for component_name in self.config.target_components[:4]:
                    if hasattr(layer.self_attn, component_name):
                        component = getattr(layer.self_attn, component_name)

                        # Check if component should be tracked
                        should_track = False
                        param_count = 0

                        if self.mode == 'lora':
                            should_track = self._has_lora(component)
                            if should_track:
                                lora_a, lora_b = self._extract_lora_weight_matrices(component)
                                if lora_a is not None and lora_b is not None:
                                    param_count = lora_a.numel() + lora_b.numel()
                        else:
                            should_track = any(p.requires_grad for p in component.parameters())
                            if should_track:
                                param_count = sum(p.numel() for p in component.parameters() if p.requires_grad)

                        if should_track:
                            key = f"layer_{i:02d}_attn_{component_name}"
                            self.component_stats[key] = ComponentStats(
                                name=key,
                                component_type=self.mode,
                                param_count=param_count
                            )
                            initialized_count += 1

            # Check MLP components
            if hasattr(layer, 'mlp'):
                for component_name in self.config.target_components[4:]:
                    if hasattr(layer.mlp, component_name):
                        component = getattr(layer.mlp, component_name)

                        should_track = False
                        param_count = 0

                        if self.mode == 'lora':
                            should_track = self._has_lora(component)
                            if should_track:
                                lora_a, lora_b = self._extract_lora_weight_matrices(component)
                                if lora_a is not None and lora_b is not None:
                                    param_count = lora_a.numel() + lora_b.numel()
                        else:
                            should_track = any(p.requires_grad for p in component.parameters())
                            if should_track:
                                param_count = sum(p.numel() for p in component.parameters() if p.requires_grad)

                        if should_track:
                            key = f"layer_{i:02d}_mlp_{component_name}"
                            self.component_stats[key] = ComponentStats(
                                name=key,
                                component_type=self.mode,
                                param_count=param_count
                            )
                            initialized_count += 1

        return initialized_count

    def _extract_lora_weight_matrices(self, component) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get LoRA A and B matrices from a component."""
        if hasattr(component, 'lora_A') and hasattr(component, 'lora_B'):
            if 'default' in component.lora_A and 'default' in component.lora_B:
                a = component.lora_A['default']
                b = component.lora_B['default']
                if a is not None and b is not None:
                    return a.weight, b.weight
        return None, None

    def _calculate_component_change(self, component, stats: ComponentStats) -> float:
        """Calculate weight change for a component."""
        if self.mode == 'lora':
            return self._calculate_lora_change(component, stats)
        else:
            return self._calculate_full_param_change(component, stats)

    def _calculate_lora_change(self, component, stats: ComponentStats) -> float:
        """Calculate change for LoRA components."""
        lora_a, lora_b = self._extract_lora_weight_matrices(component)
        if lora_a is None or lora_b is None:
            return 0.0

        try:
            with torch.no_grad():
                # Use parameters' native device (don't force transfer)
                device = lora_a.device
                use_non_blocking = device.type == 'cuda' and self.cuda_available

                # Initialize cache on first call
                if stats.weight_cache is None:
                    stats.weight_cache = {
                        'lora_a': lora_a.clone(),
                        'lora_b': lora_b.clone()
                    }
                    return 0.0

                # Ensure cache is on same device as current weights
                if stats.weight_cache['lora_a'].device != device:
                    stats.weight_cache['lora_a'] = stats.weight_cache['lora_a'].to(device, non_blocking=use_non_blocking)
                    stats.weight_cache['lora_b'] = stats.weight_cache['lora_b'].to(device, non_blocking=use_non_blocking)

                # Calculate L1 norm of change
                delta_a = torch.norm(lora_a - stats.weight_cache['lora_a'], p=1)
                delta_b = torch.norm(lora_b - stats.weight_cache['lora_b'], p=1)
                total_change = (delta_a + delta_b).item()

                # Update cache
                stats.weight_cache['lora_a'].copy_(lora_a, non_blocking=use_non_blocking)
                stats.weight_cache['lora_b'].copy_(lora_b, non_blocking=use_non_blocking)

                return total_change

        except Exception as e:
            logger.debug(f"Error calculating LoRA change: {e}")
            return 0.0

    def _calculate_full_param_change(self, component, stats: ComponentStats) -> float:
        """Calculate change for full parameter components."""
        try:
            with torch.no_grad():
                # Get current weights
                current_weights = []
                device = None

                for param in component.parameters():
                    if param.requires_grad:
                        current_weights.append(param.data.flatten())
                        if device is None:
                            device = param.device

                if not current_weights:
                    return 0.0

                # Concatenate all parameters (keep on native device)
                current_tensor = torch.cat(current_weights)
                use_non_blocking = device.type == 'cuda' and self.cuda_available

                # Initialize cache on first call
                if stats.weight_cache is None:
                    stats.weight_cache = current_tensor.clone()
                    return 0.0

                # Ensure cache is on same device as current weights
                if stats.weight_cache.device != device:
                    stats.weight_cache = stats.weight_cache.to(device, non_blocking=use_non_blocking)

                # Calculate L1 norm of change
                change = torch.norm(current_tensor - stats.weight_cache, p=1).item()

                # Update cache
                stats.weight_cache.copy_(current_tensor, non_blocking=use_non_blocking)

                return change

        except Exception as e:
            logger.debug(f"Error calculating full param change: {e}")
            return 0.0

    def _freeze_component(self, model, component_key: str, step: int):
        """Freeze a component's parameters."""
        if component_key in self.frozen_components:
            return

        component = self._get_component_by_key(model, component_key)
        if component is None:
            return

        frozen_count = 0

        if self.mode == 'lora':
            # Freeze LoRA parameters
            for param_name in ['lora_A', 'lora_B']:
                if hasattr(component, param_name):
                    for adapter_name in getattr(component, param_name):
                        param = getattr(component, param_name)[adapter_name].weight
                        if param.requires_grad:
                            param.requires_grad = False
                            frozen_count += 1
        else:
            # Freeze all parameters
            for param in component.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1

        # Clear weight cache to save memory
        if component_key in self.component_stats:
            self.component_stats[component_key].weight_cache = None

        # Update statistics
        stats = self.component_stats[component_key]
        stats.frozen = True
        stats.frozen_at_step = step
        self.frozen_components.add(component_key)

        # Record event
        self.frozen_events.append({
            "step": step,
            "component": component_key,
            "mode": self.mode,
            "param_count": stats.param_count
        })

    def _should_stop_training(self) -> bool:
        """Check if training should stop based on frozen ratio."""
        if not self.component_stats:
            return False
        frozen_ratio = len(self.frozen_components) / len(self.component_stats)
        return frozen_ratio >= self.config.max_frozen_ratio

    def _get_model_layers(self, model: torch.nn.Module) -> Optional[List[torch.nn.Module]]:
        """Get model layers with caching."""
        if self.model_layers_cache is None:
            # Attempt to locate model layers through common HuggingFace model hierarchies.
            # Different models (BERT, GPT, LLaMA) have varying attribute paths to access layers
            for attr_path in [
                ['base_model', 'model', 'model', 'layers'],
                ['base_model', 'model', 'layers'],
                ['model', 'layers'],
                ['layers']
            ]:
                obj = model
                for attr in attr_path:
                    if hasattr(obj, attr):
                        obj = getattr(obj, attr)
                    else:
                        break
                else:
                    self.model_layers_cache = obj
                    break

        return self.model_layers_cache

    def _get_component_by_key(self, model: torch.nn.Module, component_key: str) -> Optional[torch.nn.Module]:
        """Get a component by its tracking key."""
        try:
            parts = component_key.split('_')
            layer_idx = int(parts[1])
            component_type = parts[2]  # 'attn' or 'mlp'
            component_name = '_'.join(parts[3:])

            layers = self._get_model_layers(model)
            if layers and layer_idx < len(layers):
                layer = layers[layer_idx]
                parent = getattr(layer, 'self_attn' if component_type == 'attn' else 'mlp', None)
                if parent:
                    return getattr(parent, component_name, None)
        except Exception as e:
            logger.debug(f"Error getting component {component_key}: {e}")
        return None

    def _save_freezing_history(self, total_time: float):
        """Save detailed freezing history to file."""
        if not self.config.output_dir:
            logger.warning("No output directory specified, skipping history save")
            return

        try:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create output directory: {e}")
            return

        # Group components by type for summary
        component_summary = defaultdict(lambda: {"total": 0, "frozen": 0})
        for component_key, stats in self.component_stats.items():
            parts = component_key.split('_')
            comp_type = f"{parts[2]}_{parts[3]}"
            component_summary[comp_type]["total"] += 1
            if stats.frozen:
                component_summary[comp_type]["frozen"] += 1

        # Prepare statistics
        stats_to_save = {
            "metadata": {
                "version": __version__,
                "mode": self.mode,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "config": {
                "tau": self.config.tau,
                "alpha": self.config.alpha,
                "max_frozen_ratio": self.config.max_frozen_ratio,
                "compute_interval": self.config.compute_interval,
                "target_components": self.config.target_components
            },
            "summary": {
                "total_time_seconds": round(total_time, 2),
                "total_time_formatted": f"{int(total_time//60)}m {int(total_time%60)}s",
                "current_training_step": self.current_training_step,
                "total_training_steps": self.total_training_steps,
                "training_completion": f"{(self.current_training_step/self.total_training_steps*100):.1f}%" if self.total_training_steps > 0 else "N/A",
                "all_components_frozen_at_step": self.all_components_frozen_at_step,
                "final_frozen_count": len(self.frozen_components),
                "total_components": len(self.component_stats),
                "frozen_percentage": f"{(len(self.frozen_components)/len(self.component_stats)*100):.1f}%" if self.component_stats else "0.0%",
                "component_breakdown": dict(component_summary),
            },
            "freezing_timeline": {
                "total_events": len(self.frozen_events),
                "events": self.frozen_events,
            },
            "component_details": {
                name: {
                    "component_type": stats.component_type,
                    "frozen": stats.frozen,
                    "frozen_at_step": stats.frozen_at_step,
                    "frozen_at_progress": f"{(stats.frozen_at_step/self.total_training_steps*100):.1f}%" if stats.frozen_at_step and self.total_training_steps > 0 else None,
                    "param_count": stats.param_count,
                    "param_count_formatted": f"{stats.param_count:,}",
                    "total_change_accumulated": round(stats.total_change_accumulated, 6),
                }
                for name, stats in self.component_stats.items()
            },
            "performance_metrics": {
                "avg_time_per_component": round(total_time / len(self.component_stats), 3) if self.component_stats else 0,
                "total_parameters": sum(stats.param_count for stats in self.component_stats.values()),
                "frozen_parameters": sum(stats.param_count for stats in self.component_stats.values() if stats.frozen),
            }
        }

        # Save to JSON
        output_file = output_dir / "dynamic_freezing_history.json"
        with open(output_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)

        # Create a human-readable summary file
        summary_file = output_dir / "freezing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Gradient-Based Early Stopping - Training Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Training Mode: {self.mode.upper()} fine-tuning\n")
            f.write(f"Total Training Time: {int(total_time//60)}m {int(total_time%60)}s\n")
            f.write(f"Training Steps: {self.current_training_step}/{self.total_training_steps}\n")
            f.write(f"Components Frozen: {len(self.frozen_components)}/{len(self.component_stats)} ({(len(self.frozen_components)/len(self.component_stats)*100):.1f}%)\n\n")

            f.write("Component Breakdown:\n")
            f.write("-" * 40 + "\n")
            for comp_type, counts in component_summary.items():
                ratio = counts["frozen"] / counts["total"] if counts["total"] > 0 else 0
                f.write(f"  {comp_type:20s}: {counts['frozen']:3d}/{counts['total']:3d} frozen ({ratio:.1%})\n")

            if self.frozen_events:
                f.write("\nFreezing Timeline (First 10 events):\n")
                f.write("-" * 40 + "\n")
                for event in self.frozen_events[:10]:
                    progress = (event['step'] / self.total_training_steps * 100) if self.total_training_steps > 0 else 0
                    f.write(f"  Step {event['step']:6d} ({progress:5.1f}%): {event['component']:30s} ({event['param_count']:,} params)\n")

                if len(self.frozen_events) > 10:
                    f.write(f"  ... and {len(self.frozen_events) - 10} more events\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Full details saved to: {output_file}\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Freezing history saved to {output_file}")
        logger.info(f"Human-readable summary saved to {summary_file}")

    def _log_to_wandb(self, current_step: int, component_changes: Dict[str, List[float]]):
        """Log aggregated component changes to wandb using efficient matrix computation."""
        try:
            import wandb

            metrics = {
                "GradES/global/frozen_components": len(self.frozen_components),
                "GradES/global/frozen_ratio": len(self.frozen_components) / len(self.component_stats) if self.component_stats else 0,
                "GradES/global/total_components": len(self.component_stats)
            }

            # Efficient aggregation: compute statistics across all component types
            if component_changes:
                # Convert to numpy array for efficient computation
                all_changes = []
                for changes in component_changes.values():
                    if changes:
                        all_changes.extend(changes)

                if all_changes:
                    changes_matrix = np.array(all_changes)

                    # Compute aggregate statistics
                    metrics["GradES/changes/mean"] = float(np.mean(changes_matrix))
                    metrics["GradES/changes/max"] = float(np.max(changes_matrix))
                    metrics["GradES/changes/min"] = float(np.min(changes_matrix))
                    metrics["GradES/changes/std"] = float(np.std(changes_matrix))

                    # Optional: Add percentiles for better distribution understanding
                    metrics["GradES/changes/median"] = float(np.median(changes_matrix))
                    metrics["GradES/changes/p25"] = float(np.percentile(changes_matrix, 25))
                    metrics["GradES/changes/p75"] = float(np.percentile(changes_matrix, 75))

            # Log all metrics at once
            if metrics:
                wandb.log(metrics, step=current_step)

        except Exception as e:
            logger.debug(f"Error logging to wandb: {e}")

    def _cleanup_memory(self):
        """Clean up memory after training."""
        # Clear weight caches
        for stats in self.component_stats.values():
            stats.weight_cache = None

        # Clear CUDA cache
        if self.cuda_available:
            torch.cuda.empty_cache()
