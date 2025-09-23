"""
Gradient-based Early Stopping Callback for Vision-Language Models (VLMs)

A TrainerCallback implementing gradient-based early stopping specifically designed for 
Vision-Language Models that monitors gradient magnitudes during backpropagation and freezes 
individual components (vision encoder and language model matrices) when their gradients fall 
below convergence thresholds, supporting both LoRA and full parameter fine-tuning modes.

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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

__version__ = "1.0.0"
__all__ = ["VLMGradEarlyStoppingCallback", "VLMGradEarlyStoppingConfig", "VLMComponentStats"]

logger = logging.getLogger(__name__)

# Define constants
DEFAULT_HISTORY_MAXLEN = 100  # Maximum change history entries per component
VISION_COMPONENTS = ["qkv", "q_proj", "k_proj", "v_proj", "o_proj", "proj", "fc1", "fc2"]
LANGUAGE_ATTENTION_COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LANGUAGE_MLP_COMPONENTS = ["gate_proj", "up_proj", "down_proj"]

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class VLMGradEarlyStoppingConfig:
    """
    Configuration for the VLMGradEarlyStoppingCallback.
    
    Args:
        vision_tau (`float`, *optional*, defaults to 1e-5):
            Weight change threshold below which vision encoder components are considered converged.
        language_tau (`float`, *optional*, defaults to 1e-4):
            Weight change threshold below which language model components are considered converged.
        alpha (`float`, *optional*, defaults to 0.3):
            Minimum fraction of total training steps before freezing is allowed.
        max_frozen_ratio (`float`, *optional*, defaults to 1.0):
            Maximum fraction of components that can be frozen before early stopping.
        compute_interval (`int`, *optional*, defaults to 1):
            Number of steps between weight change computations.
        freeze_vision (`bool`, *optional*, defaults to True):
            Whether to enable freezing for vision encoder components.
        freeze_language (`bool`, *optional*, defaults to True):
            Whether to enable freezing for language model components.
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
    
    vision_tau: float = 1e-5
    language_tau: float = 1e-4
    alpha: float = 0.3
    max_frozen_ratio: float = 1.0
    compute_interval: int = 1
    freeze_vision: bool = True
    freeze_language: bool = True
    auto_detect_mode: bool = True
    use_cuda_acceleration: bool = True
    save_freezing_history: bool = False
    enable_wandb_logging: bool = False
    output_dir: Optional[str] = None


@dataclass
class VLMComponentStats:
    """Statistics for tracking VLM component weight changes."""
    
    name: str
    component_type: str  # 'vision' or 'language'
    module_type: str  # 'lora' or 'full'
    layer_index: int
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


class VLMGradEarlyStoppingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that dynamically freezes Vision-Language Model parameters during 
    training based on weight change convergence.
    
    This callback monitors the weight changes of vision encoder and language model components 
    during training and freezes them when their changes fall below component-specific thresholds, 
    indicating convergence. It supports both LoRA adapter fine-tuning and full parameter 
    fine-tuning for VLMs.
    
    Args:
        config (`VLMGradEarlyStoppingConfig`, *optional*):
            Configuration for the dynamic freezing behavior. If None, uses default config.
    
    Example:
        ```python
        from transformers import Trainer
        
        # Create callback with custom configuration
        freeze_config = VLMGradEarlyStoppingConfig(
            vision_tau=1e-5,
            language_tau=1e-4,
            alpha=0.3,
            compute_interval=50
        )
        freeze_callback = VLMGradEarlyStoppingCallback(config=freeze_config)
        
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
        This callback is optimized for Vision-Language Models including Qwen2.5-VL, LLaVA, 
        and similar architectures. It provides separate convergence thresholds for vision 
        and language components to account for their different learning dynamics.
    """
    
    DEFAULT_HISTORY_MAXLEN = DEFAULT_HISTORY_MAXLEN
    VISION_COMPONENTS = VISION_COMPONENTS
    LANGUAGE_ATTENTION_COMPONENTS = LANGUAGE_ATTENTION_COMPONENTS
    LANGUAGE_MLP_COMPONENTS = LANGUAGE_MLP_COMPONENTS
    
    def __init__(self, config: Optional[VLMGradEarlyStoppingConfig] = None):
        self.config = config or VLMGradEarlyStoppingConfig()
        
        # Validate configuration
        if not 0 < self.config.vision_tau:
            raise ValueError(f"vision_tau must be greater than 0, got {self.config.vision_tau}")
        if not 0 < self.config.language_tau:
            raise ValueError(f"language_tau must be greater than 0, got {self.config.language_tau}")
        if not 0 <= self.config.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.config.alpha}")
        if not 0 < self.config.max_frozen_ratio <= 1:
            raise ValueError(f"max_frozen_ratio must be between 0 and 1, got {self.config.max_frozen_ratio}")
        if self.config.compute_interval < 1:
            raise ValueError(f"compute_interval must be >= 1, got {self.config.compute_interval}")
        
        # Core tracking structures
        self.component_stats: Dict[str, VLMComponentStats] = {}
        self.frozen_components: Set[str] = set()
        
        # Model component references
        self.vision_encoder = None
        self.language_model = None
        self.model_layers_cache = None
        
        # Training mode detection
        self.mode: Optional[str] = None  # 'lora' or 'full'
        
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
            self.wandb_available = WANDB_AVAILABLE and wandb.run is not None
            if self.config.enable_wandb_logging and not WANDB_AVAILABLE:
                logger.warning("Wandb requested but not installed. Install with: pip install wandb")
            elif self.wandb_available:
                logger.info("Wandb integration enabled for VLM component tracking")
                self._initialize_wandb()
        
        # Detect model components
        self.vision_encoder, self.language_model = self._detect_model_components(model)
        
        # Detect training mode
        if self.config.auto_detect_mode:
            self.mode = self._detect_training_mode(model)
            logger.info(f"VLMGradEarlyStoppingCallback detected {self.mode.upper()} parameter fine-tuning")
        else:
            self.mode = 'full'  # Default to full if not auto-detecting
            logger.info("VLMGradEarlyStoppingCallback using full parameter mode (auto_detect disabled)")
        
        # Initialize component tracking
        vision_count, language_count = self._initialize_component_tracking(model)
        
        if vision_count == 0 and language_count == 0:
            logger.warning(
                "VLMGradEarlyStoppingCallback found no components to track. "
                "Check that your model has trainable parameters in the target components."
            )
            self.initialized = False
            return
        
        self.initialized = True
        
        # Log initialization summary
        total_params = sum(stats.param_count for stats in self.component_stats.values())
        logger.info(
            f"VLMGradEarlyStoppingCallback initialized: "
            f"tracking {vision_count} vision and {language_count} language components "
            f"with {total_params:,} parameters"
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
                    
                    # Track for wandb
                    component_changes[stats.component_type].append(change)
                    
                    # Check if component should be frozen
                    if current_step >= min_steps_before_freeze:
                        threshold = (self.config.vision_tau if stats.component_type == 'vision'
                                   else self.config.language_tau)
                        
                        if change < threshold:
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
                
                # Log freezing event
                vision_frozen = sum(1 for k in components_to_freeze 
                                  if self.component_stats[k].component_type == 'vision')
                language_frozen = len(components_to_freeze) - vision_frozen
                
                logger.info(
                    f"Step {current_step}: Frozen {len(components_to_freeze)} components "
                    f"(vision: {vision_frozen}, language: {language_frozen}). "
                    f"Total frozen: {len(self.frozen_components)}/{len(self.component_stats)}"
                )
            
            # Log to wandb if available
            if self.wandb_available:
                self._log_to_wandb(current_step, component_changes, len(components_to_freeze))
            
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
        
        # Component summary
        component_summary = self._get_component_summary()
        
        logger.info(
            f"VLMGradEarlyStoppingCallback completed in {total_time:.1f}s: "
            f"{len(self.frozen_components)}/{len(self.component_stats)} components frozen"
        )
        
        for comp_type, counts in component_summary.items():
            ratio = counts["frozen"] / counts["total"] if counts["total"] > 0 else 0
            logger.info(f"  {comp_type}: {counts['frozen']}/{counts['total']} frozen ({ratio:.1%})")
        
        # Save statistics if requested
        if self.config.save_freezing_history:
            self._save_freezing_history(total_time, component_summary)
        
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
                "vision_tau": self.config.vision_tau,
                "language_tau": self.config.language_tau,
                "alpha": self.config.alpha,
                "max_frozen_ratio": self.config.max_frozen_ratio,
                "compute_interval": self.config.compute_interval,
                "freeze_vision": self.config.freeze_vision,
                "freeze_language": self.config.freeze_language,
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
    
    def _detect_model_components(self, model) -> Tuple[Optional[Any], Optional[Any]]:
        """Detect vision encoder and language model components in VLM."""
        vision_encoder = None
        language_model = None
        
        # Handle different model wrappers
        actual_model = model
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            actual_model = model.base_model.model
        elif hasattr(model, 'model'):
            actual_model = model.model
        
        # Find vision encoder
        for attr in ['visual', 'vision_model', 'vision_encoder', 'vision_tower']:
            if hasattr(actual_model, attr):
                vision_encoder = getattr(actual_model, attr)
                break
        
        # Find language model
        for attr in ['language_model', 'text_model', 'transformer', 'model']:
            if hasattr(actual_model, attr):
                candidate = getattr(actual_model, attr)
                # Check if it has transformer layers
                if hasattr(candidate, 'layers') or hasattr(candidate, 'h'):
                    language_model = candidate
                    break
        
        return vision_encoder, language_model
    
    def _detect_training_mode(self, model) -> str:
        """Detect whether using LoRA or full parameter fine-tuning."""
        # Check if any module has LoRA adapters
        for module in model.modules():
            if self._has_lora(module):
                return 'lora'
        return 'full'
    
    def _has_lora(self, component) -> bool:
        """Check if a component has LoRA adapters."""
        return (hasattr(component, 'lora_A') and 
                hasattr(component, 'lora_B') and 
                len(getattr(component, 'lora_A', {})) > 0)
    
    def _initialize_component_tracking(self, model) -> Tuple[int, int]:
        """Initialize tracking for all VLM components."""
        vision_count = 0
        language_count = 0
        
        # Track vision encoder components
        if self.vision_encoder and self.config.freeze_vision:
            vision_count = self._track_vision_components(self.vision_encoder)
        
        # Track language model components
        if self.language_model and self.config.freeze_language:
            language_count = self._track_language_components(self.language_model)
        
        return vision_count, language_count
    
    def _track_vision_components(self, vision_encoder) -> int:
        """Track vision encoder components."""
        count = 0
        
        # Find transformer blocks in vision encoder
        blocks = self._get_vision_blocks(vision_encoder)
        if not blocks:
            return 0
        
        for i, block in enumerate(blocks):
            # Track attention components
            for attn_name in ['attn', 'self_attn', 'attention']:
                if hasattr(block, attn_name):
                    attn = getattr(block, attn_name)
                    for proj in self.VISION_COMPONENTS:
                        if hasattr(attn, proj):
                            module = getattr(attn, proj)
                            if self._should_track_module(module):
                                key = f"vision_L{i:02d}_attn_{proj}"
                                self.component_stats[key] = VLMComponentStats(
                                    name=key,
                                    component_type='vision',
                                    module_type=self.mode,
                                    layer_index=i,
                                    param_count=self._count_module_params(module)
                                )
                                count += 1
            
            # Track MLP components
            for mlp_name in ['mlp', 'ffn', 'feed_forward']:
                if hasattr(block, mlp_name):
                    mlp = getattr(block, mlp_name)
                    for proj in ['fc1', 'fc2', 'gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(mlp, proj):
                            module = getattr(mlp, proj)
                            if self._should_track_module(module):
                                key = f"vision_L{i:02d}_mlp_{proj}"
                                self.component_stats[key] = VLMComponentStats(
                                    name=key,
                                    component_type='vision',
                                    module_type=self.mode,
                                    layer_index=i,
                                    param_count=self._count_module_params(module)
                                )
                                count += 1
        
        return count
    
    def _track_language_components(self, language_model) -> int:
        """Track language model components."""
        count = 0
        
        # Find transformer layers
        layers = self._get_language_layers(language_model)
        if not layers:
            return 0
        
        for i, layer in enumerate(layers):
            # Track attention components
            if hasattr(layer, 'self_attn'):
                for proj in self.LANGUAGE_ATTENTION_COMPONENTS:
                    if hasattr(layer.self_attn, proj):
                        module = getattr(layer.self_attn, proj)
                        if self._should_track_module(module):
                            key = f"language_L{i:02d}_attn_{proj}"
                            self.component_stats[key] = VLMComponentStats(
                                name=key,
                                component_type='language',
                                module_type=self.mode,
                                layer_index=i,
                                param_count=self._count_module_params(module)
                            )
                            count += 1
            
            # Track MLP components
            if hasattr(layer, 'mlp'):
                for proj in self.LANGUAGE_MLP_COMPONENTS:
                    if hasattr(layer.mlp, proj):
                        module = getattr(layer.mlp, proj)
                        if self._should_track_module(module):
                            key = f"language_L{i:02d}_mlp_{proj}"
                            self.component_stats[key] = VLMComponentStats(
                                name=key,
                                component_type='language',
                                module_type=self.mode,
                                layer_index=i,
                                param_count=self._count_module_params(module)
                            )
                            count += 1
        
        return count
    
    def _get_vision_blocks(self, vision_encoder) -> Optional[List]:
        """Get vision encoder transformer blocks."""
        # Common attribute paths for vision encoder blocks
        for attr in ['blocks', 'layers']:
            if hasattr(vision_encoder, attr):
                return getattr(vision_encoder, attr)
        
        # Check for nested encoder structure
        if hasattr(vision_encoder, 'encoder'):
            encoder = vision_encoder.encoder
            for attr in ['layers', 'blocks']:
                if hasattr(encoder, attr):
                    return getattr(encoder, attr)
        
        return None
    
    def _get_language_layers(self, language_model) -> Optional[List]:
        """Get language model transformer layers."""
        # Common attribute paths for language model layers
        for attr in ['layers', 'h']:
            if hasattr(language_model, attr):
                return getattr(language_model, attr)
        return None
    
    def _should_track_module(self, module) -> bool:
        """Check if a module should be tracked."""
        if self.mode == 'lora':
            return self._has_lora(module)
        return any(p.requires_grad for p in module.parameters())
    
    def _count_module_params(self, module) -> int:
        """Count trainable parameters in a module."""
        if self.mode == 'lora':
            lora_a, lora_b = self._extract_lora_weight_matrices(module)
            if lora_a is not None and lora_b is not None:
                return lora_a.numel() + lora_b.numel()
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    def _extract_lora_weight_matrices(self, component) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get LoRA A and B matrices from a component."""
        if hasattr(component, 'lora_A') and hasattr(component, 'lora_B'):
            if 'default' in component.lora_A and 'default' in component.lora_B:
                a = component.lora_A['default']
                b = component.lora_B['default']
                # Extract weight tensors
                a_weight = a.weight if hasattr(a, 'weight') else a
                b_weight = b.weight if hasattr(b, 'weight') else b
                if a_weight is not None and b_weight is not None:
                    return a_weight, b_weight
        return None, None
    
    def _calculate_component_change(self, component, stats: VLMComponentStats) -> float:
        """Calculate weight change for a component."""
        if self.mode == 'lora':
            return self._calculate_lora_change(component, stats)
        else:
            return self._calculate_full_param_change(component, stats)
    
    def _calculate_lora_change(self, component, stats: VLMComponentStats) -> float:
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
    
    def _calculate_full_param_change(self, component, stats: VLMComponentStats) -> float:
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
                        lora_module = getattr(component, param_name)[adapter_name]
                        if hasattr(lora_module, 'weight'):
                            if lora_module.weight.requires_grad:
                                lora_module.weight.requires_grad = False
                                frozen_count += 1
                        elif hasattr(lora_module, 'requires_grad'):
                            lora_module.requires_grad = False
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
            "component_type": stats.component_type,
            "mode": self.mode,
            "param_count": stats.param_count
        })
    
    def _should_stop_training(self) -> bool:
        """Check if training should stop based on frozen ratio."""
        if not self.component_stats:
            return False
        frozen_ratio = len(self.frozen_components) / len(self.component_stats)
        return frozen_ratio >= self.config.max_frozen_ratio
    
    def _get_component_by_key(self, model, component_key: str) -> Optional[torch.nn.Module]:
        """Get a component by its tracking key."""
        try:
            parts = component_key.split('_')
            component_type = parts[0]  # 'vision' or 'language'
            layer_idx = int(parts[1][1:])  # Remove 'L' prefix
            module_type = parts[2]  # 'attn' or 'mlp'
            proj_name = '_'.join(parts[3:])
            
            # Select component
            if component_type == 'vision':
                component = self.vision_encoder
                layers = self._get_vision_blocks(component) if component else None
            else:  # language
                component = self.language_model
                layers = self._get_language_layers(component) if component else None
            
            if not layers or layer_idx >= len(layers):
                return None
            
            layer = layers[layer_idx]
            
            # Get module parent
            if module_type == 'attn':
                parent = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
            else:  # mlp
                parent = getattr(layer, 'mlp', None) or getattr(layer, 'ffn', None)
            
            if parent and hasattr(parent, proj_name):
                return getattr(parent, proj_name)
                
        except Exception as e:
            logger.debug(f"Error getting component {component_key}: {e}")
        
        return None
    
    def _get_component_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary statistics by component type."""
        summary = defaultdict(lambda: {"total": 0, "frozen": 0})
        
        for stats in self.component_stats.values():
            summary[stats.component_type]["total"] += 1
            if stats.frozen:
                summary[stats.component_type]["frozen"] += 1
        
        return dict(summary)
    
    def _initialize_wandb(self):
        """Initialize wandb if not already initialized."""
        if not self.wandb_available:
            return
        
        try:
            if wandb.run is None:
                logger.info(
                    "Wandb run not initialized. Initialize wandb with your project/entity "
                    "settings before creating the trainer to enable VLM component tracking."
                )
                self.wandb_available = False
            else:
                # Update config with VLM-specific parameters
                wandb.config.update({
                    "vlm_mode": self.mode,
                    "vlm_vision_tau": self.config.vision_tau,
                    "vlm_language_tau": self.config.language_tau,
                    "vlm_alpha": self.config.alpha,
                    "vlm_max_frozen_ratio": self.config.max_frozen_ratio,
                    "vlm_compute_interval": self.config.compute_interval,
                })
        except Exception as e:
            logger.warning(f"Failed to update wandb config: {e}")
            self.wandb_available = False
    
    def _log_to_wandb(self, current_step: int, component_changes: Dict[str, List[float]], 
                      frozen_count: int):
        """Log metrics to wandb."""
        if not self.wandb_available:
            return
        
        try:
            # Calculate component-specific statistics
            vision_stats = self._get_component_type_stats('vision')
            language_stats = self._get_component_type_stats('language')
            
            metrics = {
                "VLMGradES/step": current_step,
                "VLMGradES/global/total_frozen": len(self.frozen_components),
                "VLMGradES/global/frozen_ratio": len(self.frozen_components) / len(self.component_stats),
                "VLMGradES/global/frozen_this_step": frozen_count,
                
                # Vision metrics
                "VLMGradES/vision/frozen": vision_stats["frozen"],
                "VLMGradES/vision/total": vision_stats["total"],
                "VLMGradES/vision/frozen_ratio": vision_stats["frozen_ratio"],
                
                # Language metrics
                "VLMGradES/language/frozen": language_stats["frozen"],
                "VLMGradES/language/total": language_stats["total"],
                "VLMGradES/language/frozen_ratio": language_stats["frozen_ratio"],
            }
            
            # Add component change metrics
            for comp_type, changes in component_changes.items():
                if changes:
                    metrics[f"VLMGradES/{comp_type}/avg_change"] = np.mean(changes)
                    metrics[f"VLMGradES/{comp_type}/max_change"] = np.max(changes)
                    metrics[f"VLMGradES/{comp_type}/min_change"] = np.min(changes)
            
            wandb.log(metrics, step=current_step)
            
        except Exception as e:
            logger.debug(f"Error logging to wandb: {e}")
    
    def _get_component_type_stats(self, component_type: str) -> Dict[str, Any]:
        """Get statistics for a specific component type."""
        total = sum(1 for stats in self.component_stats.values() 
                   if stats.component_type == component_type)
        frozen = sum(1 for stats in self.component_stats.values() 
                    if stats.component_type == component_type and stats.frozen)
        
        return {
            "total": total,
            "frozen": frozen,
            "frozen_ratio": frozen / total if total > 0 else 0
        }
    
    def _save_freezing_history(self, total_time: float, component_summary: Dict):
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
        
        # Prepare statistics
        stats_to_save = {
            "metadata": {
                "version": __version__,
                "mode": self.mode,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "config": {
                "vision_tau": self.config.vision_tau,
                "language_tau": self.config.language_tau,
                "alpha": self.config.alpha,
                "max_frozen_ratio": self.config.max_frozen_ratio,
                "compute_interval": self.config.compute_interval,
                "freeze_vision": self.config.freeze_vision,
                "freeze_language": self.config.freeze_language,
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
                "component_breakdown": component_summary,
            },
            "freezing_timeline": {
                "total_events": len(self.frozen_events),
                "events": self.frozen_events,
            },
            "component_details": {
                name: {
                    "component_type": stats.component_type,
                    "module_type": stats.module_type,
                    "layer_index": stats.layer_index,
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
        output_file = output_dir / "vlm_dynamic_freezing_history.json"
        with open(output_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        # Create a human-readable summary file
        summary_file = output_dir / "vlm_freezing_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VLM Gradient-Based Early Stopping - Training Summary\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Training Mode: {self.mode.upper()} fine-tuning\n")
            f.write(f"Total Training Time: {int(total_time//60)}m {int(total_time%60)}s\n")
            f.write(f"Training Steps: {self.current_training_step}/{self.total_training_steps}\n")
            f.write(f"Components Frozen: {len(self.frozen_components)}/{len(self.component_stats)} ({(len(self.frozen_components)/len(self.component_stats)*100):.1f}%)\n\n")
            
            f.write("Component Breakdown:\n")
            f.write("-" * 40 + "\n")
            for comp_type, counts in component_summary.items():
                ratio = counts["frozen"] / counts["total"] if counts["total"] > 0 else 0
                f.write(f"  {comp_type.capitalize():10s}: {counts['frozen']:3d}/{counts['total']:3d} frozen ({ratio:.1%})\n")
            
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
        
        logger.info(f"VLM freezing history saved to {output_file}")
        logger.info(f"Human-readable summary saved to {summary_file}")
    
    def _cleanup_memory(self):
        """Clean up memory after training."""
        # Clear weight caches
        for stats in self.component_stats.values():
            stats.weight_cache = None
        
        # Clear CUDA cache
        if self.cuda_available:
            torch.cuda.empty_cache()


# Utility function for easy configuration
def create_vlm_freeze_config(
    vision_tau: float = 1e-5,
    language_tau: float = 1e-4,
    alpha: float = 0.3,
    enable_wandb: bool = False,
) -> VLMGradEarlyStoppingConfig:
    """
    Create a VLM freezing configuration with sensible defaults.
    
    Args:
        vision_tau: Weight change threshold for vision encoder
        language_tau: Weight change threshold for language model
        alpha: Minimum training progress before freezing (0-1)
        enable_wandb: Whether to enable wandb logging
    
    Returns:
        VLMGradEarlyStoppingConfig: Freezing configuration
    
    Example:
        >>> config = create_vlm_freeze_config(vision_tau=1e-5)
        >>> callback = VLMGradEarlyStoppingCallback(config)
        >>> trainer = Trainer(..., callbacks=[callback])
    """
    return VLMGradEarlyStoppingConfig(
        vision_tau=vision_tau,
        language_tau=language_tau,
        alpha=alpha,
        enable_wandb_logging=enable_wandb,
    )
