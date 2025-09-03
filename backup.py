"""
Dynamic Module Freezing Callback for Transformers

A TrainerCallback that dynamically freezes model parameters during training
based on weight change convergence, supporting both LoRA and full parameter fine-tuning.

Authors: [Your Name]
Paper: [Your Paper Title and Link]
License: MIT License
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import IntervalStrategy

logger = logging.getLogger(__name__)


@dataclass
class DynamicFreezingConfig:
    """
    Configuration for the DynamicFreezingCallback.
    
    Args:
        tau (`float`, *optional*, defaults to 1e-4):
            Weight change threshold below which a module is considered converged and frozen.
        alpha (`float`, *optional*, defaults to 0.3):
            Minimum fraction of total training steps before freezing is allowed.
        max_frozen_ratio (`float`, *optional*, defaults to 1.0):
            Maximum fraction of modules that can be frozen before early stopping.
        compute_interval (`int`, *optional*, defaults to 1):
            Number of steps between weight change computations.
        target_modules (`List[str]`, *optional*):
            List of module names to monitor. Defaults to common transformer modules.
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
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    auto_detect_mode: bool = True
    use_cuda_acceleration: bool = True
    save_freezing_history: bool = False
    enable_wandb_logging: bool = False
    output_dir: Optional[str] = None


@dataclass
class ModuleStats:
    """Statistics for tracking module weight changes."""
    
    name: str
    module_type: str  # 'lora' or 'full'
    frozen: bool = False
    frozen_at_step: Optional[int] = None
    change_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    weight_cache: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None
    total_change_accumulated: float = 0.0
    param_count: int = 0
    
    def add_change(self, change: float, step: int):
        """Records a weight change for a given step."""
        self.change_history.append((step, change))
        self.total_change_accumulated += change


class DynamicFreezingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that dynamically freezes model parameters during training
    based on weight change convergence.
    
    This callback monitors the weight changes of specified modules during training and
    freezes them when their changes fall below a threshold, indicating convergence.
    It supports both LoRA adapter fine-tuning and full parameter fine-tuning.
    
    Args:
        config (`DynamicFreezingConfig`, *optional*):
            Configuration for the dynamic freezing behavior. If None, uses default config.
    
    Example:
        ```python
        from transformers import Trainer
        
        # Create callback with custom configuration
        freeze_config = DynamicFreezingConfig(
            tau=1e-5,
            alpha=0.2,
            compute_interval=50
        )
        freeze_callback = DynamicFreezingCallback(config=freeze_config)
        
        # Add to trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            callbacks=[freeze_callback],
            ...
        )
        ```
    
    Note:
        This callback works best when used with gradient accumulation and mixed precision training.
        For optimal performance, ensure your TrainingArguments includes appropriate settings for
        these features.
    """
    
    def __init__(self, config: Optional[DynamicFreezingConfig] = None):
        self.config = config or DynamicFreezingConfig()
        
        # Core tracking structures
        self.module_stats: Dict[str, ModuleStats] = {}
        self.frozen_modules: Set[str] = set()
        
        # Training mode detection
        self.mode: Optional[str] = None  # 'lora' or 'full'
        self.model_layers_cache = None
        
        # Device management
        self.device = None
        self.cuda_available = False
        
        # Global statistics
        self.start_time: Optional[float] = None
        self.total_steps: int = 0
        self.max_steps: int = 0
        self.all_modules_frozen_at_step: Optional[int] = None
        self.frozen_events: List[Dict] = []
        
        # Wandb logging
        self.wandb_available: bool = False
        
        # For state export
        self.initialized = False
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                       control: TrainerControl, model=None, **kwargs):
        """Initialize the callback and detect training mode."""
        
        # Validate configuration
        if args.eval_strategy == IntervalStrategy.NO and self.config.save_freezing_history:
            logger.warning(
                "DynamicFreezingCallback with save_freezing_history=True works best with "
                "evaluation enabled. Consider setting eval_strategy to 'steps' or 'epoch'."
            )
        
        # Set output directory
        if self.config.output_dir is None:
            self.config.output_dir = args.output_dir
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                  self.config.use_cuda_acceleration else "cpu")
        self.cuda_available = torch.cuda.is_available() and self.config.use_cuda_acceleration
        
        # Initialize timing
        self.start_time = time.time()
        self.max_steps = state.max_steps
        
        # Check wandb availability
        if self.config.enable_wandb_logging:
            try:
                import wandb
                self.wandb_available = wandb.run is not None
                if self.wandb_available:
                    logger.info("Wandb integration enabled for layer-wise component tracking")
            except ImportError:
                logger.warning("Wandb not available. Install with: pip install wandb")
        
        # Detect training mode
        if self.config.auto_detect_mode:
            self.mode = self._detect_training_mode(model)
            logger.info(f"DynamicFreezingCallback detected {self.mode.upper()} parameter fine-tuning")
        else:
            self.mode = 'full'  # Default to full if not auto-detecting
            logger.info("DynamicFreezingCallback using full parameter mode (auto_detect disabled)")
        
        # Initialize module tracking
        initialized_count = self._initialize_module_tracking(model)
        
        if initialized_count == 0:
            logger.warning(
                "DynamicFreezingCallback found no modules to track. "
                "Check that your model has trainable parameters in the target modules."
            )
            self.initialized = False
            return
        
        self.initialized = True
        
        # Log initialization summary
        total_params = sum(stats.param_count for stats in self.module_stats.values())
        logger.info(
            f"DynamicFreezingCallback initialized: "
            f"tracking {initialized_count} modules with {total_params:,} parameters"
        )
        
        if self.cuda_available:
            logger.info("CUDA acceleration enabled for weight change computations")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        """Monitor modules and freeze converged ones."""
        
        if not self.initialized:
            return control
        
        current_step = state.global_step
        self.total_steps = current_step
        
        # Calculate minimum steps before freezing is allowed
        min_steps_before_freeze = int(self.max_steps * self.config.alpha)
        
        # Only compute changes at specified intervals
        if current_step > 0 and current_step % self.config.compute_interval == 0:
            
            # Get active (non-frozen) modules
            active_modules = [k for k, v in self.module_stats.items() if not v.frozen]
            
            if not active_modules:
                # All modules are frozen
                if self._should_stop_training():
                    control.should_training_stop = True
                return control
            
            # Process each active module
            modules_to_freeze = []
            
            for module_key in active_modules:
                module = self._get_module_by_key(model, module_key)
                if module is None:
                    continue
                
                try:
                    # Calculate weight change
                    stats = self.module_stats[module_key]
                    change = self._calculate_module_change(module, stats)
                    stats.add_change(change, current_step)
                    
                    # Check if module should be frozen
                    if (current_step >= min_steps_before_freeze and 
                        change < self.config.tau):
                        modules_to_freeze.append(module_key)
                
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM while processing {module_key}, clearing cache")
                    if self.cuda_available:
                        torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    logger.error(f"Error processing module {module_key}: {e}")
                    continue
            
            # Freeze converged modules
            if modules_to_freeze:
                for module_key in modules_to_freeze:
                    self._freeze_module(model, module_key, current_step)
                
                # Clear CUDA cache after freezing
                if self.cuda_available:
                    torch.cuda.empty_cache()
                
                logger.info(
                    f"Step {current_step}: Frozen {len(modules_to_freeze)} modules. "
                    f"Total frozen: {len(self.frozen_modules)}/{len(self.module_stats)}"
                )
            
            # Log to wandb if available
            if self.wandb_available:
                self._log_to_wandb(current_step)
            
            # Check if all modules are now frozen
            if self.all_modules_frozen_at_step is None and len(self.frozen_modules) == len(self.module_stats):
                self.all_modules_frozen_at_step = current_step
                logger.info(f"All modules frozen at step {current_step}")
                if self._should_stop_training():
                    control.should_training_stop = True
        
        # Check early stopping based on frozen ratio
        if self._should_stop_training():
            frozen_ratio = len(self.frozen_modules) / len(self.module_stats) if self.module_stats else 0
            logger.info(
                f"Early stopping triggered at step {current_step}: "
                f"{frozen_ratio:.1%} modules frozen (threshold: {self.config.max_frozen_ratio:.1%})"
            )
            control.should_training_stop = True
        
        return control
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, metrics: Dict[str, float], **kwargs):
        """Log freezing statistics during evaluation."""
        
        if not self.initialized:
            return
        
        # Add freezing metrics to evaluation metrics
        if self.module_stats:
            frozen_ratio = len(self.frozen_modules) / len(self.module_stats)
            metrics['frozen_modules'] = len(self.frozen_modules)
            metrics['frozen_ratio'] = frozen_ratio
            metrics['total_modules'] = len(self.module_stats)
            
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, model=None, **kwargs):
        """Save final statistics and clean up."""
        
        if not self.initialized:
            return
        
        total_time = time.time() - self.start_time
        
        logger.info(
            f"DynamicFreezingCallback: Training completed in {total_time:.2f} seconds. "
            f"Final state: {len(self.frozen_modules)}/{len(self.module_stats)} modules frozen"
        )
        
        # Save statistics if requested
        if self.config.save_freezing_history:
            self._save_freezing_history()
        
        # Clean up memory
        self._cleanup_memory()
    
    def state(self) -> dict:
        """Export callback state for checkpointing."""
        return {
            "config": {
                "tau": self.config.tau,
                "alpha": self.config.alpha,
                "max_frozen_ratio": self.config.max_frozen_ratio,
                "compute_interval": self.config.compute_interval,
                "target_modules": self.config.target_modules,
                "auto_detect_mode": self.config.auto_detect_mode,
            },
            "attributes": {
                "mode": self.mode,
                "frozen_modules": list(self.frozen_modules),
                "all_modules_frozen_at_step": self.all_modules_frozen_at_step,
                "frozen_events": self.frozen_events,
                "total_steps": self.total_steps,
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
        for layer in layers[:3]:
            if hasattr(layer, 'self_attn'):
                for module_name in self.config.target_modules[:4]:
                    if hasattr(layer.self_attn, module_name):
                        module = getattr(layer.self_attn, module_name)
                        if self._has_lora(module):
                            return 'lora'
        return 'full'
    
    def _has_lora(self, module) -> bool:
        """Check if a module has LoRA adapters."""
        return (hasattr(module, 'lora_A') and 
                hasattr(module, 'lora_B') and 
                len(getattr(module, 'lora_A', {})) > 0)
    
    def _initialize_module_tracking(self, model) -> int:
        """Initialize tracking for all target modules."""
        layers = self._get_model_layers(model)
        if not layers:
            return 0
        
        initialized_count = 0
        
        for i, layer in enumerate(layers):
            # Check attention modules
            if hasattr(layer, 'self_attn'):
                for module_name in self.config.target_modules[:4]:
                    if hasattr(layer.self_attn, module_name):
                        module = getattr(layer.self_attn, module_name)
                        
                        # Check if module should be tracked
                        should_track = False
                        param_count = 0
                        
                        if self.mode == 'lora':
                            should_track = self._has_lora(module)
                            if should_track:
                                lora_a, lora_b = self._get_lora_matrices(module)
                                if lora_a is not None and lora_b is not None:
                                    param_count = lora_a.numel() + lora_b.numel()
                        else:
                            should_track = any(p.requires_grad for p in module.parameters())
                            if should_track:
                                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                        
                        if should_track:
                            key = f"layer_{i:02d}_attn_{module_name}"
                            self.module_stats[key] = ModuleStats(
                                name=key,
                                module_type=self.mode,
                                param_count=param_count
                            )
                            initialized_count += 1
            
            # Check MLP modules
            if hasattr(layer, 'mlp'):
                for module_name in self.config.target_modules[4:]:
                    if hasattr(layer.mlp, module_name):
                        module = getattr(layer.mlp, module_name)
                        
                        should_track = False
                        param_count = 0
                        
                        if self.mode == 'lora':
                            should_track = self._has_lora(module)
                            if should_track:
                                lora_a, lora_b = self._get_lora_matrices(module)
                                if lora_a is not None and lora_b is not None:
                                    param_count = lora_a.numel() + lora_b.numel()
                        else:
                            should_track = any(p.requires_grad for p in module.parameters())
                            if should_track:
                                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                        
                        if should_track:
                            key = f"layer_{i:02d}_mlp_{module_name}"
                            self.module_stats[key] = ModuleStats(
                                name=key,
                                module_type=self.mode,
                                param_count=param_count
                            )
                            initialized_count += 1
        
        return initialized_count
    
    def _get_lora_matrices(self, module) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get LoRA A and B matrices from a module."""
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if 'default' in module.lora_A and 'default' in module.lora_B:
                a = module.lora_A['default']
                b = module.lora_B['default']
                if a is not None and b is not None:
                    return a.weight, b.weight
        return None, None
    
    def _calculate_module_change(self, module, stats: ModuleStats) -> float:
        """Calculate weight change for a module."""
        if self.mode == 'lora':
            return self._calculate_lora_change(module, stats)
        else:
            return self._calculate_full_param_change(module, stats)
    
    def _calculate_lora_change(self, module, stats: ModuleStats) -> float:
        """Calculate change for LoRA modules."""
        lora_a, lora_b = self._get_lora_matrices(module)
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
    
    def _calculate_full_param_change(self, module, stats: ModuleStats) -> float:
        """Calculate change for full parameter modules."""
        try:
            with torch.no_grad():
                # Get current weights
                current_weights = []
                device = None
                
                for param in module.parameters():
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
    
    def _freeze_module(self, model, module_key: str, step: int):
        """Freeze a module's parameters."""
        if module_key in self.frozen_modules:
            return
        
        module = self._get_module_by_key(model, module_key)
        if module is None:
            return
        
        frozen_count = 0
        
        if self.mode == 'lora':
            # Freeze LoRA parameters
            for param_name in ['lora_A', 'lora_B']:
                if hasattr(module, param_name):
                    for adapter_name in getattr(module, param_name):
                        param = getattr(module, param_name)[adapter_name].weight
                        if param.requires_grad:
                            param.requires_grad = False
                            frozen_count += 1
        else:
            # Freeze all parameters
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += 1
        
        # Clear weight cache to save memory
        if module_key in self.module_stats:
            self.module_stats[module_key].weight_cache = None
        
        # Update statistics
        stats = self.module_stats[module_key]
        stats.frozen = True
        stats.frozen_at_step = step
        self.frozen_modules.add(module_key)
        
        # Record event
        self.frozen_events.append({
            "step": step,
            "module": module_key,
            "mode": self.mode,
            "param_count": stats.param_count
        })
    
    def _should_stop_training(self) -> bool:
        """Check if training should stop based on frozen ratio."""
        if not self.module_stats:
            return False
        frozen_ratio = len(self.frozen_modules) / len(self.module_stats)
        return frozen_ratio >= self.config.max_frozen_ratio
    
    def _get_model_layers(self, model):
        """Get model layers with caching."""
        if self.model_layers_cache is None:
            # Try common model structures
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
    
    def _get_module_by_key(self, model, module_key: str):
        """Get a module by its tracking key."""
        try:
            parts = module_key.split('_')
            layer_idx = int(parts[1])
            module_type = parts[2]  # 'attn' or 'mlp'
            module_name = '_'.join(parts[3:])
            
            layers = self._get_model_layers(model)
            if layers and layer_idx < len(layers):
                layer = layers[layer_idx]
                parent = getattr(layer, 'self_attn' if module_type == 'attn' else 'mlp', None)
                if parent:
                    return getattr(parent, module_name, None)
        except Exception as e:
            logger.debug(f"Error getting module {module_key}: {e}")
        return None
    
    def _save_freezing_history(self):
        """Save detailed freezing history to file."""
        if not self.config.output_dir:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare statistics
        stats_to_save = {
            "mode": self.mode,
            "config": {
                "tau": self.config.tau,
                "alpha": self.config.alpha,
                "max_frozen_ratio": self.config.max_frozen_ratio,
                "compute_interval": self.config.compute_interval,
                "target_modules": self.config.target_modules
            },
            "summary": {
                "total_time_seconds": time.time() - self.start_time if self.start_time else 0,
                "total_steps": self.total_steps,
                "max_steps": self.max_steps,
                "all_modules_frozen_at_step": self.all_modules_frozen_at_step,
                "final_frozen_count": len(self.frozen_modules),
                "total_modules": len(self.module_stats)
            },
            "frozen_events": self.frozen_events,
            "module_details": {
                name: {
                    "module_type": stats.module_type,
                    "frozen": stats.frozen,
                    "frozen_at_step": stats.frozen_at_step,
                    "param_count": stats.param_count,
                    "total_change": stats.total_change_accumulated,
                }
                for name, stats in self.module_stats.items()
            }
        }
        
        # Save to JSON
        output_file = output_dir / "dynamic_freezing_history.json"
        with open(output_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        logger.info(f"Freezing history saved to {output_file}")
    
    def _log_to_wandb(self, current_step: int):
        """Log component changes to wandb."""
        try:
            import wandb
            
            metrics = {}
            
            # Group by component type across all layers
            component_stats = {}
            
            for module_key, stats in self.module_stats.items():
                if not stats.change_history:
                    continue
                    
                # Extract component type (e.g., "attn_q_proj", "mlp_gate_proj")
                parts = module_key.split('_')
                component_type = f"{parts[2]}_{parts[3]}"
                
                # Get the latest change value
                if stats.change_history:
                    latest_change = stats.change_history[-1][1]  # (step, change)
                    
                    if component_type not in component_stats:
                        component_stats[component_type] = []
                    component_stats[component_type].append(latest_change)
            
            # Log average change per component type (each gets its own line)
            for component_type, changes in component_stats.items():
                avg_change = sum(changes) / len(changes)
                metrics[f"GradES/components/{component_type}"] = avg_change
            
            # Add global metrics
            frozen_ratio = len(self.frozen_modules) / len(self.module_stats) if self.module_stats else 0
            metrics.update({
                "GradES/global/frozen_modules": len(self.frozen_modules),
                "GradES/global/frozen_ratio": frozen_ratio,
                "GradES/global/total_modules": len(self.module_stats)
            })
            
            # Log all metrics at once
            if metrics:
                wandb.log(metrics, step=current_step)
            
        except Exception as e:
            logger.debug(f"Error logging to wandb: {e}")
    
    def _cleanup_memory(self):
        """Clean up memory after training."""
        # Clear weight caches
        for stats in self.module_stats.values():
            stats.weight_cache = None
        
        # Clear CUDA cache
        if self.cuda_available:
            torch.cuda.empty_cache()