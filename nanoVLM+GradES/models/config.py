from dataclasses import dataclass, field
from .gradient_early_stopping import GradEarlyStoppingConfig

@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip2-base-patch16-512'

    lm_hidden_dim: int = 960
    lm_inter_dim: int = 2560
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 17
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount
    lm_n_heads: int = 15
    lm_n_kv_heads: int = 5
    lm_dropout: float = 0.0
    lm_n_blocks: int = 32
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 1024
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_tokenizer: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 1024

    vlm_extra_tokens: dict[str, str] = field(default_factory=lambda: {"image_token": "<|image|>",
      "r1c1": "<row_1_col_1>", "r1c2": "<row_1_col_2>", "r1c3": "<row_1_col_3>", "r1c4": "<row_1_col_4>",
      "r2c1": "<row_2_col_1>", "r2c2": "<row_2_col_2>", "r2c3": "<row_2_col_3>", "r2c4": "<row_2_col_4>",
      "r3c1": "<row_3_col_1>", "r3c2": "<row_3_col_2>", "r3c3": "<row_3_col_3>", "r3c4": "<row_3_col_4>",
      "r4c1": "<row_4_col_1>", "r4c2": "<row_4_col_2>", "r4c3": "<row_4_col_3>", "r4c4": "<row_4_col_4>"})
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'checkpoints'
    hf_repo_name: str = 'nanoVLM'


@dataclass
class TrainConfig:
    lr_mp: float = 0.00512
    lr_backbones: float = 5e-5
    data_cutoff_idx: int = None
    val_ratio: float = 0.025
    batch_size: int = 12
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = gradient_accumulation_steps * 100
    stats_log_interval: int = gradient_accumulation_steps * 25
    max_training_steps: int = 8000
    max_images_per_example: int = 24
    max_images_per_knapsack: int = 18
    max_sample_length: int = 1024
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    train_dataset_name: tuple[str, ...] = ("all", )
    wandb_entity: str = "your-wandb-entity"
    log_wandb: bool = True
    use_lmms_eval: bool = False
    lmms_eval_tasks: str = 'mmstar,mmmu,ocrbench,textvqa'
    lmms_eval_limit: int = 2000
    lmms_eval_batch_size: int = 128

# Add a dedicated configuration for GradES to keep things clean.
@dataclass
class GradESConfig(GradEarlyStoppingConfig):
    # Inherits from the main GradES config and sets nanoVLM specific defaults.
    vision_tau: float = 0.30
    language_tau: float = 6.0
    alpha: float = 0.28
    max_frozen_ratio: float = 1.0
    compute_interval: int = 1
