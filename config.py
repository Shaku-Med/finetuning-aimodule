from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-instruct"
    output_dir: str = "./finetuned_model"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 8e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5
    logging_steps: int = 1
    save_steps: int = 20
    eval_steps: int = 20
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    save_total_limit: int = 3
    remove_unused_columns: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 0
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    bf16: bool = False
    use_peft: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]