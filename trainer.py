import os
import torch
from transformers import Trainer, TrainingArguments
from peft import PeftModel
from config import TrainingConfig

class CustomTrainer:
    def __init__(self, config: TrainingConfig, model, tokenizer, train_dataset, eval_dataset):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
    def setup_training_args(self):
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            max_grad_norm=self.config.max_grad_norm,
            lr_scheduler_type=self.config.lr_scheduler_type,
            save_total_limit=self.config.save_total_limit,
            remove_unused_columns=self.config.remove_unused_columns,
            push_to_hub=self.config.push_to_hub,
            hub_model_id=self.config.hub_model_id,
            hub_token=self.config.hub_token,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_num_workers=self.config.dataloader_num_workers,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            report_to="wandb" if self.config.push_to_hub else "none",
            run_name="deepseek-finetune" if self.config.push_to_hub else None
        )
        return training_args
    
    def train(self):
        training_args = self.setup_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        
        if self.config.use_peft:
            self.save_peft_model()
        else:
            trainer.save_model()
        
        return trainer
    
    def save_peft_model(self):
        output_dir = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    def load_finetuned_model(self, model_path: str):
        if self.config.use_peft:
            model = PeftModel.from_pretrained(self.model, model_path)
        else:
            model = self.model.__class__.from_pretrained(model_path)
        
        return model 