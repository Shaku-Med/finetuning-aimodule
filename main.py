import os
import torch
from config import TrainingConfig
from model_setup import ModelSetup
from data_processor import DataProcessor
from trainer import CustomTrainer

def main():
    config = TrainingConfig()
    
    print("Setting up model and tokenizer...")
    model_setup = ModelSetup(config)
    model, tokenizer = model_setup.setup_model_and_tokenizer()
    
    print("Setting up data processor...")
    data_processor = DataProcessor(tokenizer)
    
    print("Preparing mixed dataset (personal + general programming knowledge)...")
    
    # Use the new mixed dataset approach
    train_dataset, eval_dataset = data_processor.prepare_mixed_dataset()
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    print("Setting up trainer...")
    trainer = CustomTrainer(config, model, tokenizer, train_dataset, eval_dataset)
    
    print("Starting training...")
    trainer.train()
    
    print("Training completed!")
    print(f"Model saved to: {config.output_dir}")
    print("\nMedzy AI now has:")
    print("- Personal knowledge about Mohamed Amara (Medzy)")
    print("- General programming and web development knowledge")
    print("- Ability to think beyond the training data")
    print("- A unique personality as Medzy AI assistant")

if __name__ == "__main__":
    main()