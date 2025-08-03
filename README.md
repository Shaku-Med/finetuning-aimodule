# DeepSeek R1 Fine-tuning Framework

A complete Python framework for fine-tuning DeepSeek R1 models using LoRA (Low-Rank Adaptation) and instruction-following datasets.

## Features

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **4-bit Quantization**: Memory-efficient training with BitsAndBytes
- **Instruction Following**: Optimized for instruction-following tasks
- **Flexible Data Loading**: Support for JSON and CSV data formats
- **Easy Configuration**: Centralized configuration management
- **Inference Ready**: Built-in inference engine for testing

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have sufficient GPU memory (recommended: 8GB+ for 1.3B model)

## Usage

### 1. Prepare Your Data

Create a JSON file with instruction-following format:
```json
[
    {
        "instruction": "Your instruction here",
        "input": "Optional input context",
        "output": "Expected output/response"
    }
]
```

### 2. Configure Training

Edit `config.py` to customize training parameters:
- Model name and size
- Learning rate and batch size
- LoRA parameters
- Training epochs

### 3. Start Training

```bash
python main.py
```

### 4. Test the Model

```bash
python inference.py
```

## File Structure

```
ChatAI/
├── config.py              # Training configuration
├── data_processor.py      # Data loading and preprocessing
├── model_setup.py         # Model and tokenizer setup
├── trainer.py             # Custom training logic
├── inference.py           # Inference engine
├── main.py               # Main training script
├── sample_data.json      # Example training data
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Configuration Options

### Model Settings
- `model_name`: DeepSeek model to fine-tune
- `use_peft`: Enable LoRA fine-tuning
- `lora_r`: LoRA rank
- `lora_alpha`: LoRA alpha parameter

### Training Settings
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation steps

### Hardware Settings
- `fp16`: Enable FP16 training
- `bf16`: Enable BF16 training
- `dataloader_num_workers`: Number of data loader workers

## Customization

### Adding Custom Data

1. Create your dataset in the instruction format
2. Save as JSON or CSV
3. Update the data loading path in `main.py`

### Modifying Model Architecture

1. Edit `model_setup.py` for different model configurations
2. Adjust LoRA target modules in `config.py`
3. Modify quantization settings as needed

### Custom Training Logic

1. Extend `CustomTrainer` class in `trainer.py`
2. Add custom evaluation metrics
3. Implement custom loss functions

## Troubleshooting

### Memory Issues
- Reduce batch size
- Enable gradient accumulation
- Use smaller LoRA rank
- Enable 4-bit quantization

### Training Issues
- Check data format
- Verify model compatibility
- Monitor learning rate
- Check for gradient clipping

## Performance Tips

1. **Use LoRA**: Reduces memory usage significantly
2. **4-bit Quantization**: Further memory savings
3. **Gradient Accumulation**: Effective larger batch sizes
4. **Mixed Precision**: Faster training with FP16/BF16
5. **Data Quality**: Ensure high-quality instruction data

## License

This project is for educational and research purposes. 