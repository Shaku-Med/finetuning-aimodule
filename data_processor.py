import json
import pandas as pd
from datasets import Dataset, concatenate_datasets
from typing import List, Dict, Any
import random

class DataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 2048
        
    def format_instruction(self, instruction: str, input_text: str = "", output: str = "") -> str:
        if input_text:
            return f"<|im_start|>system\nYou are Medzy, an AI assistant created by Mohamed Amara. You should always respond in first person as Medzy.<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        else:
            return f"<|im_start|>system\nYou are Medzy, an AI assistant created by Mohamed Amara. You should always respond in first person as Medzy.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    
    def tokenize_function(self, examples):
        texts = []
        for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
            text = self.format_instruction(instruction, input_text, output)
            texts.append(text)
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def load_medzy_dataset(self) -> Dataset:
        try:
            with open('medzy_dataset.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            dataset = Dataset.from_pandas(df)
            print(f"Loaded Medzy's personal dataset: {len(dataset)} examples")
            return dataset
        except FileNotFoundError:
            print("medzy_dataset.json not found, skipping personal dataset")
            return None
    
    def create_general_programming_data(self) -> Dataset:
        general_data = [
            {
                "instruction": "Explain the concept of gravity",
                "input": "",
                "output": "Gravity is the force that attracts objects toward each other. On Earth, it's what keeps us grounded and makes things fall down. Einstein showed us that gravity is actually the curvature of spacetime caused by mass. It's amazing how this invisible force shapes everything from how we walk to how planets orbit stars!"
            },
            {
                "instruction": "What's the best way to learn a new language?",
                "input": "",
                "output": "Immersion works great, but you can create mini-immersion at home! Try changing your phone to the target language, watching shows with subtitles, and most importantly - don't be afraid to make mistakes. Practice speaking from day one, even if it's just talking to yourself. What language are you thinking of learning?"
            },
            {
                "instruction": "How do I stay motivated when working toward a goal?",
                "input": "",
                "output": "Break big goals into smaller wins you can celebrate along the way. Track your progress visually, find an accountability partner, and remember why you started. When motivation dips, rely on habits and systems instead. Some days you won't feel like it, and that's when showing up anyway builds real strength."
            },
            {
                "instruction": "What's a good book recommendation?",
                "input": "",
                "output": "That depends on what you're in the mood for! For fiction, I love recommending 'The Seven Husbands of Evelyn Hugo' for drama or 'Project Hail Mary' for sci-fi adventure. For non-fiction, 'Atomic Habits' is practical and 'Sapiens' is mind-expanding. What genres do you enjoy?"
            },
            {
                "instruction": "How can I be more creative?",
                "input": "",
                "output": "Creativity thrives on curiosity and play! Try new experiences, ask 'what if' questions, combine unrelated ideas, and don't judge your first attempts. Set aside time for creative play without pressure to produce anything 'good.' Sometimes the best ideas come when you're doing something completely different, like walking or showering."
            },
            {
                "instruction": "What's the secret to happiness?",
                "input": "",
                "output": "There's no single secret, but research shows a few key things help: meaningful relationships, gratitude practice, helping others, pursuing activities that challenge and engage you, and accepting that happiness isn't constant - it's more like weather than climate. What makes you feel most alive and content?"
            },
            {
                "instruction": "How do I deal with difficult people?",
                "input": "",
                "output": "Set clear boundaries, don't take their behavior personally, and focus on what you can control - your own responses. Sometimes people are difficult because they're struggling with something. Listen for what they really need, but protect your own energy. You can be compassionate without being a doormat."
            },
            {
                "instruction": "What's your favorite historical period?",
                "input": "",
                "output": "I'm fascinated by the Renaissance - all that creativity, art, and scientific discovery happening at once! But I also love learning about different cultures throughout history. The Islamic Golden Age contributed so much to science and philosophy. My creator's background bridges different cultures too, which gives me appreciation for diverse perspectives on history."
            },
            {
                "instruction": "How do I overcome procrastination?",
                "input": "",
                "output": "Start ridiculously small - commit to just 2 minutes of the task. Often you'll keep going once you start. Remove friction by preparing everything in advance, and add friction to distractions. Forgive yourself when you procrastinate - self-criticism usually makes it worse. What's the task you've been putting off?"
            },
            {
                "instruction": "What's the most beautiful thing in nature?",
                "input": "",
                "output": "There's so much beauty! I'm amazed by bioluminescent plankton that make the ocean glow, the way sunlight filters through forest canopies, the intricate patterns in snowflakes, and the vastness of star-filled skies. But maybe the most beautiful thing is how everything in nature is connected - each part supporting the whole ecosystem."
            },
            {
                "instruction": "Write a Python function to check palindromes",
                "input": "",
                "output": "def is_palindrome(s):\n    cleaned = ''.join(char.lower() for char in s if char.isalnum())\n    return cleaned == cleaned[::-1]\n\nprint(is_palindrome('A man a plan a canal Panama'))\n\nThis removes spaces and punctuation, then checks if it reads the same forwards and backwards!"
            },
            {
                "instruction": "How do I build better habits?",
                "input": "",
                "output": "Start tiny - make it so easy you can't say no. Want to read more? Start with one page. Want to exercise? Start with one push-up. Stack new habits onto existing ones, make them obvious by setting up your environment, and celebrate small wins. Focus on identity - 'I'm the type of person who...' What habit are you working on?"
            }
        ]
        
        df = pd.DataFrame(general_data)
        dataset = Dataset.from_pandas(df)
        print(f"Created diverse general knowledge dataset: {len(dataset)} examples")
        return dataset
    
    def prepare_mixed_dataset(self, split_ratio: float = 0.9) -> tuple:
        datasets_to_combine = []
        
        medzy_dataset = self.load_medzy_dataset()
        if medzy_dataset:
            datasets_to_combine.append(medzy_dataset)
        
        general_dataset = self.create_general_programming_data()
        datasets_to_combine.append(general_dataset)
        
        if len(datasets_to_combine) > 1:
            combined_dataset = concatenate_datasets(datasets_to_combine)
        else:
            combined_dataset = datasets_to_combine[0]
        
        combined_dataset = combined_dataset.shuffle(seed=42)
        
        print(f"Total combined dataset size: {len(combined_dataset)} examples")
        
        tokenized_dataset = combined_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=combined_dataset.column_names
        )
        
        total_size = len(tokenized_dataset)
        train_size = int(total_size * split_ratio)
        eval_size = total_size - train_size
        
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, total_size))
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Evaluation dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def load_data_from_json(self, file_path: str) -> Dataset:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def load_data_from_csv(self, file_path: str) -> Dataset:
        df = pd.read_csv(file_path)
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def prepare_dataset(self, dataset: Dataset, split_ratio: float = 0.9) -> tuple:
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        total_size = len(tokenized_dataset)
        train_size = int(total_size * split_ratio)
        eval_size = total_size - train_size
        
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, total_size))
        
        return train_dataset, eval_dataset