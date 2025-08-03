import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import TrainingConfig

class InferenceEngine:
    def __init__(self, model_path: str, config: TrainingConfig):
        self.config = config
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if self.config.use_peft:
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        
    def generate_response(self, instruction: str, input_text: str = "", max_length: int = 512):
        if self.model is None:
            self.load_model()
        
        if input_text:
            prompt = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return response
    
    def batch_generate(self, instructions: list, max_length: int = 512):
        responses = []
        for instruction in instructions:
            response = self.generate_response(instruction, max_length=max_length)
            responses.append(response)
        return responses

def main():
    config = TrainingConfig()
    inference = InferenceEngine("./finetuned_model/final_model", config)
    
    test_questions = [
        "Write a Python function to calculate the factorial of a number",
        "Explain what is deep learning",
        "How do you implement a binary search algorithm?"
    ]
    
    print("Testing fine-tuned model:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 30)
        response = inference.generate_response(question)
        print(f"Response: {response}")
        print()

if __name__ == "__main__":
    main() 