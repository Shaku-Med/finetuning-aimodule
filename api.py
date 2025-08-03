from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import TrainingConfig
import uvicorn
import os

app = FastAPI(title="Medzy AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_history: Optional[List[ChatMessage]] = []
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3

class ChatResponse(BaseModel):
    response: str
    user_id: str
    model_used: str

class MedzyAI:
    def __init__(self):
        self.config = TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        model_path = "./finetuned_model/final_model"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if self.config.use_peft and os.path.exists(model_path):
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = base_model
            
        self.model.eval()
    
    def build_context(self, conversation_history: List[ChatMessage]) -> str:
        if not conversation_history:
            return ""
        
        context = "Previous conversation:\n"
        recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        
        for msg in recent_history:
            if msg.role == "user":
                context += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                context += f"Assistant: {msg.content}\n"
        
        return context
    
    def generate_response(self, message: str, conversation_history: List[ChatMessage], max_tokens: int, temperature: float) -> str:
        context = self.build_context(conversation_history)
        
        if context:
            prompt = f"<|im_start|>system\nYou are Medzy, an AI assistant created by Mohamed Amara. You should always respond in first person as Medzy. Remember the conversation context.<|im_end|>\n<|im_start|>user\n{context}Current message: {message}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\nYou are Medzy, an AI assistant created by Mohamed Amara. You should always respond in first person as Medzy.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=40,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.replace(prompt, "").strip()
        
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        if "<|im_start|>" in response:
            response = response.split("<|im_start|>")[0].strip()
        
        return response

medzy_ai = MedzyAI()

@app.get("/")
async def root():
    return {"message": "Medzy AI API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": medzy_ai.model is not None,
        "device": str(medzy_ai.device)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = medzy_ai.generate_response(
            message=request.message,
            conversation_history=request.conversation_history,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            model_used="Medzy AI v1.0"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/chat/simple")
async def simple_chat(message: str):
    try:
        response = medzy_ai.generate_response(
            message=message,
            conversation_history=[],
            max_tokens=512,
            temperature=0.3
        )
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/model/info")
async def model_info():
    return {
        "model_name": medzy_ai.config.model_name,
        "device": str(medzy_ai.device),
        "use_peft": medzy_ai.config.use_peft,
        "creator": "Mohamed Amara",
        "specialization": "React Native and Software Engineering"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )