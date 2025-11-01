import torch
import os
from pathlib import Path
from typing import Dict, Any
from huggingface_hub import snapshot_download

class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        
        self.model_configs = {
            "codegen-2b": {
                "repo_id": "Salesforce/codegen-2B-mono",
                "files": ["pytorch_model.bin", "config.json"],
                "type": "generation"
            },
            "codellama-7b": {
                "repo_id": "codellama/CodeLlama-7b-hf",
                "files": ["pytorch_model.bin", "config.json"],
                "type": "generation"
            },
            "starcoder-1b": {
                "repo_id": "bigcode/starcoderbase-1b",
                "files": ["pytorch_model.bin", "config.json"],
                "type": "generation"
            },
            "incoder-1b": {
                "repo_id": "facebook/incoder-1B",
                "files": ["pytorch_model.bin", "config.json"],
                "type": "generation"
            }
        }
    
    def download_model(self, model_name: str) -> str:
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_path = self.model_dir / model_name
        config = self.model_configs[model_name]
        
        if model_path.exists():
            return str(model_path)
        
        try:
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            return str(model_path)
        except Exception as e:
            raise Exception(f"Failed to download model {model_name}: {str(e)}")
    
    def load_model(self, model_name: str, force_reload: bool = False):
        if model_name in self.loaded_models and not force_reload:
            return self.loaded_models[model_name]
        
        model_path = self.download_model(model_name)
        config = self.model_configs[model_name]
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.loaded_models[model_name] = model
            self.loaded_models[f"{model_name}_tokenizer"] = tokenizer
            return model
        
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {str(e)}")
    
    def unload_model(self, model_name: str):
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_available_models(self):
        return list(self.model_configs.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        model_path = self.model_dir / model_name
        
        return {
            "name": model_name,
            "repo_id": config["repo_id"],
            "type": config["type"],
            "downloaded": model_path.exists(),
            "loaded": model_name in self.loaded_models,
            "path": str(model_path)
        }