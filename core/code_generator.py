import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any
import re

class CodeGenerator:
    def __init__(self, device="auto"):
        self.device = self._setup_device(device)
        self.models = {}
        self.current_model = None
        self.tokenizer = None
    
    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self, model_name="CodeGen-2B"):
        if model_name in self.models:
            self.current_model = self.models[model_name]
            self.tokenizer = self.models[f"{model_name}_tokenizer"]
            return
        
        model_map = {
            "CodeGen-2B": "Salesforce/codegen-2B-mono",
            "CodeLlama-7B": "codellama/CodeLlama-7b-hf",
            "StarCoder-1B": "bigcode/starcoderbase-1b",
            "InCoder-1B": "facebook/incoder-1B"
        }
        
        model_id = model_map.get(model_name, "Salesforce/codegen-2B-mono")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            model.to(self.device)
            
            self.models[model_name] = model
            self.models[f"{model_name}_tokenizer"] = tokenizer
            self.current_model = model
            self.tokenizer = tokenizer
            
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {str(e)}")
    
    def generate_code(self, prompt: str, language: str, temperature: float = 0.7, 
                     max_length: int = 300, num_return_sequences: int = 3) -> List[str]:
        self.load_model()
        
        language_prefix = self._get_language_prefix(language)
        full_prompt = f"{language_prefix}\n{prompt}\n"
        
        try:
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.current_model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.95,
                    early_stopping=True
                )
            
            generated_codes = []
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                code = self._extract_code_from_text(generated_text, full_prompt)
                if code:
                    generated_codes.append(code)
            
            return generated_codes
            
        except Exception as e:
            raise Exception(f"Code generation failed: {str(e)}")
    
    def generate_variations(self, prompt: str, language: str, num_variations: int = 3) -> List[str]:
        variations = []
        
        for temp in [0.3, 0.7, 1.0]:
            codes = self.generate_code(
                prompt=prompt,
                language=language,
                temperature=temp,
                num_return_sequences=1
            )
            variations.extend(codes)
        
        return variations[:num_variations]
    
    def generate_with_context(self, prompt: str, context: Dict[str, Any]) -> str:
        context_prompt = self._build_context_prompt(prompt, context)
        
        codes = self.generate_code(
            prompt=context_prompt,
            language=context.get('main_language', 'python'),
            num_return_sequences=1
        )
        
        return codes[0] if codes else ""
    
    def complete_code(self, partial_code: str, language: str, max_tokens: int = 100) -> str:
        self.load_model()
        
        inputs = self.tokenizer.encode(partial_code, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.current_model.generate(
                inputs,
                max_length=len(inputs[0]) + max_tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        
        completed_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completed_text[len(partial_code):]
    
    def _get_language_prefix(self, language: str) -> str:
        prefixes = {
            "python": "# Python code\n",
            "javascript": "// JavaScript code\n",
            "java": "// Java code\n",
            "cpp": "// C++ code\n",
            "typescript": "// TypeScript code\n",
            "go": "// Go code\n"
        }
        return prefixes.get(language, "")
    
    def _extract_code_from_text(self, text: str, prompt: str) -> str:
        text = text.replace(prompt, "").strip()
        
        lines = text.split('\n')
        code_lines = []
        
        for line in lines:
            if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('#'):
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def _build_context_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        context_info = f"Project: {context.get('name', 'Unknown')}\n"
        context_info += f"Main Language: {context.get('main_language', 'python')}\n"
        
        if 'structure' in context:
            context_info += f"Project Structure: {context['structure']}\n"
        
        if 'dependencies' in context:
            context_info += f"Dependencies: {', '.join(context['dependencies'])}\n"
        
        context_info += f"\nTask: {prompt}\n"
        context_info += "Generate code that fits well with this project context:\n"
        
        return context_info

class AdvancedCodeGenerator(CodeGenerator):
    def __init__(self, device="auto"):
        super().__init__(device)
        self.conversation_history = []
    
    def chat_generate(self, message: str, language: str, context: List[str] = None) -> str:
        if context:
            self.conversation_history.extend(context)
        
        self.conversation_history.append(f"User: {message}")
        
        conversation_text = "\n".join(self.conversation_history[-6:])
        full_prompt = f"Programming conversation:\n{conversation_text}\nAssistant:"
        
        code = self.generate_code(full_prompt, language, num_return_sequences=1)[0]
        
        self.conversation_history.append(f"Assistant: {code}")
        
        return code
    
    def debug_code(self, code: str, error: str, language: str) -> str:
        prompt = f"Debug this {language} code:\n\n{code}\n\nError: {error}\n\nFixed code:"
        return self.generate_code(prompt, language, num_return_sequences=1)[0]
    
    def refactor_code(self, code: str, language: str, goal: str = "improve readability") -> str:
        prompt = f"Refactor this {language} code to {goal}:\n\n{code}\n\nRefactored code:"
        return self.generate_code(prompt, language, num_return_sequences=1)[0]