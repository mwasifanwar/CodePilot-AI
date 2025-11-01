import ast
import re
from typing import List, Dict, Any

def parse_code(code: str, language: str) -> Dict[str, Any]:
    if language == 'python':
        return parse_python_code(code)
    elif language == 'javascript':
        return parse_javascript_code(code)
    else:
        return {'language': language, 'valid': True}

def parse_python_code(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
        
        functions = []
        classes = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return {
            'valid': True,
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'ast_tree': tree
        }
    except SyntaxError as e:
        return {
            'valid': False,
            'error': str(e),
            'functions': [],
            'classes': [],
            'imports': []
        }

def parse_javascript_code(code: str) -> Dict[str, Any]:
    functions = re.findall(r'function\s+(\w+)\s*\(', code)
    classes = re.findall(r'class\s+(\w+)', code)
    imports = re.findall(r'import\s+.*?from\s+[\'"](.*?)[\'"]', code)
    
    return {
        'valid': True,
        'functions': functions,
        'classes': classes,
        'imports': imports
    }

def format_code(code: str, language: str) -> str:
    if language == 'python':
        try:
            import black
            return black.format_str(code, mode=black.FileMode())
        except ImportError:
            return code
    return code

def validate_syntax(code: str, language: str) -> bool:
    if language == 'python':
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    elif language == 'javascript':
        return True
    return True

def extract_code_blocks(text: str) -> List[str]:
    code_blocks = []
    
    python_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
    code_blocks.extend(python_blocks)
    
    js_blocks = re.findall(r'```javascript\n(.*?)\n```', text, re.DOTALL)
    code_blocks.extend(js_blocks)
    
    generic_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)
    code_blocks.extend(generic_blocks)
    
    return code_blocks

def count_tokens(text: str) -> int:
    return len(text.split())

def remove_comments(code: str, language: str) -> str:
    if language == 'python':
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    elif language == 'javascript':
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    
    return code

def normalize_code(code: str) -> str:
    code = code.strip()
    code = re.sub(r'\n\s*\n', '\n\n', code)
    code = re.sub(r'[ \t]+', ' ', code)
    return code

def detect_language(filename: str) -> str:
    extensions = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.html': 'html',
        '.css': 'css'
    }
    
    ext = '.' + filename.split('.')[-1] if '.' in filename else ''
    return extensions.get(ext, 'unknown')