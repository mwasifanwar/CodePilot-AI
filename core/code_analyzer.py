import ast
import subprocess
import tempfile
import os
from typing import List, Dict, Any
import re

class CodeAnalyzer:
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java', 'cpp']
    
    def analyze_code(self, code: str, language: str, enable_linting: bool = True,
                    enable_type_checking: bool = True, enable_security_scan: bool = True) -> Dict[str, Any]:
        results = {
            'quality_issues': [],
            'security_issues': [],
            'type_issues': [],
            'suggestions': [],
            'metrics': {}
        }
        
        if language == 'python':
            if enable_linting:
                results.update(self._analyze_python_code(code))
            
            if enable_type_checking:
                results['type_issues'].extend(self._type_check_python(code))
            
            if enable_security_scan:
                results['security_issues'].extend(self._security_scan_python(code))
        
        elif language == 'javascript':
            results.update(self._analyze_javascript_code(code))
        
        results['metrics'] = self._calculate_code_metrics(code, language)
        
        return results
    
    def _analyze_python_code(self, code: str) -> Dict[str, Any]:
        issues = []
        suggestions = []
        
        try:
            tree = ast.parse(code)
            
            issues.extend(self._check_python_best_practices(tree))
            issues.extend(self._check_python_complexity(code))
            suggestions.extend(self._suggest_python_improvements(tree, code))
            
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
        
        return {'quality_issues': issues, 'suggestions': suggestions}
    
    def _analyze_javascript_code(self, code: str) -> Dict[str, Any]:
        issues = []
        suggestions = []
        
        issues.extend(self._check_javascript_patterns(code))
        suggestions.extend(self._suggest_javascript_improvements(code))
        
        return {'quality_issues': issues, 'suggestions': suggestions}
    
    def _check_python_best_practices(self, tree: ast.AST) -> List[str]:
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 5:
                    issues.append(f"Function '{node.name}' has too many parameters")
            
            if isinstance(node, ast.ListComp):
                if len(node.generators) > 2:
                    issues.append("Complex list comprehension detected")
            
            if isinstance(node, ast.ClassDef):
                if not node.name[0].isupper():
                    issues.append(f"Class name '{node.name}' should use CamelCase")
        
        return issues
    
    def _check_python_complexity(self, code: str) -> List[str]:
        issues = []
        
        lines = code.split('\n')
        if len(lines) > 50:
            issues.append("Function might be too long - consider breaking it down")
        
        if code.count('for') + code.count('while') > 3:
            issues.append("High loop complexity detected")
        
        if code.count('if') > 5:
            issues.append("Many conditional branches - consider simplifying")
        
        return issues
    
    def _check_javascript_patterns(self, code: str) -> List[str]:
        issues = []
        
        if 'eval(' in code:
            issues.append("Avoid using eval() for security reasons")
        
        if 'var ' in code and ('let ' in code or 'const ' in code):
            issues.append("Consistent variable declaration (use let/const instead of var)")
        
        if code.count('function') > 3 and 'class' not in code:
            issues.append("Consider using classes for better organization")
        
        return issues
    
    def _type_check_python(self, code: str) -> List[str]:
        issues = []
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(['mypy', '--ignore-missing-imports', temp_file], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                for line in result.stdout.split('\n'):
                    if 'error:' in line:
                        issues.append(line.strip())
            
            os.unlink(temp_file)
            
        except Exception:
            pass
        
        return issues
    
    def _security_scan_python(self, code: str) -> List[str]:
        security_issues = []
        
        dangerous_patterns = [
            (r'exec\(', 'Avoid exec() for security'),
            (r'eval\(', 'Avoid eval() for security'),
            (r'__import__\(', 'Consider using importlib instead'),
            (r'pickle\.loads', 'Be cautious with pickle - use json for untrusted data'),
            (r'subprocess\.call', 'Validate all subprocess inputs'),
            (r'os\.system', 'Use subprocess with explicit arguments instead'),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                security_issues.append(message)
        
        if 'input()' in code and 'int(' not in code and 'float(' not in code:
            security_issues.append("Validate user input from input() function")
        
        return security_issues
    
    def _suggest_python_improvements(self, tree: ast.AST, code: str) -> List[str]:
        suggestions = []
        
        if 'range(len(' in code:
            suggestions.append("Consider using enumerate() instead of range(len())")
        
        if ' == True' in code or ' == False' in code:
            suggestions.append("Use truthy/falsy evaluation instead of explicit True/False comparison")
        
        if 'type(' in code and 'isinstance(' not in code:
            suggestions.append("Prefer isinstance() over type() for type checking")
        
        return suggestions
    
    def _suggest_javascript_improvements(self, code: str) -> List[str]:
        suggestions = []
        
        if 'function ' in code and '=>' not in code:
            suggestions.append("Consider using arrow functions for better scoping")
        
        if 'var ' in code:
            suggestions.append("Use const for variables that won't be reassigned")
        
        if '==' in code or '!=' in code:
            suggestions.append("Use === and !== for strict equality checks")
        
        return suggestions
    
    def _calculate_code_metrics(self, code: str, language: str) -> Dict[str, Any]:
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        metrics = {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'characters': len(code),
            'functions': len(re.findall(r'def \w+', code) if language == 'python' else re.findall(r'function \w+', code)),
            'classes': len(re.findall(r'class \w+', code) if language == 'python' else re.findall(r'class \w+', code)),
        }
        
        return metrics

class AdvancedCodeAnalyzer(CodeAnalyzer):
    def __init__(self):
        super().__init__()
    
    def find_bugs(self, code: str, language: str) -> List[str]:
        bugs = []
        
        if language == 'python':
            bugs.extend(self._find_python_bugs(code))
        elif language == 'javascript':
            bugs.extend(self._find_javascript_bugs(code))
        
        return bugs
    
    def _find_python_bugs(self, code: str) -> List[str]:
        bugs = []
        
        if 'except:' in code and 'except Exception:' not in code:
            bugs.append("Bare except clause - specify exception types")
        
        if ' mutable=' in code.lower():
            bugs.append("Potential mutable default argument issue")
        
        if 'global ' in code:
            bugs.append("Global variables can lead to unexpected behavior")
        
        return bugs
    
    def _find_javascript_bugs(self, code: str) -> List[str]:
        bugs = []
        
        if 'null' in code and 'undefined' in code:
            bugs.append("Be consistent with null/undefined usage")
        
        if '==' in code and '===' not in code:
            bugs.append("Potential type coercion issues with ==")
        
        return bugs
    
    def optimize_code(self, code: str, language: str) -> str:
        if language == 'python':
            return self._optimize_python(code)
        elif language == 'javascript':
            return self._optimize_javascript(code)
        return code
    
    def _optimize_python(self, code: str) -> str:
        optimizations = [
            (r'len\(range\((\w+)\)\)', r'\1'),
            (r'for i in range\(len\((\w+)\)\):', r'for item in \1:'),
            (r'\[x for x in (\w+) if x\]', r'list(filter(None, \1))'),
        ]
        
        optimized = code
        for pattern, replacement in optimizations:
            optimized = re.sub(pattern, replacement, optimized)
        
        return optimized
    
    def _optimize_javascript(self, code: str) -> str:
        optimizations = [
            (r'function\s+(\w+)\s*\(\)\s*{', r'const \1 = () => {'),
            (r'Array\.from\(document\.querySelectorAll\(', r'document.querySelectorAll('),
        ]
        
        optimized = code
        for pattern, replacement in optimizations:
            optimized = re.sub(pattern, replacement, optimized)
        
        return optimized