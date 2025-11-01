import os
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import json

class ContextEngine:
    def __init__(self):
        self.loaded_projects = {}
        self.current_project = None
    
    def load_project(self, project_path: str) -> Dict[str, Any]:
        project_data = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            if project_path.endswith('.zip'):
                with zipfile.ZipFile(project_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                project_root = temp_dir
            else:
                project_root = project_path
            
            project_data = self._analyze_project_structure(project_root)
            project_data['path'] = project_root
            project_data['name'] = Path(project_path).stem
        
        self.loaded_projects[project_data['name']] = project_data
        self.current_project = project_data
        
        return project_data
    
    def _analyze_project_structure(self, project_root: str) -> Dict[str, Any]:
        project_data = {
            'structure': {},
            'files': [],
            'dependencies': [],
            'main_language': 'unknown',
            'file_count': 0
        }
        
        languages = {}
        
        for root, dirs, files in os.walk(project_root):
            relative_path = os.path.relpath(root, project_root)
            if relative_path == '.':
                relative_path = ''
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_file_path = os.path.join(relative_path, file) if relative_path else file
                
                file_info = {
                    'name': file,
                    'path': relative_file_path,
                    'language': self._detect_language(file),
                    'size': os.path.getsize(file_path)
                }
                
                project_data['files'].append(file_info)
                
                lang = file_info['language']
                if lang:
                    languages[lang] = languages.get(lang, 0) + 1
                
                if file in ['requirements.txt', 'package.json', 'pom.xml', 'Cargo.toml']:
                    project_data['dependencies'].extend(self._extract_dependencies(file_path))
        
        project_data['file_count'] = len(project_data['files'])
        
        if languages:
            project_data['main_language'] = max(languages, key=languages.get)
        
        project_data['structure'] = self._build_tree_structure(project_root)
        
        return project_data
    
    def _detect_language(self, filename: str) -> str:
        extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        ext = Path(filename).suffix.lower()
        return extensions.get(ext, 'unknown')
    
    def _extract_dependencies(self, file_path: str) -> List[str]:
        dependencies = []
        
        try:
            if os.path.basename(file_path) == 'requirements.txt':
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            dependencies.append(line.split('==')[0].split('>=')[0])
            
            elif os.path.basename(file_path) == 'package.json':
                with open(file_path, 'r') as f:
                    package_data = json.load(f)
                    if 'dependencies' in package_data:
                        dependencies.extend(package_data['dependencies'].keys())
                    if 'devDependencies' in package_data:
                        dependencies.extend(package_data['devDependencies'].keys())
        
        except Exception:
            pass
        
        return dependencies
    
    def _build_tree_structure(self, project_root: str) -> Dict[str, Any]:
        structure = {}
        
        for item in os.listdir(project_root):
            item_path = os.path.join(project_root, item)
            
            if os.path.isdir(item_path):
                if item in ['.git', '__pycache__', 'node_modules', '.idea', '.vscode']:
                    continue
                
                structure[item] = self._build_tree_structure(item_path)
            else:
                structure[item] = 'file'
        
        return structure
    
    def get_project_summary(self) -> str:
        if not self.current_project:
            return "No project loaded"
        
        project = self.current_project
        summary = f"Project: {project['name']}\n"
        summary += f"Main Language: {project['main_language']}\n"
        summary += f"Files: {project['file_count']}\n"
        summary += f"Dependencies: {', '.join(project['dependencies'][:5])}\n"
        
        file_types = {}
        for file in project['files']:
            lang = file['language']
            file_types[lang] = file_types.get(lang, 0) + 1
        
        summary += "File Types:\n"
        for lang, count in file_types.items():
            if lang != 'unknown':
                summary += f"  - {lang}: {count} files\n"
        
        return summary
    
    def find_similar_files(self, current_file: str, language: str) -> List[str]:
        if not self.current_project:
            return []
        
        similar_files = []
        for file_info in self.current_project['files']:
            if (file_info['language'] == language and 
                file_info['path'] != current_file and
                os.path.basename(file_info['path']) != '__init__.py'):
                similar_files.append(file_info['path'])
        
        return similar_files[:5]
    
    def get_file_context(self, file_path: str) -> Dict[str, Any]:
        if not self.current_project:
            return {}
        
        full_path = os.path.join(self.current_project['path'], file_path)
        
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            return {
                'content': content,
                'language': self._detect_language(file_path),
                'size': len(content),
                'lines': content.count('\n') + 1
            }
        except Exception:
            return {}

class AdvancedContextEngine(ContextEngine):
    def __init__(self):
        super().__init__()
        self.code_patterns = {}
    
    def extract_code_patterns(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        patterns = {
            'import_patterns': [],
            'function_patterns': [],
            'class_patterns': [],
            'api_patterns': []
        }
        
        for file_info in project_data['files']:
            if file_info['language'] == 'python':
                content = self.get_file_context(file_info['path']).get('content', '')
                patterns['import_patterns'].extend(self._extract_imports(content))
                patterns['function_patterns'].extend(self._extract_functions(content))
                patterns['class_patterns'].extend(self._extract_classes(content))
        
        self.code_patterns[project_data['name']] = patterns
        return patterns
    
    def _extract_imports(self, content: str) -> List[str]:
        imports = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        
        return imports
    
    def _extract_functions(self, content: str) -> List[str]:
        functions = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                functions.append(line[4:].split('(')[0])
        
        return functions
    
    def _extract_classes(self, content: str) -> List[str]:
        classes = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('class '):
                classes.append(line[6:].split('(')[0].split(':')[0])
        
        return classes