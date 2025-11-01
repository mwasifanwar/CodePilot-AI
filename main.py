import streamlit as st
import torch
import os
from pathlib import Path
import tempfile
from core.code_generator import CodeGenerator
from core.code_analyzer import CodeAnalyzer
from core.context_engine import ContextEngine
from core.model_manager import ModelManager
from utils.code_utils import parse_code, format_code, validate_syntax
from utils.config import load_config

st.set_page_config(
    page_title="CodePilot AI - Intelligent Code Assistant - Wasif",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'code_generator' not in st.session_state:
        st.session_state.code_generator = None
    if 'code_analyzer' not in st.session_state:
        st.session_state.code_analyzer = None
    if 'context_engine' not in st.session_state:
        st.session_state.context_engine = None
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = []
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None

def load_models():
    with st.spinner("🔄 Loading AI models..."):
        if st.session_state.code_generator is None:
            st.session_state.code_generator = CodeGenerator()
        if st.session_state.code_analyzer is None:
            st.session_state.code_analyzer = CodeAnalyzer()
        if st.session_state.context_engine is None:
            st.session_state.context_engine = ContextEngine()

def main():
    st.title("🤖 CodePilot AI - Intelligent Code Assistant")
    st.markdown("Generate, analyze, and debug code with AI-powered assistance!")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_choice = st.selectbox(
            "AI Model",
            ["CodeGen-2B", "CodeLlama-7B", "StarCoder-1B", "InCoder-1B"],
            help="Select the code generation model"
        )
        
        language_options = {
            "Python": "python",
            "JavaScript": "javascript", 
            "Java": "java",
            "C++": "cpp",
            "TypeScript": "typescript",
            "Go": "go"
        }
        selected_language = st.selectbox("Programming Language", list(language_options.keys()))
        
        st.subheader("Generation Parameters")
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
        max_length = st.slider("Max Length", 100, 1000, 300, 50)
        num_suggestions = st.slider("Suggestions", 1, 5, 3)
        
        st.subheader("Analysis Options")
        enable_linting = st.checkbox("Enable Code Linting", value=True)
        enable_type_checking = st.checkbox("Enable Type Checking", value=True)
        enable_security_scan = st.checkbox("Security Analysis", value=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["💻 Code Generation", "🔍 Code Analysis", "📁 Project Context", "📊 Code Insights"])
    
    with tab1:
        st.header("Code Generation")
        st.markdown("Generate code from natural language descriptions")
        
        prompt = st.text_area(
            "Describe what you want to code",
            "Create a Python function that calculates the fibonacci sequence up to n numbers",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Generate Code", type="primary"):
                generate_code(
                    prompt,
                    language_options[selected_language],
                    temperature,
                    max_length,
                    num_suggestions
                )
        
        with col2:
            if st.button("🔄 Generate Multiple Variations"):
                generate_variations(
                    prompt,
                    language_options[selected_language],
                    num_suggestions
                )
        
        if st.session_state.generated_code:
            st.subheader("Generated Code")
            for idx, (code, metadata) in enumerate(st.session_state.generated_code[-3:]):
                with st.expander(f"Solution {idx+1} - {metadata['language']}"):
                    st.code(code, language=metadata['language'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"📋 Copy {idx+1}", key=f"copy_{idx}"):
                            st.code(code, language=metadata['language'])
                    with col2:
                        if st.button(f"🔍 Analyze {idx+1}", key=f"analyze_{idx}"):
                            analyze_generated_code(code, metadata['language'])
                    with col3:
                        if st.button(f"💾 Save {idx+1}", key=f"save_{idx}"):
                            save_generated_code(code, metadata['language'], idx)
    
    with tab2:
        st.header("Code Analysis & Debugging")
        st.markdown("Analyze and debug your code with AI assistance")
        
        code_input = st.text_area(
            "Paste your code here",
            "def calculate_fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib",
            height=300
        )
        
        if st.button("🔍 Analyze Code", type="primary"):
            analyze_code(
                code_input,
                language_options[selected_language],
                enable_linting,
                enable_type_checking,
                enable_security_scan
            )
    
    with tab3:
        st.header("Project Context")
        st.markdown("Understand and work with your entire codebase")
        
        uploaded_project = st.file_uploader(
            "Upload your project (ZIP)",
            type=['zip'],
            help="Upload a ZIP file of your project for context-aware assistance"
        )
        
        if uploaded_project is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                tmp_file.write(uploaded_project.getvalue())
                project_path = tmp_file.name
            
            if st.button("📂 Load Project Context"):
                load_project_context(project_path)
        
        if st.session_state.current_project:
            st.success(f"✅ Project loaded: {st.session_state.current_project['name']}")
            
            project_context = st.session_state.context_engine.get_project_summary()
            st.write("**Project Summary:**")
            st.write(project_context)
            
            if st.button("🔄 Generate Project-Specific Code"):
                project_prompt = st.text_input("What would you like to build in this project?")
                if project_prompt:
                    generate_project_code(project_prompt)
    
    with tab4:
        st.header("Code Insights")
        st.markdown("Get insights about your coding patterns and improvements")
        
        if st.session_state.generated_code:
            st.subheader("Generation History")
            for idx, (code, metadata) in enumerate(st.session_state.generated_code):
                with st.expander(f"Code {idx+1} - {metadata['timestamp']}"):
                    st.write(f"**Language:** {metadata['language']}")
                    st.write(f"**Prompt:** {metadata['prompt']}")
                    st.code(code, language=metadata['language'])
        else:
            st.info("No code generated yet. Start by generating some code!")

def generate_code(prompt, language, temperature, max_length, num_suggestions):
    load_models()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Generating code...")
        generated_codes = st.session_state.code_generator.generate_code(
            prompt=prompt,
            language=language,
            temperature=temperature,
            max_length=max_length,
            num_return_sequences=num_suggestions
        )
        progress_bar.progress(100)
        
        for code in generated_codes:
            metadata = {
                "language": language,
                "prompt": prompt,
                "temperature": temperature,
                "timestamp": "now"
            }
            st.session_state.generated_code.append((code, metadata))
        
        status_text.text("✅ Code generation complete!")
        
    except Exception as e:
        st.error(f"❌ Code generation failed: {str(e)}")

def generate_variations(prompt, language, num_variations):
    load_models()
    
    with st.spinner("🔄 Generating variations..."):
        try:
            variations = st.session_state.code_generator.generate_variations(
                prompt=prompt,
                language=language,
                num_variations=num_variations
            )
            
            for variation in variations:
                metadata = {
                    "language": language,
                    "prompt": prompt,
                    "type": "variation",
                    "timestamp": "now"
                }
                st.session_state.generated_code.append((variation, metadata))
            
            st.success(f"✅ Generated {len(variations)} variations!")
            
        except Exception as e:
            st.error(f"❌ Variation generation failed: {str(e)}")

def analyze_code(code, language, enable_linting, enable_type_checking, enable_security_scan):
    load_models()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 Analyzing code...")
        analysis_results = st.session_state.code_analyzer.analyze_code(
            code=code,
            language=language,
            enable_linting=enable_linting,
            enable_type_checking=enable_type_checking,
            enable_security_scan=enable_security_scan
        )
        progress_bar.progress(100)
        
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Code Quality:**")
            if analysis_results.get('quality_issues'):
                for issue in analysis_results['quality_issues']:
                    st.warning(f"⚠️ {issue}")
            else:
                st.success("✅ No quality issues found")
            
            st.write("**Security Issues:**")
            if analysis_results.get('security_issues'):
                for issue in analysis_results['security_issues']:
                    st.error(f"🔒 {issue}")
            else:
                st.success("✅ No security issues found")
        
        with col2:
            st.write("**Type Issues:**")
            if analysis_results.get('type_issues'):
                for issue in analysis_results['type_issues']:
                    st.info(f"📝 {issue}")
            else:
                st.success("✅ No type issues found")
            
            st.write("**Suggestions:**")
            if analysis_results.get('suggestions'):
                for suggestion in analysis_results['suggestions']:
                    st.info(f"💡 {suggestion}")
        
        status_text.text("✅ Code analysis complete!")
        
    except Exception as e:
        st.error(f"❌ Code analysis failed: {str(e)}")

def analyze_generated_code(code, language):
    analyze_code(code, language, True, True, True)

def save_generated_code(code, language, index):
    filename = f"generated_code_{index}.{get_file_extension(language)}"
    with open(f"outputs/{filename}", "w") as f:
        f.write(code)
    st.success(f"✅ Code saved as {filename}")

def load_project_context(project_path):
    load_models()
    
    with st.spinner("📂 Loading project context..."):
        try:
            project_context = st.session_state.context_engine.load_project(project_path)
            st.session_state.current_project = project_context
            st.success("✅ Project context loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load project: {str(e)}")

def generate_project_code(prompt):
    if not st.session_state.current_project:
        st.error("Please load a project first")
        return
    
    with st.spinner("🔄 Generating project-specific code..."):
        try:
            project_code = st.session_state.code_generator.generate_with_context(
                prompt=prompt,
                context=st.session_state.current_project
            )
            
            st.subheader("Project-Specific Code")
            st.code(project_code, language=st.session_state.current_project['main_language'])
            
        except Exception as e:
            st.error(f"❌ Project code generation failed: {str(e)}")

def get_file_extension(language):
    extensions = {
        "python": "py",
        "javascript": "js",
        "java": "java",
        "cpp": "cpp",
        "typescript": "ts",
        "go": "go"
    }
    return extensions.get(language, "txt")

if __name__ == "__main__":
    main()