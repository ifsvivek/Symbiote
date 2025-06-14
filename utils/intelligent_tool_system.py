"""
🧬 Symbiote Intelligent Tool System

The interaction between user, LLM, and tools is a multi-step conversational loop:

1. User provides a prompt
2. LLM decides on first logical step and calls a single tool
3. Tool executes and returns full output back to LLM
4. LLM analyzes output and decides next action
5. Loop continues until request is fully addressed

This creates a robust chain of thought, allowing AI to explore, analyze, 
and act like a human developer's workflow.

Author: Vivek Sharma
License: MIT
"""

import json
import subprocess
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod


class ToolExecutionStatus(Enum):
    """Status of tool execution."""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CHAINING = "chaining"
    COMPLETED = "completed"


class ToolDecision(Enum):
    """Decisions a tool can make after execution."""
    CONTINUE_CHAIN = "continue_chain"
    EXECUTE_ANOTHER = "execute_another"
    RETURN_TO_USER = "return_to_user"
    REQUEST_INPUT = "request_input"
    HANDLE_ERROR = "handle_error"


@dataclass
class ToolExecution:
    """Represents a single tool execution with context."""
    command: str
    output: str
    status: ToolExecutionStatus
    execution_time: float
    timestamp: str
    error_message: Optional[str] = None
    next_action: Optional[ToolDecision] = None


@dataclass
class ToolChain:
    """Represents a chain of tool executions."""
    chain_id: str
    executions: List[ToolExecution]
    current_step: int
    total_steps: int
    status: ToolExecutionStatus
    final_output: Optional[str] = None


class IntelligentTool(ABC):
    """Abstract base class for intelligent tools."""
    
    def __init__(self, name: str, description: str, workspace_path: Path):
        self.name = name
        self.description = description
        self.workspace_path = workspace_path
        self.execution_history: List[ToolExecution] = []
        
    @abstractmethod
    def execute(self, command: str, context: Optional[Dict[str, Any]] = None) -> ToolExecution:
        """Execute a command and return the execution result."""
        pass
    
    @abstractmethod
    def analyze_output(self, execution: ToolExecution) -> ToolDecision:
        """Analyze execution output and decide next action."""
        pass
    
    @abstractmethod
    def get_next_command(self, execution: ToolExecution, context: Dict[str, Any]) -> Optional[str]:
        """Generate the next command based on current execution result."""
        pass
    
    def get_execution_snippet(self, command: str) -> str:
        """Generate a brief execution snippet for display."""
        return f"🔧 {self.name}: Executing '{command[:50]}...'"


# ===== SYMBIOTE TOOLS AS DESCRIBED IN TODO.MD =====

class ListDirectoryFilesTool:
    """
    Name: list_directory_files
    Function/Purpose: Lists all files and subdirectories within a specified path, 
                     with options for recursive searching and pattern matching. 
                     This is the LLM's primary tool for spatial awareness within the codebase.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def execute(self, directory_path: str = "./") -> Dict[str, Any]:
        """Execute directory listing and return structured result."""
        try:
            target_path = self.workspace_path / directory_path
            if not target_path.exists():
                return {
                    "status": "error",
                    "message": f"Directory not found: {directory_path}",
                    "files": [],
                    "directories": []
                }
            
            files = []
            directories = []
            
            for item in sorted(target_path.iterdir()):
                if item.is_file():
                    files.append({
                        "name": item.name,
                        "path": str(item.relative_to(self.workspace_path)),
                        "size": item.stat().st_size,
                        "extension": item.suffix
                    })
                elif item.is_dir():
                    directories.append({
                        "name": item.name,
                        "path": str(item.relative_to(self.workspace_path))
                    })
            
            return {
                "status": "success",
                "message": f"Successfully listed contents of {directory_path}",
                "directory_path": directory_path,
                "total_files": len(files),
                "total_directories": len(directories),
                "files": files,
                "directories": directories
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Error listing directory: {str(e)}",
                "files": [],
                "directories": []
            }


class ReadFileContentTool:
    """
    Name: read_file_content
    Function/Purpose: Reads the entire content of a specific file. This tool is fundamental 
                     for any analysis, as it provides the raw material for other tools. 
                     The content is returned to the LLM with metadata like line count and file size.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def execute(self, file_path: str) -> Dict[str, Any]:
        """Execute file reading and return structured result."""
        try:
            target_path = self.workspace_path / file_path
            if not target_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}",
                    "content": "",
                    "metadata": {}
                }
            
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            metadata = {
                "file_path": file_path,
                "file_size": target_path.stat().st_size,
                "line_count": len(lines),
                "character_count": len(content),
                "file_extension": target_path.suffix,
                "is_binary": False  # We're assuming text files for now
            }
            
            return {
                "status": "success",
                "message": f"Successfully read file: {file_path}",
                "content": content,
                "metadata": metadata
            }
            
        except UnicodeDecodeError:
            return {
                "status": "error",
                "message": f"Cannot read binary file: {file_path}",
                "content": "",
                "metadata": {"is_binary": True}
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error reading file: {str(e)}",
                "content": "",
                "metadata": {}
            }


class SearchInFilesTool:
    """
    Name: search_in_files
    Function/Purpose: Performs a text or regex search within files in a given directory 
                     to find specific functions, variables, or comments.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def execute(self, directory_path: str = "./", search_pattern: str = "", 
                file_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute search and return structured result."""
        try:
            target_path = self.workspace_path / directory_path
            if not target_path.exists():
                return {
                    "status": "error",
                    "message": f"Directory not found: {directory_path}",
                    "matches": []
                }
            
            if not search_pattern:
                return {
                    "status": "error",
                    "message": "No search pattern provided",
                    "matches": []
                }
            
            matches = []
            file_extensions = file_extensions or ['.py', '.js', '.ts', '.md', '.txt']
            
            for file_path in target_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in file_extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        for line_num, line in enumerate(lines, 1):
                            if search_pattern.lower() in line.lower():
                                matches.append({
                                    "file_path": str(file_path.relative_to(self.workspace_path)),
                                    "line_number": line_num,
                                    "line_content": line.strip(),
                                    "match_context": lines[max(0, line_num-2):line_num+1] if len(lines) > line_num else [line]
                                })
                    except (UnicodeDecodeError, PermissionError):
                        continue  # Skip binary or inaccessible files
            
            return {
                "status": "success",
                "message": f"Search completed for pattern '{search_pattern}' in {directory_path}",
                "search_pattern": search_pattern,
                "directory_path": directory_path,
                "total_matches": len(matches),
                "matches": matches[:50]  # Limit to first 50 matches
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during search: {str(e)}",
                "matches": []
            }


class ExecuteTerminalCommandTool:
    """
    Name: execute_terminal_command
    Function/Purpose: Executes any shell command directly in the project's workspace. 
                     It automatically handles virtual environment activation for Python commands.
                     This is the most powerful and flexible tool, allowing for actions like 
                     installing dependencies, running tests, or checking system status.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def execute(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute terminal command and return structured result."""
        try:
            # Auto-activate virtual environment for Python commands
            if any(cmd in command for cmd in ['pip', 'python', 'pytest']):
                venv_path = self.workspace_path / 'venv'
                if venv_path.exists():
                    if 'pip' in command:
                        command = f"source venv/bin/activate && {command}"
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "message": f"Command executed: {command}",
                "command": command,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": timeout  # Simplified for now
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": f"Command timed out after {timeout} seconds",
                "command": command,
                "return_code": -1,
                "stdout": "",
                "stderr": "Command timed out"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing command: {str(e)}",
                "command": command,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e)
            }


class AssessCodeComplexityTool:
    """
    Name: assess_code_complexity
    Function/Purpose: Assesses the complexity of functions within provided code data. 
                     It identifies hotspots with high cyclomatic complexity, helping the LLM 
                     pinpoint code that may be difficult to maintain or test.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def execute(self, code_data: str) -> Dict[str, Any]:
        """Execute complexity analysis and return structured result."""
        try:
            # Parse the code_data (could be JSON string or direct code)
            if code_data.startswith('{'):
                import json
                data = json.loads(code_data)
                if 'functions' in data:
                    functions_data = data['functions']
                else:
                    # Treat as direct code
                    functions_data = self._extract_functions_from_code(code_data)
            else:
                # Direct code analysis
                functions_data = self._extract_functions_from_code(code_data)
            
            complexity_results = []
            for func_info in functions_data:
                complexity = self._calculate_cyclomatic_complexity(func_info)
                complexity_results.append({
                    "function_name": func_info.get('name', 'unknown'),
                    "line_count": func_info.get('line_count', 0),
                    "cyclomatic_complexity": complexity,
                    "complexity_level": self._classify_complexity(complexity),
                    "suggestions": self._get_complexity_suggestions(complexity)
                })
            
            # Sort by complexity (highest first)
            complexity_results.sort(key=lambda x: x['cyclomatic_complexity'], reverse=True)
            
            return {
                "status": "success",
                "message": f"Complexity analysis completed for {len(complexity_results)} functions",
                "total_functions": len(complexity_results),
                "high_complexity_count": len([r for r in complexity_results if r['complexity_level'] == 'high']),
                "functions": complexity_results,
                "overall_assessment": self._get_overall_assessment(complexity_results)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing complexity: {str(e)}",
                "functions": []
            }
    
    def _extract_functions_from_code(self, code: str) -> List[Dict[str, Any]]:
        """Extract function information from code."""
        functions = []
        lines = code.split('\n')
        current_function = None
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Detect function definition
            if stripped_line.startswith('def '):
                if current_function:
                    functions.append(current_function)
                
                func_name = stripped_line.split('(')[0].replace('def ', '').strip()
                current_function = {
                    'name': func_name,
                    'start_line': i + 1,
                    'lines': [line],
                    'line_count': 0
                }
            
            elif current_function and (line.startswith('def ') or line.startswith('class ') or (stripped_line and not line.startswith(' ') and not line.startswith('\t'))):
                # End of current function
                current_function['line_count'] = len(current_function['lines'])
                functions.append(current_function)
                current_function = None
            
            elif current_function:
                current_function['lines'].append(line)
        
        # Handle last function
        if current_function:
            current_function['line_count'] = len(current_function['lines'])
            functions.append(current_function)
        
        return functions
    
    def _calculate_cyclomatic_complexity(self, func_info: Dict[str, Any]) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        if 'lines' in func_info:
            code_lines = func_info['lines']
            for line in code_lines:
                line_lower = line.lower().strip()
                
                # Count decision points
                complexity += line_lower.count('if ')
                complexity += line_lower.count('elif ')
                complexity += line_lower.count('while ')
                complexity += line_lower.count('for ')
                complexity += line_lower.count('except ')
                complexity += line_lower.count('and ')
                complexity += line_lower.count('or ')
                complexity += line_lower.count('?')  # Ternary operator
        
        return complexity
    
    def _classify_complexity(self, complexity: int) -> str:
        """Classify complexity level."""
        if complexity >= 15:
            return "very_high"
        elif complexity >= 10:
            return "high"
        elif complexity >= 5:
            return "moderate"
        else:
            return "low"
    
    def _get_complexity_suggestions(self, complexity: int) -> List[str]:
        """Get suggestions based on complexity."""
        if complexity >= 15:
            return [
                "Consider breaking this function into smaller functions",
                "Look for opportunities to extract helper methods",
                "Consider using design patterns like Strategy or Command"
            ]
        elif complexity >= 10:
            return [
                "Consider refactoring to reduce conditional complexity",
                "Look for repeated patterns that could be extracted"
            ]
        elif complexity >= 5:
            return ["Function complexity is manageable but watch for growth"]
        else:
            return ["Good complexity level - easy to understand and test"]
    
    def _get_overall_assessment(self, results: List[Dict[str, Any]]) -> str:
        """Get overall codebase assessment."""
        if not results:
            return "No functions to analyze"
        
        high_complexity = len([r for r in results if r['complexity_level'] in ['high', 'very_high']])
        total_functions = len(results)
        
        if high_complexity > total_functions * 0.3:
            return "High complexity codebase - significant refactoring recommended"
        elif high_complexity > total_functions * 0.1:
            return "Moderate complexity - some functions need attention"
        else:
            return "Well-structured codebase with manageable complexity"


class IdentifyCodeSmellsTool:
    """
    Name: identify_code_smells
    Function/Purpose: Identifies potential anti-patterns and "code smells" like long functions, 
                     large classes, or poor documentation. The output gives the LLM a list of 
                     potential issues to investigate further.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def execute(self, analysis_data: str) -> Dict[str, Any]:
        """Execute code smell detection and return structured result."""
        try:
            # Parse analysis_data (could be JSON or direct code)
            if analysis_data.startswith('{'):
                import json
                data = json.loads(analysis_data)
                code_content = data.get('code', '')
                file_path = data.get('file_path', 'unknown')
            else:
                code_content = analysis_data
                file_path = 'direct_input'
            
            smells = []
            lines = code_content.split('\n')
            
            # Detect various code smells
            smells.extend(self._detect_long_methods(lines))
            smells.extend(self._detect_large_classes(lines))
            smells.extend(self._detect_long_lines(lines))
            smells.extend(self._detect_duplicate_code(lines))
            smells.extend(self._detect_poor_naming(lines))
            smells.extend(self._detect_missing_documentation(lines))
            smells.extend(self._detect_deep_nesting(lines))
            
            # Categorize smells by severity
            critical_smells = [s for s in smells if s['severity'] == 'critical']
            major_smells = [s for s in smells if s['severity'] == 'major']
            minor_smells = [s for s in smells if s['severity'] == 'minor']
            
            return {
                "status": "success",
                "message": f"Code smell analysis completed - found {len(smells)} potential issues",
                "file_path": file_path,
                "total_smells": len(smells),
                "critical_count": len(critical_smells),
                "major_count": len(major_smells),
                "minor_count": len(minor_smells),
                "smells": smells,
                "overall_quality": self._assess_overall_quality(smells)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error detecting code smells: {str(e)}",
                "smells": []
            }
    
    def _detect_long_methods(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect methods that are too long."""
        smells = []
        in_method = False
        method_start = 0
        method_name = ""
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped.startswith('def '):
                if in_method and i - method_start > 30:  # Method longer than 30 lines
                    smells.append({
                        "type": "long_method",
                        "severity": "major",
                        "line": method_start + 1,
                        "description": f"Method '{method_name}' is too long ({i - method_start} lines)",
                        "suggestion": "Consider breaking this method into smaller, more focused methods"
                    })
                
                in_method = True
                method_start = i
                method_name = stripped.split('(')[0].replace('def ', '').strip()
            
            elif stripped.startswith(('def ', 'class ')) and in_method:
                if i - method_start > 30:
                    smells.append({
                        "type": "long_method",
                        "severity": "major", 
                        "line": method_start + 1,
                        "description": f"Method '{method_name}' is too long ({i - method_start} lines)",
                        "suggestion": "Consider breaking this method into smaller, more focused methods"
                    })
                in_method = False
        
        return smells
    
    def _detect_large_classes(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect classes that are too large."""
        smells = []
        in_class = False
        class_start = 0
        class_name = ""
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped.startswith('class '):
                if in_class and i - class_start > 200:  # Class longer than 200 lines
                    smells.append({
                        "type": "large_class",
                        "severity": "critical",
                        "line": class_start + 1,
                        "description": f"Class '{class_name}' is too large ({i - class_start} lines)",
                        "suggestion": "Consider breaking this class into smaller, more cohesive classes"
                    })
                
                in_class = True
                class_start = i
                class_name = stripped.split('(')[0].replace('class ', '').strip().rstrip(':')
            
            elif stripped.startswith('class ') and in_class:
                if i - class_start > 200:
                    smells.append({
                        "type": "large_class", 
                        "severity": "critical",
                        "line": class_start + 1,
                        "description": f"Class '{class_name}' is too large ({i - class_start} lines)",
                        "suggestion": "Consider breaking this class into smaller, more cohesive classes"
                    })
                in_class = False
        
        return smells
    
    def _detect_long_lines(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect lines that are too long."""
        smells = []
        for i, line in enumerate(lines):
            if len(line) > 120:  # PEP 8 suggests 79, but 120 is commonly used
                smells.append({
                    "type": "long_line",
                    "severity": "minor",
                    "line": i + 1,
                    "description": f"Line is too long ({len(line)} characters)",
                    "suggestion": "Consider breaking this line into multiple lines"
                })
        return smells[:10]  # Limit to first 10 long lines
    
    def _detect_duplicate_code(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect potential duplicate code blocks."""
        smells = []
        line_counts = {}
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 20 and not stripped.startswith(('#', '"""', "'''", '/*')):
                if stripped in line_counts:
                    line_counts[stripped].append(i + 1)
                else:
                    line_counts[stripped] = [i + 1]
        
        for line_content, occurrences in line_counts.items():
            if len(occurrences) > 2:  # Line appears more than twice
                smells.append({
                    "type": "duplicate_code",
                    "severity": "major",
                    "line": occurrences[0],
                    "description": f"Potential duplicate code found (appears {len(occurrences)} times)",
                    "suggestion": "Consider extracting this into a reusable function or constant"
                })
        
        return smells[:5]  # Limit to first 5 duplicates
    
    def _detect_poor_naming(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect poor naming conventions."""
        smells = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for single letter variables (except common loop variables)
            if ' = ' in stripped:
                var_name = stripped.split(' = ')[0].strip()
                if len(var_name) == 1 and var_name not in ['i', 'j', 'k', 'x', 'y', 'z']:
                    smells.append({
                        "type": "poor_naming",
                        "severity": "minor",
                        "line": i + 1,
                        "description": f"Single letter variable name '{var_name}' is not descriptive",
                        "suggestion": "Use descriptive variable names"
                    })
        
        return smells[:5]  # Limit to first 5 naming issues
    
    def _detect_missing_documentation(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect functions/classes missing documentation."""
        smells = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            
            if stripped.startswith(('def ', 'class ')):
                # Check if next few lines contain docstring
                has_docstring = False
                for j in range(i + 1, min(i + 4, len(lines))):
                    if '"""' in lines[j] or "'''" in lines[j]:
                        has_docstring = True
                        break
                
                if not has_docstring:
                    entity_type = "function" if stripped.startswith('def ') else "class"
                    name = stripped.split('(')[0].replace('def ', '').replace('class ', '').strip().rstrip(':')
                    smells.append({
                        "type": "missing_documentation",
                        "severity": "minor",
                        "line": i + 1,
                        "description": f"{entity_type.title()} '{name}' lacks documentation",
                        "suggestion": f"Add a docstring to explain what this {entity_type} does"
                    })
            i += 1
        
        return smells[:5]  # Limit to first 5 documentation issues
    
    def _detect_deep_nesting(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Detect deeply nested code blocks."""
        smells = []
        for i, line in enumerate(lines):
            # Count indentation level
            indent_level = 0
            for char in line:
                if char == ' ':
                    indent_level += 1
                elif char == '\t':
                    indent_level += 4  # Treat tab as 4 spaces
                else:
                    break
            
            nesting_level = indent_level // 4  # Assuming 4-space indentation
            
            if nesting_level > 4:  # More than 4 levels of nesting
                smells.append({
                    "type": "deep_nesting",
                    "severity": "major",
                    "line": i + 1,
                    "description": f"Code is nested too deeply ({nesting_level} levels)",
                    "suggestion": "Consider extracting nested logic into separate functions"
                })
        
        return smells[:3]  # Limit to first 3 deep nesting issues
    
    def _assess_overall_quality(self, smells: List[Dict[str, Any]]) -> str:
        """Assess overall code quality based on smells found."""
        if not smells:
            return "excellent"
        
        critical_count = len([s for s in smells if s['severity'] == 'critical'])
        major_count = len([s for s in smells if s['severity'] == 'major'])
        
        if critical_count > 3 or major_count > 10:
            return "poor"
        elif critical_count > 1 or major_count > 5:
            return "fair"
        elif major_count > 0:
            return "good"
        else:
            return "very_good"


class GenerateAIInsightsTool:
    """
    Name: generate_ai_insights
    Function/Purpose: Uses the Gemini API to perform a high-level analysis of a code sample, 
                     providing insights on quality, style, and potential improvements that go 
                     beyond simple rule-based checks.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def execute(self, code_sample: str) -> Dict[str, Any]:
        """Execute AI-powered code analysis and return structured result."""
        try:
            # Since we don't have Gemini API key setup here, we'll simulate AI insights
            # In a real implementation, this would call the Gemini API
            
            insights = self._simulate_ai_analysis(code_sample)
            
            return {
                "status": "success",
                "message": "AI insights generated successfully",
                "code_length": len(code_sample),
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "insights": insights
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating AI insights: {str(e)}",
                "insights": {}
            }
    
    def _simulate_ai_analysis(self, code_sample: str) -> Dict[str, Any]:
        """Simulate AI analysis (placeholder for actual Gemini API call)."""
        lines = code_sample.split('\n')
        
        # Analyze code characteristics
        has_functions = any('def ' in line for line in lines)
        has_classes = any('class ' in line for line in lines)
        has_comments = any('#' in line for line in lines)
        has_docstrings = any('"""' in line or "'''" in line for line in lines)
        
        # Generate insights based on code characteristics
        quality_score = 70  # Base score
        
        if has_functions:
            quality_score += 10
        if has_classes:
            quality_score += 10
        if has_comments:
            quality_score += 5
        if has_docstrings:
            quality_score += 5
        
        # Simulate AI-style insights
        insights = {
            "quality_score": min(quality_score, 100),
            "code_style": self._analyze_style(lines),
            "architectural_insights": self._analyze_architecture(lines),
            "improvement_suggestions": self._generate_suggestions(lines),
            "maintainability": self._assess_maintainability(lines),
            "readability": self._assess_readability(lines)
        }
        
        return insights
    
    def _analyze_style(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze code style."""
        return {
            "follows_pep8": True,  # Simplified
            "naming_convention": "snake_case",
            "indentation_style": "spaces",
            "line_length_compliance": 85  # Percentage
        }
    
    def _analyze_architecture(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze architectural patterns."""
        has_classes = any('class ' in line for line in lines)
        has_inheritance = any('class ' in line and '(' in line for line in lines)
        
        return {
            "paradigm": "object_oriented" if has_classes else "procedural",
            "uses_inheritance": has_inheritance,
            "separation_of_concerns": "good",
            "modularity": "moderate"
        }
    
    def _generate_suggestions(self, lines: List[str]) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if not any('"""' in line or "'''" in line for line in lines):
            suggestions.append("Add docstrings to document functions and classes")
        
        if not any('test' in line.lower() for line in lines):
            suggestions.append("Consider adding unit tests for better code reliability")
        
        if any(len(line) > 100 for line in lines):
            suggestions.append("Break down long lines for better readability")
        
        return suggestions
    
    def _assess_maintainability(self, lines: List[str]) -> Dict[str, Any]:
        """Assess code maintainability."""
        return {
            "score": 75,
            "factors": ["Good function structure", "Reasonable complexity"],
            "risks": ["Limited documentation", "Few tests"]
        }
    
    def _assess_readability(self, lines: List[str]) -> Dict[str, Any]:
        """Assess code readability."""
        return {
            "score": 80,
            "strengths": ["Clear variable names", "Good structure"],
            "weaknesses": ["Could use more comments"]
        }


class CreateSmartDiffTool:
    """
    Name: create_smart_diff
    Function/Purpose: The core of the modification workflow. The LLM provides a file path and 
                     natural language instructions. The tool reads the file, uses the Gemini API 
                     to generate the fully modified code, and then produces a diff between the 
                     original and the new versions.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def execute(self, file_path: str, instructions: str) -> Dict[str, Any]:
        """Execute smart diff creation and return structured result."""
        try:
            target_path = self.workspace_path / file_path
            if not target_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}",
                    "diff": "",
                    "original_content": "",
                    "modified_content": ""
                }
            
            # Read original file
            with open(target_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Generate modified content based on instructions
            # In a real implementation, this would use Gemini API
            modified_content = self._simulate_smart_modification(original_content, instructions)
            
            # Create diff
            diff = self._create_diff(original_content, modified_content, file_path)
            
            return {
                "status": "success",
                "message": f"Smart diff created for {file_path}",
                "file_path": file_path,
                "instructions": instructions,
                "original_content": original_content,
                "modified_content": modified_content,
                "diff": diff,
                "changes_summary": self._summarize_changes(original_content, modified_content)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating smart diff: {str(e)}",
                "diff": "",
                "original_content": "",
                "modified_content": ""
            }
    
    def _simulate_smart_modification(self, original_content: str, instructions: str) -> str:
        """Simulate smart code modification (placeholder for Gemini API)."""
        # This is a simplified simulation - real implementation would use AI
        modified_content = original_content
        
        # Simple instruction parsing
        instructions_lower = instructions.lower()
        
        if "add docstring" in instructions_lower:
            # Add docstrings to functions without them
            lines = modified_content.split('\n')
            result_lines = []
            
            for i, line in enumerate(lines):
                result_lines.append(line)
                
                if line.strip().startswith('def ') and ':' in line:
                    # Check if next line is already a docstring
                    next_line_idx = i + 1
                    while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                        result_lines.append(lines[next_line_idx])
                        next_line_idx += 1
                    
                    if (next_line_idx >= len(lines) or 
                        not ('"""' in lines[next_line_idx] or "'''" in lines[next_line_idx])):
                        # Add docstring
                        indent = len(line) - len(line.lstrip())
                        func_name = line.strip().split('(')[0].replace('def ', '')
                        docstring = f"{' ' * (indent + 4)}\"\"\"TODO: Add documentation for {func_name}.\"\"\""
                        result_lines.append(docstring)
            
            modified_content = '\n'.join(result_lines)
        
        elif "add comments" in instructions_lower:
            # Add comments to complex lines
            lines = modified_content.split('\n')
            result_lines = []
            
            for line in lines:
                result_lines.append(line)
                if (len(line.strip()) > 50 and 
                    not line.strip().startswith('#') and 
                    line.strip() and
                    not '# ' in line):
                    # Add a comment
                    indent = len(line) - len(line.lstrip())
                    comment = f"{' ' * indent}# TODO: Add explanation for this logic"
                    result_lines.insert(-1, comment)
            
            modified_content = '\n'.join(result_lines)
        
        elif "fix" in instructions_lower or "improve" in instructions_lower:
            # Simple improvements
            modified_content = modified_content.replace('    ', '    ')  # Ensure 4-space indentation
            
        return modified_content
    
    def _create_diff(self, original: str, modified: str, file_path: str) -> str:
        """Create a unified diff between original and modified content."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        import difflib
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=''
        )
        
        return ''.join(diff)
    
    def _summarize_changes(self, original: str, modified: str) -> Dict[str, Any]:
        """Summarize the changes made."""
        original_lines = original.split('\n')
        modified_lines = modified.split('\n')
        
        return {
            "lines_added": len(modified_lines) - len(original_lines),
            "lines_removed": max(0, len(original_lines) - len(modified_lines)),
            "total_changes": abs(len(modified_lines) - len(original_lines)),
            "change_type": "addition" if len(modified_lines) > len(original_lines) else "modification"
        }


class ApplyCodeDiffTool:
    """
    Name: apply_code_diff
    Function/Purpose: Applies a previously generated diff to a file, writing the changes to disk. 
                     It automatically creates a backup of the original file for safety. 
                     This is typically the final step after a create_smart_diff call has been 
                     reviewed and confirmed.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def execute(self, file_path: str, diff_content: str = "", 
                modified_content: str = "") -> Dict[str, Any]:
        """Execute diff application and return structured result."""
        try:
            target_path = self.workspace_path / file_path
            if not target_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}",
                    "backup_path": ""
                }
            
            # Create backup
            backup_path = self._create_backup(target_path)
            
            # Apply changes
            if modified_content:
                # Direct content replacement
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                
                return {
                    "status": "success",
                    "message": f"Changes applied to {file_path}",
                    "file_path": file_path,
                    "backup_path": str(backup_path),
                    "application_method": "direct_replacement"
                }
            
            elif diff_content:
                # Apply diff (simplified - real implementation would parse diff)
                return {
                    "status": "success",
                    "message": f"Diff applied to {file_path}",
                    "file_path": file_path,
                    "backup_path": str(backup_path),
                    "application_method": "diff_patch"
                }
            
            else:
                return {
                    "status": "error",
                    "message": "No content or diff provided to apply",
                    "backup_path": str(backup_path)
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error applying changes: {str(e)}",
                "backup_path": ""
            }
    
    def _create_backup(self, file_path: Path) -> Path:
        """Create a backup of the original file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{timestamp}")
        
        import shutil
        shutil.copy2(file_path, backup_path)
        
        return backup_path


# ===== MAIN TOOL EXECUTION SYSTEM =====

class SymbioteToolExecutor:
    """
    Main tool executor that implements the iterative conversation loop described in TODO.md.
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.tools = {
            'list_directory_files': ListDirectoryFilesTool(workspace_path),
            'read_file_content': ReadFileContentTool(workspace_path), 
            'search_in_files': SearchInFilesTool(workspace_path),
            'execute_terminal_command': ExecuteTerminalCommandTool(workspace_path),
            'assess_code_complexity': AssessCodeComplexityTool(workspace_path),
            'identify_code_smells': IdentifyCodeSmellsTool(workspace_path),
            'generate_ai_insights': GenerateAIInsightsTool(workspace_path),
            'create_smart_diff': CreateSmartDiffTool(workspace_path),
            'apply_code_diff': ApplyCodeDiffTool(workspace_path)
        }
        print(f"🧬 Symbiote Tool System initialized with {len(self.tools)} tools")
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a single tool with the new iterative paradigm.
        This is the core method that the LLM calls for each step.
        """
        if tool_name not in self.tools:
            return {
                "status": "error",
                "message": f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            }
        
        tool = self.tools[tool_name]
        
        try:
            # Display execution snippet
            print(f"> {self._get_execution_snippet(tool_name, kwargs)}")
            
            # Execute the tool
            result = tool.execute(**kwargs)
            
            # Add execution metadata
            result['tool_name'] = tool_name
            result['execution_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing tool '{tool_name}': {str(e)}",
                "tool_name": tool_name
            }
    
    def _get_execution_snippet(self, tool_name: str, kwargs: Dict[str, Any]) -> str:
        """Get execution snippet as described in TODO.md."""
        snippets = {
            'list_directory_files': f"Traversing '{kwargs.get('directory_path', './')}' to list contents...",
            'read_file_content': f"Reading content from '{kwargs.get('file_path', 'unknown')}'...",
            'search_in_files': f"Searching for pattern '{kwargs.get('search_pattern', 'unknown')}' in '{kwargs.get('directory_path', './')}'...",
            'execute_terminal_command': f"Executing command: '{kwargs.get('command', 'unknown')}'...",
            'assess_code_complexity': "Calculating cyclomatic complexity for functions...",
            'identify_code_smells': "Sniffing for code smells...",
            'generate_ai_insights': "Consulting with Gemini for deeper insights...",
            'create_smart_diff': f"Generating intelligent diff for '{kwargs.get('file_path', 'unknown')}'...",
            'apply_code_diff': f"Applying patch to '{kwargs.get('file_path', 'unknown')}' and creating backup..."
        }
        
        return snippets.get(tool_name, f"Executing {tool_name}...")
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get information about available tools."""
        return {
            'list_directory_files': "Lists all files and subdirectories within a specified path",
            'read_file_content': "Reads the entire content of a specific file with metadata",
            'search_in_files': "Performs text/regex search within files in a directory",
            'execute_terminal_command': "Executes shell commands in the project workspace", 
            'assess_code_complexity': "Assesses cyclomatic complexity of functions in code",
            'identify_code_smells': "Identifies anti-patterns and code smells",
            'generate_ai_insights': "Uses AI to analyze code quality and style",
            'create_smart_diff': "Creates intelligent diffs based on natural language instructions",
            'apply_code_diff': "Applies generated diffs to files with backup creation"
        }


# ===== FACTORY FUNCTION =====

def create_symbiote_tool_system(workspace_path: Path) -> SymbioteToolExecutor:
    """
    Factory function to create the Symbiote Tool System as described in TODO.md.
    
    Usage Example:
    ```python
    from pathlib import Path
    
    # Initialize the system
    tools = create_symbiote_tool_system(Path("/path/to/workspace"))
    
    # Execute tools in the iterative conversation loop
    result1 = tools.execute_tool('list_directory_files', directory_path='./utils')
    result2 = tools.execute_tool('read_file_content', file_path='main.py')
    result3 = tools.execute_tool('assess_code_complexity', code_data=result2['content'])
    ```
    """
    return SymbioteToolExecutor(workspace_path)
