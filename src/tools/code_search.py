"""
Code search and analysis tool for Symbiote.
"""
import os
import ast
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool, ToolResult, ToolInfo, ToolCategory


class CodeSearchTool(BaseTool):
    """Tool for searching and analyzing code in the workspace."""
    
    def __init__(self):
        super().__init__(
            name="code_search",
            description="Search for code patterns, functions, classes, and analyze code structure",
            category=ToolCategory.CODE_ANALYSIS
        )
        self.supported_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php'}
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute code search operation."""
        if not self.validate_parameters(parameters):
            return self.create_error_result("Invalid parameters")
        
        operation = parameters.get("operation")
        
        try:
            if operation == "search_text":
                pattern = parameters.get("pattern", "")
                path = parameters.get("path", ".")
                return await self._search_text(pattern, path)
            elif operation == "find_function":
                function_name = parameters.get("function_name", "")
                path = parameters.get("path", ".")
                return await self._find_function(function_name, path)
            elif operation == "find_class":
                class_name = parameters.get("class_name", "")
                path = parameters.get("path", ".")
                return await self._find_class(class_name, path)
            elif operation == "analyze_file":
                file_path = parameters.get("file_path", "")
                return await self._analyze_file(file_path)
            elif operation == "list_functions":
                file_path = parameters.get("file_path", "")
                return await self._list_functions(file_path)
            elif operation == "count_lines":
                path = parameters.get("path", ".")
                return await self._count_lines(path)
            else:
                return self.create_error_result(f"Unknown operation: {operation}")
                
        except Exception as e:
            return self.create_error_result(f"Code search operation failed: {str(e)}")
    
    async def _search_text(self, pattern: str, search_path: str) -> ToolResult:
        """Search for text pattern in code files."""
        try:
            matches = []
            pattern_re = re.compile(pattern, re.IGNORECASE)
            
            for file_path in self._get_code_files(search_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        
                    for line_num, line in enumerate(lines, 1):
                        if pattern_re.search(line):
                            matches.append({
                                "file": str(file_path),
                                "line_number": line_num,
                                "line_content": line.strip(),
                                "match": pattern
                            })
                except Exception:
                    continue  # Skip files we can't read
            
            return self.create_success_result(
                {"matches": matches, "pattern": pattern, "search_path": search_path},
                {"total_matches": len(matches)}
            )
            
        except Exception as e:
            return self.create_error_result(f"Text search failed: {str(e)}")
    
    async def _find_function(self, function_name: str, search_path: str) -> ToolResult:
        """Find function definitions in Python files."""
        try:
            functions = []
            
            for file_path in self._get_python_files(search_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == function_name:
                            functions.append({
                                "file": str(file_path),
                                "function_name": node.name,
                                "line_number": node.lineno,
                                "args": [arg.arg for arg in node.args.args],
                                "docstring": ast.get_docstring(node)
                            })
                            
                except Exception:
                    continue  # Skip files we can't parse
            
            return self.create_success_result(
                {"functions": functions, "function_name": function_name},
                {"total_found": len(functions)}
            )
            
        except Exception as e:
            return self.create_error_result(f"Function search failed: {str(e)}")
    
    async def _find_class(self, class_name: str, search_path: str) -> ToolResult:
        """Find class definitions in Python files."""
        try:
            classes = []
            
            for file_path in self._get_python_files(search_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name == class_name:
                            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                            classes.append({
                                "file": str(file_path),
                                "class_name": node.name,
                                "line_number": node.lineno,
                                "methods": methods,
                                "docstring": ast.get_docstring(node)
                            })
                            
                except Exception:
                    continue  # Skip files we can't parse
            
            return self.create_success_result(
                {"classes": classes, "class_name": class_name},
                {"total_found": len(classes)}
            )
            
        except Exception as e:
            return self.create_error_result(f"Class search failed: {str(e)}")
    
    async def _analyze_file(self, file_path: str) -> ToolResult:
        """Analyze a Python file structure."""
        try:
            if not os.path.exists(file_path):
                return self.create_error_result(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            analysis = {
                "file_path": file_path,
                "total_lines": len(content.splitlines()),
                "imports": [],
                "functions": [],
                "classes": [],
                "variables": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        analysis["imports"].append(f"{node.module}.{alias.name}")
                elif isinstance(node, ast.FunctionDef):
                    analysis["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis["variables"].append(target.id)
            
            return self.create_success_result(analysis)
            
        except Exception as e:
            return self.create_error_result(f"File analysis failed: {str(e)}")
    
    async def _list_functions(self, file_path: str) -> ToolResult:
        """List all functions in a Python file."""
        try:
            if not os.path.exists(file_path):
                return self.create_error_result(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line_number": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node),
                        "is_async": isinstance(node, ast.AsyncFunctionDef)
                    })
            
            return self.create_success_result(
                {"functions": functions, "file_path": file_path},
                {"total_functions": len(functions)}
            )
            
        except Exception as e:
            return self.create_error_result(f"Function listing failed: {str(e)}")
    
    async def _count_lines(self, search_path: str) -> ToolResult:
        """Count lines of code in the project."""
        try:
            stats = {
                "total_files": 0,
                "total_lines": 0,
                "by_extension": {},
                "files": []
            }
            
            for file_path in self._get_code_files(search_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                    
                    ext = file_path.suffix.lower()
                    stats["total_files"] += 1
                    stats["total_lines"] += lines
                    
                    if ext not in stats["by_extension"]:
                        stats["by_extension"][ext] = {"files": 0, "lines": 0}
                    
                    stats["by_extension"][ext]["files"] += 1
                    stats["by_extension"][ext]["lines"] += lines
                    
                    stats["files"].append({
                        "file": str(file_path),
                        "extension": ext,
                        "lines": lines
                    })
                    
                except Exception:
                    continue  # Skip files we can't read
            
            return self.create_success_result(stats)
            
        except Exception as e:
            return self.create_error_result(f"Line counting failed: {str(e)}")
    
    def _get_code_files(self, search_path: str) -> List[Path]:
        """Get all code files in the given path."""
        files = []
        path = Path(search_path)
        
        if path.is_file() and path.suffix in self.supported_extensions:
            return [path]
        
        if path.is_dir():
            for file_path in path.rglob("*"):
                if (file_path.is_file() and 
                    file_path.suffix in self.supported_extensions and
                    not any(part.startswith('.') for part in file_path.parts)):
                    files.append(file_path)
        
        return files
    
    def _get_python_files(self, search_path: str) -> List[Path]:
        """Get all Python files in the given path."""
        files = []
        path = Path(search_path)
        
        if path.is_file() and path.suffix == '.py':
            return [path]
        
        if path.is_dir():
            for file_path in path.rglob("*.py"):
                if (file_path.is_file() and
                    not any(part.startswith('.') for part in file_path.parts)):
                    files.append(file_path)
        
        return files
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if "operation" not in parameters:
            return False
        
        operation = parameters.get("operation")
        valid_operations = [
            "search_text", "find_function", "find_class", 
            "analyze_file", "list_functions", "count_lines"
        ]
        
        if operation not in valid_operations:
            return False
        
        # Check operation-specific requirements
        if operation == "search_text" and "pattern" not in parameters:
            return False
        if operation == "find_function" and "function_name" not in parameters:
            return False
        if operation == "find_class" and "class_name" not in parameters:
            return False
        if operation in ["analyze_file", "list_functions"] and "file_path" not in parameters:
            return False
        
        return True
    
    def get_info(self) -> ToolInfo:
        """Get tool information."""
        return ToolInfo(
            name=self.name,
            description=self.description,
            category=self.category,
            parameters={
                "operation": {
                    "type": "string",
                    "required": True,
                    "options": ["search_text", "find_function", "find_class", "analyze_file", "list_functions", "count_lines"],
                    "description": "Type of code search operation"
                },
                "pattern": {
                    "type": "string",
                    "required": "for search_text",
                    "description": "Text pattern to search for"
                },
                "function_name": {
                    "type": "string",
                    "required": "for find_function",
                    "description": "Name of function to find"
                },
                "class_name": {
                    "type": "string",
                    "required": "for find_class",
                    "description": "Name of class to find"
                },
                "file_path": {
                    "type": "string",
                    "required": "for analyze_file, list_functions",
                    "description": "Path to specific file to analyze"
                },
                "path": {
                    "type": "string",
                    "required": False,
                    "description": "Search path (default: current directory)"
                }
            },
            examples={
                "search_text": {
                    "operation": "search_text",
                    "pattern": "TODO",
                    "path": "src/"
                },
                "find_function": {
                    "operation": "find_function",
                    "function_name": "main",
                    "path": "."
                },
                "analyze_file": {
                    "operation": "analyze_file",
                    "file_path": "src/main.py"
                }
            }
        )
