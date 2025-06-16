"""
Python AST-based code analysis tool for deep code structure analysis.
Implements advanced code understanding and pattern extraction.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..tools.base_tool import BaseTool, ToolResult, ToolInfo, ToolCategory


@dataclass
class CodeStructure:
    """Represents the structure of analyzed code."""
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    imports: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    complexity_metrics: Dict[str, Any]
    patterns: List[Dict[str, Any]]


class ASTAnalyzer:
    """AST-based code analyzer for deep structure analysis."""
    
    def __init__(self):
        self.current_file = None
        self.complexity_score = 0
        self.patterns_found = []
    
    def analyze_code(self, code: str, file_path: Optional[str] = None) -> CodeStructure:
        """Analyze Python code using AST."""
        try:
            self.current_file = file_path
            self.complexity_score = 0
            self.patterns_found = []
            
            # Parse the code
            tree = ast.parse(code)
            
            # Extract structures
            classes = self._extract_classes(tree)
            functions = self._extract_functions(tree)
            imports = self._extract_imports(tree)
            variables = self._extract_variables(tree)
            
            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity(tree)
            
            # Detect patterns
            patterns = self._detect_patterns(tree, code)
            
            return CodeStructure(
                classes=classes,
                functions=functions,
                imports=imports,
                variables=variables,
                complexity_metrics=complexity_metrics,
                patterns=patterns
            )
            
        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {e}")
        except Exception as e:
            raise ValueError(f"Error analyzing code: {e}")
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class definitions and their details."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "bases": [self._get_name(base) for base in node.bases],
                    "methods": [],
                    "properties": [],
                    "decorators": [self._get_name(dec) for dec in node.decorator_list],
                    "docstring": ast.get_docstring(node)
                }
                
                # Extract methods and properties
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            "name": item.name,
                            "line_number": item.lineno,
                            "args": [arg.arg for arg in item.args.args],
                            "decorators": [self._get_name(dec) for dec in item.decorator_list],
                            "is_property": any("property" in self._get_name(dec) for dec in item.decorator_list),
                            "is_static": any("staticmethod" in self._get_name(dec) for dec in item.decorator_list),
                            "is_class_method": any("classmethod" in self._get_name(dec) for dec in item.decorator_list),
                            "docstring": ast.get_docstring(item)
                        }
                        
                        if method_info["is_property"]:
                            class_info["properties"].append(method_info)
                        else:
                            class_info["methods"].append(method_info)
                
                classes.append(class_info)
        
        return classes
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function definitions."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip methods (already handled in classes)
                if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                       if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                    continue
                
                function_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "defaults": len(node.args.defaults),
                    "decorators": [self._get_name(dec) for dec in node.decorator_list],
                    "returns": self._get_name(node.returns) if node.returns else None,
                    "docstring": ast.get_docstring(node),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "complexity": self._calculate_function_complexity(node)
                }
                
                functions.append(function_info)
        
        return functions
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract import statements."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line_number": node.lineno
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "from_import",
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line_number": node.lineno,
                        "level": node.level
                    })
        
        return imports
    
    def _extract_variables(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract variable assignments."""
        variables = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_info = {
                            "name": target.id,
                            "line_number": node.lineno,
                            "type": self._infer_type(node.value),
                            "scope": "global"  # Simplified
                        }
                        variables.append(var_info)
        
        return variables
    
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate complexity metrics."""
        complexity = {
            "cyclomatic_complexity": 1,  # Base complexity
            "lines_of_code": 0,
            "number_of_functions": 0,
            "number_of_classes": 0,
            "max_nesting_depth": 0,
            "maintainability_index": 0
        }
        
        # Count control flow statements for cyclomatic complexity
        control_flow_nodes = (
            ast.If, ast.While, ast.For, ast.AsyncFor,
            ast.Try, ast.With, ast.AsyncWith
        )
        
        for node in ast.walk(tree):
            if isinstance(node, control_flow_nodes):
                complexity["cyclomatic_complexity"] += 1
            elif isinstance(node, ast.FunctionDef):
                complexity["number_of_functions"] += 1
            elif isinstance(node, ast.ClassDef):
                complexity["number_of_classes"] += 1
        
        # Calculate lines of code (approximate)
        if hasattr(tree, 'body') and getattr(tree, 'body', None):
            last_node = tree.body[-1]  # type: ignore
            if hasattr(last_node, 'lineno'):
                complexity["lines_of_code"] = last_node.lineno
        
        # Calculate maintainability index (simplified)
        loc = complexity["lines_of_code"]
        cc = complexity["cyclomatic_complexity"]
        if loc > 0:
            mi = 171 - 5.2 * (cc / loc) - 0.23 * cc - 16.2 * (loc / 100)
            complexity["maintainability_index"] = max(0, int(mi))
        
        return complexity
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate complexity for a specific function."""
        complexity = 1  # Base complexity
        
        control_flow_nodes = (
            ast.If, ast.While, ast.For, ast.AsyncFor,
            ast.Try, ast.With, ast.AsyncWith
        )
        
        for child in ast.walk(node):
            if isinstance(child, control_flow_nodes):
                complexity += 1
        
        return complexity
    
    def _detect_patterns(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Detect common code patterns."""
        patterns = []
        
        # Design pattern detection
        patterns.extend(self._detect_design_patterns(tree))
        
        # Code smell detection
        patterns.extend(self._detect_code_smells(tree, code))
        
        # Best practice patterns
        patterns.extend(self._detect_best_practices(tree))
        
        return patterns
    
    def _detect_design_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect design patterns in the code."""
        patterns = []
        
        # Singleton pattern detection
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for __new__ method that suggests singleton
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__new__":
                        patterns.append({
                            "type": "design_pattern",
                            "pattern": "singleton",
                            "description": f"Possible Singleton pattern in class {node.name}",
                            "line_number": node.lineno,
                            "confidence": 0.7
                        })
        
        # Factory pattern detection
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if "create" in node.name.lower() or "factory" in node.name.lower():
                    patterns.append({
                        "type": "design_pattern",
                        "pattern": "factory",
                        "description": f"Possible Factory pattern in function {node.name}",
                        "line_number": node.lineno,
                        "confidence": 0.6
                    })
        
        return patterns
    
    def _detect_code_smells(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Detect code smells."""
        patterns = []
        
        # Long function detection
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_lines = self._count_function_lines(node, code)
                if function_lines > 50:
                    patterns.append({
                        "type": "code_smell",
                        "pattern": "long_function",
                        "description": f"Function {node.name} is too long ({function_lines} lines)",
                        "line_number": node.lineno,
                        "severity": "medium"
                    })
        
        # Too many parameters
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                if param_count > 5:
                    patterns.append({
                        "type": "code_smell",
                        "pattern": "too_many_parameters",
                        "description": f"Function {node.name} has too many parameters ({param_count})",
                        "line_number": node.lineno,
                        "severity": "medium"
                    })
        
        return patterns
    
    def _detect_best_practices(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect adherence to best practices."""
        patterns = []
        
        # Type hints usage
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_type_hints = any(arg.annotation for arg in node.args.args)
                if has_type_hints:
                    patterns.append({
                        "type": "best_practice",
                        "pattern": "type_hints",
                        "description": f"Function {node.name} uses type hints",
                        "line_number": node.lineno,
                        "positive": True
                    })
        
        # Docstring usage
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if ast.get_docstring(node):
                    patterns.append({
                        "type": "best_practice",
                        "pattern": "documentation",
                        "description": f"{type(node).__name__} {node.name} has documentation",
                        "line_number": node.lineno,
                        "positive": True
                    })
        
        return patterns
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(type(node).__name__)
    
    def _infer_type(self, node: ast.AST) -> str:
        """Infer type from assignment value."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        else:
            return "unknown"
    
    def _count_function_lines(self, node: ast.FunctionDef, code: str) -> int:
        """Count lines in a function."""
        lines = code.split('\n')
        start_line = node.lineno - 1
        
        # Find the end of the function (simplified)
        indent_level = None
        end_line = len(lines)
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                if indent_level is None:
                    indent_level = current_indent
                elif current_indent <= indent_level and not line.strip().startswith(('"""', "'''")):
                    end_line = i
                    break
        
        return end_line - start_line


class CodeAnalysisTool(BaseTool):
    """Tool for deep code analysis using Python AST."""
    
    def __init__(self):
        super().__init__(
            name="code_analysis",
            description="Deep code analysis using Python AST",
            category=ToolCategory.CODE_ANALYSIS
        )
        self.analyzer = ASTAnalyzer()
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute code analysis operation."""
        if not self.validate_parameters(parameters):
            return self.create_error_result("Invalid parameters")
        
        operation = parameters.get("operation", "analyze")
        
        try:
            if operation == "analyze":
                return await self._analyze_code(parameters)
            elif operation == "analyze_file":
                return await self._analyze_file(parameters)
            elif operation == "get_metrics":
                return await self._get_metrics(parameters)
            elif operation == "detect_patterns":
                return await self._detect_patterns(parameters)
            else:
                return self.create_error_result(f"Unknown operation: {operation}")
                
        except Exception as e:
            return self.create_error_result(f"Analysis failed: {str(e)}")
    
    async def _analyze_code(self, parameters: Dict[str, Any]) -> ToolResult:
        """Analyze code from string."""
        code = parameters.get("code", "")
        if not code:
            return self.create_error_result("No code provided")
        
        try:
            structure = self.analyzer.analyze_code(code)
            
            return ToolResult(
                success=True,
                data={
                    "analysis": {
                        "classes": structure.classes,
                        "functions": structure.functions,
                        "imports": structure.imports,
                        "variables": structure.variables,
                        "complexity": structure.complexity_metrics,
                        "patterns": structure.patterns
                    },
                    "summary": {
                        "total_classes": len(structure.classes),
                        "total_functions": len(structure.functions),
                        "total_imports": len(structure.imports),
                        "complexity_score": structure.complexity_metrics.get("cyclomatic_complexity", 0),
                        "maintainability_index": structure.complexity_metrics.get("maintainability_index", 0)
                    }
                }
            )
            
        except Exception as e:
            return self.create_error_result(f"Code analysis failed: {str(e)}")
    
    async def _analyze_file(self, parameters: Dict[str, Any]) -> ToolResult:
        """Analyze code from file."""
        file_path = parameters.get("file_path", "")
        if not file_path:
            return self.create_error_result("No file path provided")
        
        try:
            path = Path(file_path)
            if not path.exists():
                return self.create_error_result(f"File not found: {file_path}")
            
            if not path.suffix == ".py":
                return self.create_error_result("Only Python files are supported")
            
            code = path.read_text(encoding="utf-8")
            structure = self.analyzer.analyze_code(code, str(path))
            
            return ToolResult(
                success=True,
                data={
                    "file_path": str(path),
                    "analysis": {
                        "classes": structure.classes,
                        "functions": structure.functions,
                        "imports": structure.imports,
                        "variables": structure.variables,
                        "complexity": structure.complexity_metrics,
                        "patterns": structure.patterns
                    }
                }
            )
            
        except Exception as e:
            return self.create_error_result(f"File analysis failed: {str(e)}")
    
    async def _get_metrics(self, parameters: Dict[str, Any]) -> ToolResult:
        """Get complexity metrics for code."""
        code = parameters.get("code", "")
        if not code:
            return self.create_error_result("No code provided")
        
        try:
            structure = self.analyzer.analyze_code(code)
            
            return ToolResult(
                success=True,
                data={
                    "metrics": structure.complexity_metrics,
                    "recommendations": self._generate_recommendations(structure)
                }
            )
            
        except Exception as e:
            return self.create_error_result(f"Metrics calculation failed: {str(e)}")
    
    async def _detect_patterns(self, parameters: Dict[str, Any]) -> ToolResult:
        """Detect patterns in code."""
        code = parameters.get("code", "")
        if not code:
            return self.create_error_result("No code provided")
        
        try:
            structure = self.analyzer.analyze_code(code)
            
            return ToolResult(
                success=True,
                data={
                    "patterns": structure.patterns,
                    "pattern_summary": self._summarize_patterns(structure.patterns)
                }
            )
            
        except Exception as e:
            return self.create_error_result(f"Pattern detection failed: {str(e)}")
    
    def _generate_recommendations(self, structure: CodeStructure) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        complexity = structure.complexity_metrics
        cc = complexity.get("cyclomatic_complexity", 0)
        mi = complexity.get("maintainability_index", 0)
        
        if cc > 10:
            recommendations.append("Consider breaking down complex functions to reduce cyclomatic complexity")
        
        if mi < 20:
            recommendations.append("Low maintainability index - consider refactoring for better readability")
        
        # Check for code smells
        for pattern in structure.patterns:
            if pattern.get("type") == "code_smell":
                recommendations.append(f"Address code smell: {pattern.get('description')}")
        
        return recommendations
    
    def _summarize_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize detected patterns."""
        summary = {
            "total_patterns": len(patterns),
            "by_type": defaultdict(int),
            "code_smells": 0,
            "best_practices": 0,
            "design_patterns": 0
        }
        
        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")
            summary["by_type"][pattern_type] += 1
            
            if pattern_type == "code_smell":
                summary["code_smells"] += 1
            elif pattern_type == "best_practice":
                summary["best_practices"] += 1
            elif pattern_type == "design_pattern":
                summary["design_patterns"] += 1
        
        return dict(summary)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        required_params = {
            "analyze": ["code"],
            "analyze_file": ["file_path"],
            "get_metrics": ["code"],
            "detect_patterns": ["code"]
        }
        
        operation = parameters.get("operation", "analyze")
        if operation not in required_params:
            return False
        
        for param in required_params[operation]:
            if param not in parameters or not parameters[param]:
                return False
        
        return True
    
    def get_info(self) -> ToolInfo:
        """Get tool information."""
        return ToolInfo(
            name="code_analysis",
            description="Deep code analysis using Python AST. Analyzes structure, complexity, patterns, and code quality.",
            category=ToolCategory.CODE_ANALYSIS,
            parameters={
                "operation": "Operation to perform: analyze, analyze_file, get_metrics, detect_patterns",
                "code": "Python code to analyze (for analyze, get_metrics, detect_patterns)",
                "file_path": "Path to Python file to analyze (for analyze_file)"
            },
            examples={
                "analyze": "Analyze code structure and complexity",
                "detect_patterns": "Detect design patterns and code smells",
                "get_metrics": "Generate maintainability metrics",
                "analyze_file": "Analyze file for code quality"
            }
        )
