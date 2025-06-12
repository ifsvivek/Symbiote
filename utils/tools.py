"""
🧬 Symbiote Tools Module

This module provides a comprehensive set of tools for code analysis, style learning,
and AI-powered insights. These tools are used by the various agents in the Symbiote
ecosystem to perform specialized tasks.

Key Features:
- Code analysis and pattern detection tools
- File system and git operations tools
- AI-powered insight generation tools
- Style preference learning and recommendation tools
- Code quality assessment tools
- Terminal command execution capabilities
- Interactive file operations
- Enhanced autonomous tool execution

Author: Vivek Sharma
License: MIT
"""

import os
import ast
import json
import subprocess
import os
import json
import subprocess
import shlex
import re
import difflib
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import asdict
from collections import Counter, defaultdict
from datetime import datetime
import inspect

from langchain_core.tools import Tool
from google import genai
from google.genai import types

from utils.code_parser import (
    CodeAnalysisResult,
    CodePattern,
    AIInsight,
    Language,
    FunctionSignature,
    ClassStructure,
)


class SymbioteTools:
    """
    Central hub for all Symbiote tools and utilities.
    Provides a comprehensive set of tools for code analysis and AI assistance.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        debug: bool = False,
        workspace_path: Optional[Path] = None,
    ):
        self.debug = debug
        self.gemini_client = None
        self.workspace_path = workspace_path or Path.cwd()

        # Virtual environment detection and setup
        self.venv_path = self._detect_virtual_environment()
        self.venv_activated = False

        if self.debug and self.venv_path:
            print(f"🐍 Virtual environment detected: {self.venv_path}")

        # Initialize Gemini client if available
        if gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=gemini_api_key)
                if self.debug:
                    print("🤖 Gemini client initialized for tools")
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Failed to initialize Gemini client: {e}")

        # Initialize all tools
        self.tools = self._create_tools()

    def _detect_virtual_environment(self) -> Optional[Path]:
        """Detect virtual environment in the workspace."""
        # Common virtual environment directory names
        venv_names = [".venv", "venv", "env", ".env"]

        for venv_name in venv_names:
            venv_path = self.workspace_path / venv_name
            if venv_path.exists() and venv_path.is_dir():
                # Check if it's a valid Python virtual environment
                if (venv_path / "bin" / "python").exists() or (
                    venv_path / "Scripts" / "python.exe"
                ).exists():
                    return venv_path

        # Check if we're already in a virtual environment
        if os.environ.get("VIRTUAL_ENV"):
            return Path(os.environ["VIRTUAL_ENV"])

        return None

    def _get_venv_activation_command(self) -> Optional[str]:
        """Get the command to activate the virtual environment."""
        if not self.venv_path:
            return None

        # Unix/Linux/macOS
        if (self.venv_path / "bin" / "activate").exists():
            return f"source {self.venv_path / 'bin' / 'activate'}"

        # Windows
        if (self.venv_path / "Scripts" / "activate.bat").exists():
            return f"{self.venv_path / 'Scripts' / 'activate.bat'}"

        return None

    def _prepare_command_with_venv(self, command: str) -> str:
        """Prepare command with virtual environment activation if available."""
        if not self.venv_path:
            return command

        activation_cmd = self._get_venv_activation_command()
        if not activation_cmd:
            return command

        # For commands that should use the virtual environment
        python_commands = ["python", "pip", "pytest", "black", "flake8", "mypy"]
        command_parts = command.split()

        if command_parts and any(cmd in command_parts[0] for cmd in python_commands):
            return f"{activation_cmd} && {command}"

        return command

    def _create_tools(self) -> List[Tool]:
        """Create and return all available tools."""
        return [
            # Code Analysis Tools
            Tool(
                name="analyze_naming_conventions",
                description="Analyze naming conventions in code and detect patterns",
                func=self.analyze_naming_conventions,
            ),
            Tool(
                name="evaluate_code_structure",
                description="Evaluate code organization, structure, and architectural patterns",
                func=self.evaluate_code_structure,
            ),
            Tool(
                name="assess_code_complexity",
                description="Assess code complexity metrics and identify potential issues",
                func=self.assess_code_complexity,
            ),
            Tool(
                name="detect_design_patterns",
                description="Detect design patterns and architectural decisions in code",
                func=self.detect_design_patterns,
            ),
            Tool(
                name="analyze_import_patterns",
                description="Analyze import organization and dependency patterns",
                func=self.analyze_import_patterns,
            ),
            # AI-Powered Analysis Tools
            Tool(
                name="generate_ai_insights",
                description="Generate AI-powered insights about code quality and style",
                func=self.generate_ai_insights,
            ),
            Tool(
                name="suggest_improvements",
                description="Suggest specific improvements based on code analysis",
                func=self.suggest_improvements,
            ),
            Tool(
                name="compare_with_best_practices",
                description="Compare code against industry best practices",
                func=self.compare_with_best_practices,
            ),
            # Enhanced File System Tools
            Tool(
                name="read_file_content",
                description="Read and analyze the content of a specific file",
                func=self.read_file_content,
            ),
            Tool(
                name="read_file_with_lines",
                description="Read file content with line numbers for better analysis",
                func=self.read_file_with_lines,
            ),
            Tool(
                name="list_directory_files",
                description="List files in a directory with filtering options",
                func=self.list_directory_files,
            ),
            Tool(
                name="get_file_metadata",
                description="Get metadata information about a file",
                func=self.get_file_metadata,
            ),
            Tool(
                name="search_in_files",
                description="Search for patterns or text within files",
                func=self.search_in_files,
            ),
            # Git and Version Control Tools
            Tool(
                name="get_git_info",
                description="Get git repository information and recent changes",
                func=self.get_git_info,
            ),
            Tool(
                name="analyze_git_history",
                description="Analyze git commit history for patterns",
                func=self.analyze_git_history,
            ),
            # Style Learning Tools
            Tool(
                name="extract_style_preferences",
                description="Extract style preferences from code analysis",
                func=self.extract_style_preferences,
            ),
            Tool(
                name="compare_coding_styles",
                description="Compare different coding styles and patterns",
                func=self.compare_coding_styles,
            ),
            Tool(
                name="generate_style_recommendations",
                description="Generate personalized style recommendations",
                func=self.generate_style_recommendations,
            ),
            Tool(
                name="update_style_preferences",
                description="Update learned style preferences based on new evidence",
                func=self.update_style_preferences,
            ),
            # Quality Assessment Tools
            Tool(
                name="calculate_maintainability_score",
                description="Calculate a maintainability score for code",
                func=self.calculate_maintainability_score,
            ),
            Tool(
                name="identify_code_smells",
                description="Identify potential code smells and anti-patterns",
                func=self.identify_code_smells,
            ),
            Tool(
                name="assess_test_coverage",
                description="Assess test coverage and testing patterns",
                func=self.assess_test_coverage,
            ),
            Tool(
                name="update_style_preferences",
                description="Update learned style preferences based on new evidence",
                func=self.update_style_preferences,
            ),
            # Diff-Based Tools
            Tool(
                name="generate_code_diff",
                description="Generate a unified diff between original and modified code",
                func=self.generate_code_diff,
            ),
            Tool(
                name="apply_code_diff",
                description="Apply a unified diff to modify a file",
                func=self.apply_code_diff,
            ),
            Tool(
                name="create_smart_diff",
                description="Create intelligent diff that preserves context and applies changes safely",
                func=self.create_smart_diff,
            ),
            Tool(
                name="preview_diff_changes",
                description="Preview what changes a diff would make to a file",
                func=self.preview_diff_changes,
            ),
            Tool(
                name="modify_file_with_diff",
                description="Modify a file using AI-generated diff instructions",
                func=self.modify_file_with_diff,
            ),
        ]

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return [tool.name for tool in self.tools]

    def get_tools_description(self) -> str:
        """Get formatted description of all tools."""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)

    # Code Analysis Tools Implementation

    def analyze_naming_conventions(self, code_data: str) -> str:
        """Analyze naming conventions in code and detect patterns."""
        try:
            data = json.loads(code_data) if isinstance(code_data, str) else code_data

            naming_analysis = {
                "function_naming": {},
                "class_naming": {},
                "variable_naming": {},
                "constant_naming": {},
                "consistency_score": 0.0,
                "recommendations": [],
            }

            # Analyze function names
            functions = data.get("functions", [])
            if functions:
                func_names = [f.get("name", "") for f in functions]
                naming_analysis["function_naming"] = self._analyze_naming_pattern(
                    func_names
                )

            # Analyze class names
            classes = data.get("classes", [])
            if classes:
                class_names = [c.get("name", "") for c in classes]
                naming_analysis["class_naming"] = self._analyze_naming_pattern(
                    class_names
                )

            # Calculate consistency score
            all_patterns = []
            for category in ["function_naming", "class_naming"]:
                patterns = naming_analysis[category]
                if patterns:
                    dominant_pattern = max(patterns, key=patterns.get)
                    all_patterns.append(patterns[dominant_pattern])

            if all_patterns:
                naming_analysis["consistency_score"] = sum(all_patterns) / len(
                    all_patterns
                )

            # Generate recommendations
            if naming_analysis["consistency_score"] < 0.8:
                naming_analysis["recommendations"].append(
                    "Consider standardizing naming conventions across the codebase"
                )

            return json.dumps(naming_analysis, indent=2)

        except Exception as e:
            return f"Error analyzing naming conventions: {str(e)}"

    def evaluate_code_structure(self, code_data: str) -> str:
        """Evaluate code organization, structure, and architectural patterns."""
        try:
            data = json.loads(code_data) if isinstance(code_data, str) else code_data

            structure_analysis = {
                "file_organization": {},
                "function_metrics": {},
                "class_metrics": {},
                "import_organization": {},
                "architectural_patterns": [],
                "recommendations": [],
            }

            # Analyze file organization
            total_files = data.get("total_files", 0)
            total_functions = len(data.get("functions", []))
            total_classes = len(data.get("classes", []))

            structure_analysis["file_organization"] = {
                "functions_per_file": total_functions / max(total_files, 1),
                "classes_per_file": total_classes / max(total_files, 1),
                "total_files": total_files,
            }

            # Analyze function metrics
            functions = data.get("functions", [])
            if functions:
                line_counts = [f.get("line_count", 0) for f in functions]
                complexities = [f.get("complexity", 0) for f in functions]

                structure_analysis["function_metrics"] = {
                    "average_lines": sum(line_counts) / len(line_counts),
                    "average_complexity": sum(complexities) / len(complexities),
                    "large_functions": len([lc for lc in line_counts if lc > 50]),
                    "complex_functions": len([c for c in complexities if c > 10]),
                }

            # Analyze class metrics
            classes = data.get("classes", [])
            if classes:
                method_counts = [len(c.get("methods", [])) for c in classes]
                structure_analysis["class_metrics"] = {
                    "average_methods_per_class": sum(method_counts)
                    / len(method_counts),
                    "large_classes": len([mc for mc in method_counts if mc > 20]),
                }

            # Generate recommendations
            func_metrics = structure_analysis["function_metrics"]
            if func_metrics.get("large_functions", 0) > 0:
                structure_analysis["recommendations"].append(
                    f"Consider breaking down {func_metrics['large_functions']} large functions"
                )

            return json.dumps(structure_analysis, indent=2)

        except Exception as e:
            return f"Error evaluating code structure: {str(e)}"

    def assess_code_complexity(self, code_data: str) -> str:
        """Assess code complexity metrics and identify potential issues."""
        try:
            data = json.loads(code_data) if isinstance(code_data, str) else code_data

            complexity_analysis = {
                "overall_complexity": 0.0,
                "complexity_distribution": {"low": 0, "medium": 0, "high": 0},
                "hotspots": [],
                "recommendations": [],
            }

            functions = data.get("functions", [])
            if functions:
                complexities = [f.get("complexity", 1) for f in functions]
                complexity_analysis["overall_complexity"] = sum(complexities) / len(
                    complexities
                )

                # Distribution analysis
                for complexity in complexities:
                    if complexity <= 5:
                        complexity_analysis["complexity_distribution"]["low"] += 1
                    elif complexity <= 10:
                        complexity_analysis["complexity_distribution"]["medium"] += 1
                    else:
                        complexity_analysis["complexity_distribution"]["high"] += 1

                # Identify hotspots (high complexity functions)
                for func in functions:
                    if func.get("complexity", 0) > 10:
                        complexity_analysis["hotspots"].append(
                            {
                                "function": func.get("name", "unknown"),
                                "complexity": func.get("complexity", 0),
                                "file": func.get("file_path", "unknown"),
                            }
                        )

                # Generate recommendations
                high_complexity_count = complexity_analysis["complexity_distribution"][
                    "high"
                ]
                if high_complexity_count > 0:
                    complexity_analysis["recommendations"].append(
                        f"Refactor {high_complexity_count} high-complexity functions to improve maintainability"
                    )

            return json.dumps(complexity_analysis, indent=2)

        except Exception as e:
            return f"Error assessing code complexity: {str(e)}"

    def detect_design_patterns(self, code_data: str) -> str:
        """Detect design patterns and architectural decisions in code."""
        try:
            data = json.loads(code_data) if isinstance(code_data, str) else code_data

            pattern_analysis = {
                "detected_patterns": [],
                "architectural_insights": [],
                "pattern_coverage": 0.0,
                "recommendations": [],
            }

            classes = data.get("classes", [])
            functions = data.get("functions", [])

            # Common pattern detection
            class_names = [c.get("name", "") for c in classes]

            # Factory pattern
            factory_classes = [name for name in class_names if "Factory" in name]
            if factory_classes:
                pattern_analysis["detected_patterns"].append(
                    {
                        "pattern": "Factory Pattern",
                        "instances": factory_classes,
                        "confidence": 0.8,
                    }
                )

            # Observer pattern
            observer_classes = [
                name for name in class_names if "Observer" in name or "Listener" in name
            ]
            if observer_classes:
                pattern_analysis["detected_patterns"].append(
                    {
                        "pattern": "Observer Pattern",
                        "instances": observer_classes,
                        "confidence": 0.7,
                    }
                )

            # Singleton pattern (detect through method analysis)
            singleton_indicators = []
            for cls in classes:
                methods = cls.get("methods", [])
                if "getInstance" in methods or "__new__" in methods:
                    singleton_indicators.append(cls.get("name", ""))

            if singleton_indicators:
                pattern_analysis["detected_patterns"].append(
                    {
                        "pattern": "Singleton Pattern",
                        "instances": singleton_indicators,
                        "confidence": 0.6,
                    }
                )

            # Architectural insights
            if len(classes) > 10:
                pattern_analysis["architectural_insights"].append(
                    "Object-oriented architecture with multiple classes"
                )

            if len(functions) > len(classes) * 3:
                pattern_analysis["architectural_insights"].append(
                    "Function-heavy architecture"
                )

            return json.dumps(pattern_analysis, indent=2)

        except Exception as e:
            return f"Error detecting design patterns: {str(e)}"

    def analyze_import_patterns(self, code_data: str) -> str:
        """Analyze import organization and dependency patterns."""
        try:
            data = json.loads(code_data) if isinstance(code_data, str) else code_data

            import_analysis = {
                "import_statistics": {},
                "dependency_patterns": {},
                "organization_score": 0.0,
                "recommendations": [],
            }

            imports = data.get("imports", [])
            if imports:
                # Basic statistics
                import_analysis["import_statistics"] = {
                    "total_imports": len(imports),
                    "unique_modules": len(
                        set(imp.get("module_name", "") for imp in imports)
                    ),
                    "standard_library": 0,
                    "third_party": 0,
                    "local": 0,
                }

                # Categorize imports
                standard_lib_modules = {
                    "os",
                    "sys",
                    "json",
                    "re",
                    "ast",
                    "pathlib",
                    "collections",
                    "typing",
                    "dataclasses",
                    "enum",
                    "datetime",
                    "pickle",
                }

                for imp in imports:
                    module = imp.get("module_name", "").split(".")[0]
                    if module in standard_lib_modules:
                        import_analysis["import_statistics"]["standard_library"] += 1
                    elif "." in imp.get("module_name", "") and not imp.get(
                        "module_name", ""
                    ).startswith("."):
                        import_analysis["import_statistics"]["third_party"] += 1
                    else:
                        import_analysis["import_statistics"]["local"] += 1

                # Dependency patterns
                module_counter = Counter(imp.get("module_name", "") for imp in imports)
                most_used = module_counter.most_common(5)
                import_analysis["dependency_patterns"]["most_used_modules"] = most_used

                # Organization score (simple heuristic)
                total = len(imports)
                stdlib_ratio = (
                    import_analysis["import_statistics"]["standard_library"] / total
                )
                import_analysis["organization_score"] = min(1.0, stdlib_ratio + 0.5)

            return json.dumps(import_analysis, indent=2)

        except Exception as e:
            return f"Error analyzing import patterns: {str(e)}"

    # AI-Powered Tools Implementation

    def generate_ai_insights(self, code_sample: str) -> str:
        """Generate AI-powered insights about code quality and style."""
        if not self.gemini_client:
            return json.dumps({"error": "Gemini client not available"})

        try:
            prompt = f"""
            Analyze this code sample and provide detailed insights about:
            1. Code quality and maintainability
            2. Style consistency and preferences
            3. Best practices adherence
            4. Potential improvements
            5. Architecture and design decisions
            
            Code sample:
            ```
            {code_sample[:3000]}
            ```
            
            Provide insights in JSON format:
            {{
                "insights": [
                    {{
                        "category": "quality|style|architecture|performance",
                        "type": "strength|weakness|suggestion",
                        "description": "Detailed insight description",
                        "confidence": 0.8,
                        "evidence": ["supporting evidence"],
                        "recommendation": "specific recommendation"
                    }}
                ]
            }}
            """

            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                    max_output_tokens=1500,
                ),
            )

            return response.text or json.dumps({"insights": []})

        except Exception as e:
            return json.dumps({"error": f"AI insight generation failed: {str(e)}"})

    def suggest_improvements(self, analysis_data: str) -> str:
        """Suggest specific improvements based on code analysis."""
        if not self.gemini_client:
            return json.dumps({"suggestions": []})

        try:
            prompt = f"""
            Based on this code analysis data, suggest specific, actionable improvements:
            
            Analysis Data:
            {analysis_data[:2000]}
            
            Provide suggestions in JSON format:
            {{
                "suggestions": [
                    {{
                        "category": "performance|maintainability|readability|security",
                        "priority": "high|medium|low",
                        "title": "Brief title",
                        "description": "Detailed description",
                        "implementation": "How to implement this improvement",
                        "benefits": ["list of benefits"]
                    }}
                ]
            }}
            """

            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                    max_output_tokens=1200,
                ),
            )

            return response.text or json.dumps({"suggestions": []})

        except Exception as e:
            return json.dumps({"error": f"Suggestion generation failed: {str(e)}"})

    def compare_with_best_practices(self, code_data: str) -> str:
        """Compare code against industry best practices."""
        try:
            data = json.loads(code_data) if isinstance(code_data, str) else code_data

            best_practices_analysis = {
                "adherence_score": 0.0,
                "practices_followed": [],
                "practices_violated": [],
                "recommendations": [],
            }

            score_components = []

            # Check naming conventions
            functions = data.get("functions", [])
            if functions:
                func_names = [f.get("name", "") for f in functions]
                snake_case_count = sum(
                    1 for name in func_names if "_" in name and name.islower()
                )
                if snake_case_count / len(func_names) > 0.8:
                    best_practices_analysis["practices_followed"].append(
                        "Consistent snake_case naming for functions"
                    )
                    score_components.append(0.2)
                else:
                    best_practices_analysis["practices_violated"].append(
                        "Inconsistent function naming"
                    )

            # Check function length
            if functions:
                long_functions = [f for f in functions if f.get("line_count", 0) > 50]
                if len(long_functions) == 0:
                    best_practices_analysis["practices_followed"].append(
                        "Functions are appropriately sized"
                    )
                    score_components.append(0.2)
                else:
                    best_practices_analysis["practices_violated"].append(
                        f"{len(long_functions)} functions are too long"
                    )

            # Check documentation
            documented_functions = [f for f in functions if f.get("docstring")]
            if (
                documented_functions
                and len(documented_functions) / len(functions) > 0.7
            ):
                best_practices_analysis["practices_followed"].append(
                    "Good documentation coverage"
                )
                score_components.append(0.2)
            else:
                best_practices_analysis["practices_violated"].append(
                    "Insufficient documentation"
                )

            # Check complexity
            complex_functions = [f for f in functions if f.get("complexity", 0) > 10]
            if len(complex_functions) == 0:
                best_practices_analysis["practices_followed"].append(
                    "Low complexity functions"
                )
                score_components.append(0.2)
            else:
                best_practices_analysis["practices_violated"].append(
                    f"{len(complex_functions)} functions are too complex"
                )

            # Calculate overall score
            best_practices_analysis["adherence_score"] = sum(score_components)

            return json.dumps(best_practices_analysis, indent=2)

        except Exception as e:
            return f"Error comparing with best practices: {str(e)}"

    # File System Tools Implementation

    def read_file_content(self, file_path: str) -> str:
        """Read and analyze the content of a specific file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return json.dumps({"error": f"File not found: {file_path}"})

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            file_info = {
                "path": str(path),
                "size": len(content),
                "lines": len(content.splitlines()),
                "extension": path.suffix,
                "content_preview": (
                    content[:500] + "..." if len(content) > 500 else content
                ),
            }

            return json.dumps(file_info, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error reading file: {str(e)}"})

    def read_file_with_lines(self, file_path: str) -> str:
        """Read file content with line numbers for better analysis."""
        try:
            path = Path(file_path)
            if not path.exists():
                return json.dumps({"error": f"File not found: {file_path}"})

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            file_info = {
                "path": str(path),
                "total_lines": len(lines),
                "content_with_lines": [
                    {"line_number": i + 1, "content": line.strip()}
                    for i, line in enumerate(lines)
                ],
            }

            return json.dumps(file_info, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error reading file with lines: {str(e)}"})

    def list_directory_files(self, directory_path: str, pattern: str = "*") -> str:
        """List files in a directory with filtering options."""
        try:
            path = Path(directory_path)
            if not path.exists():
                return json.dumps({"error": f"Directory not found: {directory_path}"})

            files = list(path.glob(pattern))

            file_list = {
                "directory": str(path),
                "pattern": pattern,
                "total_files": len(files),
                "files": [
                    {
                        "name": f.name,
                        "path": str(f),
                        "is_dir": f.is_dir(),
                        "size": f.stat().st_size if f.is_file() else 0,
                    }
                    for f in files[:50]  # Limit to first 50 files
                ],
            }

            return json.dumps(file_list, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error listing directory: {str(e)}"})

    def get_file_metadata(self, file_path: str) -> str:
        """Get metadata information about a file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return json.dumps({"error": f"File not found: {file_path}"})

            stat = path.stat()
            metadata = {
                "path": str(path),
                "name": path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "extension": path.suffix,
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
            }

            return json.dumps(metadata, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error getting file metadata: {str(e)}"})

    def search_in_files(self, directory_path: str, search_pattern: str) -> str:
        """Search for patterns or text within files."""
        try:
            path = Path(directory_path)
            if not path.exists():
                return json.dumps({"error": f"Directory not found: {directory_path}"})

            matching_files = []
            for file in path.glob("**/*.py"):
                try:
                    with open(file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if re.search(search_pattern, content):
                            matching_files.append(str(file))
                except Exception:
                    continue

            search_results = {
                "directory": str(path),
                "search_pattern": search_pattern,
                "matching_files": matching_files,
                "total_matches": len(matching_files),
            }

            return json.dumps(search_results, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error searching in files: {str(e)}"})

    # Git Tools Implementation

    def get_git_info(self, repository_path: str = ".") -> str:
        """Get git repository information and recent changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return json.dumps(
                    {"error": "Not a git repository or git not available"}
                )

            # Get branch info
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repository_path,
                capture_output=True,
                text=True,
            )

            git_info = {
                "current_branch": branch_result.stdout.strip(),
                "status": result.stdout.strip(),
                "has_changes": bool(result.stdout.strip()),
            }

            return json.dumps(git_info, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error getting git info: {str(e)}"})

    def analyze_git_history(self, repository_path: str = ".", limit: int = 10) -> str:
        """Analyze git commit history for patterns."""
        try:
            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"--max-count={limit}",
                    "--pretty=format:%h|%an|%ad|%s",
                    "--date=short",
                ],
                cwd=repository_path,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return json.dumps({"error": "Failed to get git history"})

            commits = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("|", 3)
                    if len(parts) == 4:
                        commits.append(
                            {
                                "hash": parts[0],
                                "author": parts[1],
                                "date": parts[2],
                                "message": parts[3],
                            }
                        )

            history_analysis = {
                "total_commits": len(commits),
                "recent_commits": commits,
                "authors": list(set(c["author"] for c in commits)),
                "commit_frequency": len(commits) / max(limit, 1),
            }

            return json.dumps(history_analysis, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error analyzing git history: {str(e)}"})

    # Style Learning Tools Implementation

    def extract_style_preferences(self, analysis_data: str) -> str:
        """Extract style preferences from code analysis."""
        try:
            data = (
                json.loads(analysis_data)
                if isinstance(analysis_data, str)
                else analysis_data
            )

            style_preferences = {
                "naming_style": {},
                "code_organization": {},
                "complexity_preference": {},
                "documentation_style": {},
                "import_style": {},
            }

            # Extract naming preferences
            functions = data.get("functions", [])
            if functions:
                func_names = [f.get("name", "") for f in functions]
                style_preferences["naming_style"] = self._analyze_naming_pattern(
                    func_names
                )

            # Extract organization preferences
            if functions:
                avg_function_length = sum(
                    f.get("line_count", 0) for f in functions
                ) / len(functions)
                style_preferences["code_organization"][
                    "preferred_function_length"
                ] = avg_function_length

            # Extract complexity preferences
            if functions:
                avg_complexity = sum(f.get("complexity", 0) for f in functions) / len(
                    functions
                )
                style_preferences["complexity_preference"][
                    "average_complexity"
                ] = avg_complexity

            # Extract documentation preferences
            documented_funcs = [f for f in functions if f.get("docstring")]
            if functions:
                style_preferences["documentation_style"]["documentation_ratio"] = len(
                    documented_funcs
                ) / len(functions)

            return json.dumps(style_preferences, indent=2)

        except Exception as e:
            return f"Error extracting style preferences: {str(e)}"

    def compare_coding_styles(self, style_data_1: str, style_data_2: str) -> str:
        """Compare different coding styles and patterns."""
        try:
            style1 = (
                json.loads(style_data_1)
                if isinstance(style_data_1, str)
                else style_data_1
            )
            style2 = (
                json.loads(style_data_2)
                if isinstance(style_data_2, str)
                else style_data_2
            )

            comparison = {
                "similarities": [],
                "differences": [],
                "compatibility_score": 0.0,
                "recommendations": [],
            }

            # Compare naming styles
            naming1 = style1.get("naming_style", {})
            naming2 = style2.get("naming_style", {})

            if naming1 and naming2:
                common_patterns = set(naming1.keys()) & set(naming2.keys())
                if common_patterns:
                    comparison["similarities"].append(
                        f"Common naming patterns: {list(common_patterns)}"
                    )

                different_patterns = set(naming1.keys()) ^ set(naming2.keys())
                if different_patterns:
                    comparison["differences"].append(
                        f"Different naming patterns: {list(different_patterns)}"
                    )

            # Calculate compatibility score (simple heuristic)
            similarity_count = len(comparison["similarities"])
            difference_count = len(comparison["differences"])
            total_factors = similarity_count + difference_count

            if total_factors > 0:
                comparison["compatibility_score"] = similarity_count / total_factors

            return json.dumps(comparison, indent=2)

        except Exception as e:
            return f"Error comparing coding styles: {str(e)}"

    def generate_style_recommendations(self, learned_preferences: str) -> str:
        """Generate personalized style recommendations."""
        try:
            preferences = (
                json.loads(learned_preferences)
                if isinstance(learned_preferences, str)
                else learned_preferences
            )

            recommendations = {
                "style_suggestions": [],
                "consistency_improvements": [],
                "best_practice_alignment": [],
                "priority_actions": [],
            }

            # Analyze naming consistency
            naming_style = preferences.get("naming_style", {})
            if naming_style:
                dominant_style = max(naming_style, key=naming_style.get)
                if naming_style[dominant_style] < 0.9:
                    recommendations["consistency_improvements"].append(
                        f"Standardize to {dominant_style} naming convention"
                    )

            # Analyze function length preferences
            org_prefs = preferences.get("code_organization", {})
            avg_length = org_prefs.get("preferred_function_length", 0)
            if avg_length > 50:
                recommendations["style_suggestions"].append(
                    "Consider breaking down large functions into smaller, more focused ones"
                )

            # Analyze documentation
            doc_style = preferences.get("documentation_style", {})
            doc_ratio = doc_style.get("documentation_ratio", 0)
            if doc_ratio < 0.7:
                recommendations["best_practice_alignment"].append(
                    "Increase documentation coverage for better maintainability"
                )

            # Priority actions
            if recommendations["consistency_improvements"]:
                recommendations["priority_actions"].extend(
                    recommendations["consistency_improvements"]
                )

            return json.dumps(recommendations, indent=2)

        except Exception as e:
            return f"Error generating style recommendations: {str(e)}"

    def update_style_preferences(self, preference_data: str) -> str:
        """Update learned style preferences based on new evidence."""
        try:
            update_data = (
                json.loads(preference_data)
                if isinstance(preference_data, str)
                else preference_data
            )

            # This is a placeholder implementation - in practice this would interact
            # with a persistent storage system for learned preferences
            preference_update = {
                "status": "preferences_updated",
                "updated_categories": (
                    list(update_data.keys()) if isinstance(update_data, dict) else []
                ),
                "timestamp": datetime.now().isoformat(),
                "confidence_adjustments": {},
                "recommendations": [],
            }

            # Analyze preference updates
            if isinstance(update_data, dict):
                for category, values in update_data.items():
                    if category in [
                        "naming",
                        "structure",
                        "complexity",
                        "documentation",
                    ]:
                        preference_update["confidence_adjustments"][category] = "high"
                        preference_update["recommendations"].append(
                            f"Strong evidence found for {category} preferences"
                        )

            return json.dumps(preference_update, indent=2)

        except Exception as e:
            return f"Error updating style preferences: {str(e)}"

    # Quality Assessment Tools Implementation

    def calculate_maintainability_score(self, analysis_data: str) -> str:
        """Calculate a maintainability score for code."""
        try:
            data = (
                json.loads(analysis_data)
                if isinstance(analysis_data, str)
                else analysis_data
            )

            score_components = {
                "complexity_score": 0.0,
                "documentation_score": 0.0,
                "naming_score": 0.0,
                "organization_score": 0.0,
                "overall_score": 0.0,
            }

            functions = data.get("functions", [])

            if functions:
                # Complexity score (lower complexity = higher score)
                avg_complexity = sum(f.get("complexity", 1) for f in functions) / len(
                    functions
                )
                score_components["complexity_score"] = max(
                    0, 1 - (avg_complexity - 1) / 10
                )

                # Documentation score
                documented = [f for f in functions if f.get("docstring")]
                score_components["documentation_score"] = len(documented) / len(
                    functions
                )

                # Naming score (consistency)
                func_names = [f.get("name", "") for f in functions]
                naming_patterns = self._analyze_naming_pattern(func_names)
                if naming_patterns:
                    max_consistency = max(naming_patterns.values())
                    score_components["naming_score"] = max_consistency

                # Organization score (function size)
                avg_length = sum(f.get("line_count", 0) for f in functions) / len(
                    functions
                )
                score_components["organization_score"] = max(
                    0, 1 - max(0, avg_length - 20) / 80
                )

            # Calculate overall score
            weights = [0.3, 0.2, 0.2, 0.3]  # complexity, docs, naming, organization
            scores = [
                score_components["complexity_score"],
                score_components["documentation_score"],
                score_components["naming_score"],
                score_components["organization_score"],
            ]

            score_components["overall_score"] = sum(
                w * s for w, s in zip(weights, scores)
            )

            maintainability_result = {
                "scores": score_components,
                "grade": self._score_to_grade(score_components["overall_score"]),
                "recommendations": self._get_maintainability_recommendations(
                    score_components
                ),
            }

            return json.dumps(maintainability_result, indent=2)

        except Exception as e:
            return f"Error calculating maintainability score: {str(e)}"

    def identify_code_smells(self, analysis_data: str) -> str:
        """Identify potential code smells and anti-patterns."""
        try:
            data = (
                json.loads(analysis_data)
                if isinstance(analysis_data, str)
                else analysis_data
            )

            code_smells = {
                "smells_detected": [],
                "severity_breakdown": {"high": 0, "medium": 0, "low": 0},
                "recommendations": [],
            }

            functions = data.get("functions", [])
            classes = data.get("classes", [])

            # Long functions
            long_functions = [f for f in functions if f.get("line_count", 0) > 50]
            if long_functions:
                code_smells["smells_detected"].append(
                    {
                        "smell": "Long Functions",
                        "count": len(long_functions),
                        "severity": "medium",
                        "description": "Functions with more than 50 lines",
                    }
                )
                code_smells["severity_breakdown"]["medium"] += len(long_functions)

            # High complexity functions
            complex_functions = [f for f in functions if f.get("complexity", 0) > 10]
            if complex_functions:
                code_smells["smells_detected"].append(
                    {
                        "smell": "High Complexity",
                        "count": len(complex_functions),
                        "severity": "high",
                        "description": "Functions with cyclomatic complexity > 10",
                    }
                )
                code_smells["severity_breakdown"]["high"] += len(complex_functions)

            # Large classes
            large_classes = [c for c in classes if len(c.get("methods", [])) > 20]
            if large_classes:
                code_smells["smells_detected"].append(
                    {
                        "smell": "Large Classes",
                        "count": len(large_classes),
                        "severity": "medium",
                        "description": "Classes with more than 20 methods",
                    }
                )
                code_smells["severity_breakdown"]["medium"] += len(large_classes)

            # Undocumented functions
            undocumented = [f for f in functions if not f.get("docstring")]
            if len(undocumented) > len(functions) * 0.5:
                code_smells["smells_detected"].append(
                    {
                        "smell": "Poor Documentation",
                        "count": len(undocumented),
                        "severity": "low",
                        "description": "More than 50% of functions lack documentation",
                    }
                )
                code_smells["severity_breakdown"]["low"] += 1

            # Generate recommendations
            for smell in code_smells["smells_detected"]:
                if smell["smell"] == "Long Functions":
                    code_smells["recommendations"].append(
                        "Break down long functions into smaller, focused functions"
                    )
                elif smell["smell"] == "High Complexity":
                    code_smells["recommendations"].append(
                        "Refactor complex functions to reduce cyclomatic complexity"
                    )
                elif smell["smell"] == "Large Classes":
                    code_smells["recommendations"].append(
                        "Consider splitting large classes using Single Responsibility Principle"
                    )
                elif smell["smell"] == "Poor Documentation":
                    code_smells["recommendations"].append(
                        "Add docstrings to improve code documentation"
                    )

            return json.dumps(code_smells, indent=2)

        except Exception as e:
            return f"Error identifying code smells: {str(e)}"

    def assess_test_coverage(self, directory_path: str) -> str:
        """Assess test coverage and testing patterns."""
        try:
            path = Path(directory_path)

            test_analysis = {
                "test_files_found": [],
                "test_patterns": [],
                "coverage_estimate": 0.0,
                "recommendations": [],
            }

            # Find test files
            test_patterns = ["*test*.py", "*_test.py", "test_*.py", "tests/*.py"]
            test_files = []

            for pattern in test_patterns:
                test_files.extend(path.glob(pattern))

            test_analysis["test_files_found"] = [str(f) for f in test_files]

            # Find all source files
            source_files = list(path.glob("**/*.py"))
            source_files = [f for f in source_files if "test" not in str(f).lower()]

            # Estimate coverage (simple heuristic)
            if source_files:
                test_analysis["coverage_estimate"] = min(
                    1.0, len(test_files) / len(source_files)
                )

            # Analyze test patterns
            if test_files:
                test_analysis["test_patterns"].append("Unit tests present")

                # Check for common test frameworks
                for test_file in test_files[:5]:  # Check first 5 test files
                    try:
                        with open(test_file, "r") as f:
                            content = f.read()
                            if "unittest" in content:
                                test_analysis["test_patterns"].append(
                                    "unittest framework"
                                )
                            if "pytest" in content:
                                test_analysis["test_patterns"].append(
                                    "pytest framework"
                                )
                            if "mock" in content:
                                test_analysis["test_patterns"].append("mocking used")
                    except:
                        pass

            # Generate recommendations
            if test_analysis["coverage_estimate"] < 0.5:
                test_analysis["recommendations"].append(
                    "Increase test coverage - current coverage appears low"
                )

            if not test_files:
                test_analysis["recommendations"].append(
                    "Add unit tests to improve code reliability"
                )

            return json.dumps(test_analysis, indent=2)

        except Exception as e:
            return f"Error assessing test coverage: {str(e)}"

    # Diff-based File Modification Tools

    def generate_code_diff(self, original_content: str, modified_content: str, file_path: str = "file") -> str:
        """Generate a unified diff between original and modified code."""
        try:
            # Split content into lines for difflib
            original_lines = original_content.splitlines(keepends=True)
            modified_lines = modified_content.splitlines(keepends=True)
            
            # Generate unified diff
            diff = difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm=""
            )
            
            diff_content = ''.join(diff)
            
            if not diff_content:
                return json.dumps({
                    "status": "no_changes",
                    "message": "No differences found between original and modified content",
                    "diff": ""
                })
            
            return json.dumps({
                "status": "success",
                "message": "Diff generated successfully",
                "diff": diff_content,
                "stats": self._get_diff_stats(diff_content)
            })
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error generating diff: {str(e)}"
            })

    def apply_code_diff(self, file_path: str, diff_content: str, backup: bool = True) -> str:
        """Apply a unified diff to modify a file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"File {file_path} does not exist"
                })
            
            # Read original content
            with open(path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Create backup if requested
            backup_path = None
            if backup:
                backup_path = path.with_suffix(path.suffix + f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
            
            # Apply the diff
            result = self._apply_diff_to_content(original_content, diff_content)
            
            if result["status"] == "success":
                # Write the modified content
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(result["modified_content"])
                
                return json.dumps({
                    "status": "success",
                    "message": f"Diff applied successfully to {file_path}",
                    "backup_created": backup_path.name if backup_path else None,
                    "changes_applied": result["changes_applied"]
                })
            else:
                return json.dumps(result)
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error applying diff: {str(e)}"
            })

    def create_smart_diff(self, file_path: str, instructions: str) -> str:
        """Create intelligent diff that preserves context and applies changes safely."""
        try:
            path = Path(file_path)
            if not path.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"File {file_path} does not exist"
                })
            
            # Read current content
            with open(path, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            if not self.gemini_client:
                return json.dumps({
                    "status": "error",
                    "message": "AI client not available for smart diff generation"
                })
            
            # Use AI to generate the modified content
            prompt = f"""
            You are a code modification assistant. Given the current file content and instructions, 
            generate the modified version that implements the requested changes while preserving 
            the existing structure and style.

            Current file content:
            ```
            {current_content}
            ```

            Instructions: {instructions}

            Rules:
            1. Make minimal changes to implement the instructions
            2. Preserve existing code style and formatting
            3. Keep all existing functionality intact
            4. Add comments where appropriate
            5. Return ONLY the modified file content, no explanations

            Modified content:
            """

            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=prompt
            )

            if response and response.text:
                modified_content = response.text.strip()
                
                # Generate diff
                diff_result = self.generate_code_diff(current_content, modified_content, file_path)
                diff_data = json.loads(diff_result)
                
                if diff_data["status"] == "success":
                    return json.dumps({
                        "status": "success",
                        "message": "Smart diff created successfully",
                        "diff": diff_data["diff"],
                        "stats": diff_data["stats"],
                        "preview": self._create_diff_preview(current_content, modified_content)
                    })
                else:
                    return diff_result
            else:
                return json.dumps({
                    "status": "error",
                    "message": "AI failed to generate modified content"
                })
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error creating smart diff: {str(e)}"
            })

    def preview_diff_changes(self, file_path: str, diff_content: str) -> str:
        """Preview what changes a diff would make to a file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"File {file_path} does not exist"
                })
            
            # Read original content
            with open(path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Apply diff to get preview
            result = self._apply_diff_to_content(original_content, diff_content)
            
            if result["status"] == "success":
                return json.dumps({
                    "status": "success",
                    "message": "Preview generated successfully",
                    "preview": self._create_diff_preview(original_content, result["modified_content"]),
                    "changes_summary": result["changes_applied"],
                    "modified_content": result["modified_content"]
                })
            else:
                return json.dumps(result)
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error previewing diff: {str(e)}"
            })

    def modify_file_with_diff(self, file_path: str, modification_instructions: str, auto_apply: bool = False) -> str:
        """Modify a file using AI-generated diff instructions."""
        try:
            # Generate smart diff
            diff_result = self.create_smart_diff(file_path, modification_instructions)
            diff_data = json.loads(diff_result)
            
            if diff_data["status"] != "success":
                return diff_result
            
            if auto_apply:
                # Apply the diff automatically
                apply_result = self.apply_code_diff(file_path, diff_data["diff"])
                apply_data = json.loads(apply_result)
                
                if apply_data["status"] == "success":
                    return json.dumps({
                        "status": "success",
                        "message": f"File {file_path} modified successfully",
                        "diff": diff_data["diff"],
                        "preview": diff_data["preview"],
                        "applied": True,
                        "backup_created": apply_data.get("backup_created")
                    })
                else:
                    return apply_result
            else:
                # Return diff for manual review
                return json.dumps({
                    "status": "ready_to_apply",
                    "message": f"Diff ready for {file_path}. Review and apply manually.",
                    "diff": diff_data["diff"],
                    "preview": diff_data["preview"],
                    "applied": False,
                    "instructions": "Use apply_code_diff() to apply these changes"
                })
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Error modifying file with diff: {str(e)}"
            })

    # Helper methods for diff operations

    def _apply_diff_to_content(self, original_content: str, diff_content: str) -> Dict[str, Any]:
        """Apply a unified diff to content and return the result."""
        try:
            original_lines = original_content.splitlines(keepends=True)
            
            # Parse the diff
            diff_lines = diff_content.splitlines()
            
            # Apply the patch
            patched_lines = list(original_lines)
            changes_applied = []
            
            i = 0
            line_offset = 0
            
            while i < len(diff_lines):
                line = diff_lines[i]
                
                # Look for hunk headers (@@)
                if line.startswith('@@'):
                    # Parse hunk header: @@ -start,count +start,count @@
                    parts = line.split()
                    if len(parts) >= 3:
                        old_info = parts[1][1:]  # Remove the '-'
                        new_info = parts[2][1:]  # Remove the '+'
                        
                        old_start = int(old_info.split(',')[0]) - 1  # Convert to 0-based
                        old_start += line_offset
                        
                        # Process the hunk
                        i += 1
                        hunk_changes = []
                        
                        while i < len(diff_lines) and not diff_lines[i].startswith('@@'):
                            hunk_line = diff_lines[i]
                            
                            if hunk_line.startswith('-'):
                                # Line to remove
                                removed_content = hunk_line[1:]
                                hunk_changes.append(('remove', removed_content))
                            elif hunk_line.startswith('+'):
                                # Line to add
                                added_content = hunk_line[1:] + '\n'
                                hunk_changes.append(('add', added_content))
                            elif hunk_line.startswith(' '):
                                # Context line (unchanged)
                                hunk_changes.append(('context', hunk_line[1:]))
                            
                            i += 1
                        
                        # Apply hunk changes
                        self._apply_hunk_changes(patched_lines, old_start, hunk_changes, changes_applied)
                        
                        continue
                
                i += 1
            
            modified_content = ''.join(patched_lines)
            
            return {
                "status": "success",
                "modified_content": modified_content,
                "changes_applied": changes_applied
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error applying diff: {str(e)}"
            }

    def _apply_hunk_changes(self, lines: List[str], start_line: int, changes: List[Tuple[str, str]], changes_applied: List[str]):
        """Apply changes from a hunk to the lines list."""
        current_line = start_line
        
        for change_type, content in changes:
            if change_type == 'remove':
                if current_line < len(lines):
                    removed_line = lines.pop(current_line)
                    changes_applied.append(f"Removed line {current_line + 1}: {removed_line.strip()}")
            elif change_type == 'add':
                lines.insert(current_line, content)
                changes_applied.append(f"Added line {current_line + 1}: {content.strip()}")
                current_line += 1
            elif change_type == 'context':
                current_line += 1

    def _get_diff_stats(self, diff_content: str) -> Dict[str, int]:
        """Get statistics about a diff."""
        lines = diff_content.splitlines()
        stats = {
            "additions": 0,
            "deletions": 0,
            "hunks": 0
        }
        
        for line in lines:
            if line.startswith('@@'):
                stats["hunks"] += 1
            elif line.startswith('+') and not line.startswith('+++'):
                stats["additions"] += 1
            elif line.startswith('-') and not line.startswith('---'):
                stats["deletions"] += 1
        
        return stats

    def _create_diff_preview(self, original: str, modified: str) -> str:
        """Create a human-readable preview of changes."""
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()
        
        # Use difflib to create a nice side-by-side comparison
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile="Original",
            tofile="Modified",
            lineterm=""
        )
        
        preview_lines = []
        for line in diff:
            if line.startswith('+++') or line.startswith('---'):
                continue
            elif line.startswith('@@'):
                preview_lines.append(f"\n{line}")
            elif line.startswith('+'):
                preview_lines.append(f"+ {line[1:]}")
            elif line.startswith('-'):
                preview_lines.append(f"- {line[1:]}")
            else:
                preview_lines.append(f"  {line[1:] if line else ''}")
        
        return '\n'.join(preview_lines)

    def _analyze_naming_pattern(self, names: List[str]) -> Dict[str, float]:
        """Analyze naming patterns in a list of names."""
        if not names:
            return {}
        
        patterns = {
            "snake_case": 0,
            "camelCase": 0,
            "PascalCase": 0,
            "UPPER_CASE": 0
        }
        
        for name in names:
            if not name:
                continue
                
            if '_' in name and name.islower():
                patterns["snake_case"] += 1
            elif '_' in name and name.isupper():
                patterns["UPPER_CASE"] += 1
            elif name[0].isupper() and any(c.isupper() for c in name[1:]):
                patterns["PascalCase"] += 1
            elif name[0].islower() and any(c.isupper() for c in name[1:]):
                patterns["camelCase"] += 1
        
        # Convert to percentages
        total = len(names)
        return {pattern: count / total for pattern, count in patterns.items() if count > 0}

    # Terminal Command Execution Methods

    def execute_terminal_command(self, command: str) -> tuple[bool, str]:
        """
        Execute a terminal command safely and return the result.
        Automatically activates virtual environment for Python-related commands.

        Args:
            command: The shell command to execute

        Returns:
            Tuple of (success: bool, output: str)
        """
        try:
            # Prepare command with virtual environment activation if needed
            prepared_command = self._prepare_command_with_venv(command)

            if self.debug:
                if prepared_command != command:
                    print(f"🔧 Executing with venv: {prepared_command}")
                else:
                    print(f"🔧 Executing terminal command: {command}")

            # For commands with shell operators (&&, |, etc.), use shell=True
            use_shell = any(
                op in prepared_command for op in ["&&", "||", "|", ">", "<", ";"]
            )

            if use_shell:
                # Execute with shell for complex commands
                result = subprocess.run(
                    prepared_command,
                    shell=True,
                    cwd=self.workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            else:
                # Parse command safely using shlex for simple commands
                parsed_command = shlex.split(prepared_command)
                result = subprocess.run(
                    parsed_command,
                    cwd=self.workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            parsed_command = shlex.split(command)

            # Execute the command
            result = subprocess.run(
                parsed_command,
                cwd=self.workspace_path,  # Execute in workspace directory
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                check=False,  # Don't raise exception on non-zero exit code
            )

            # Prepare output
            output_lines = []
            if result.stdout:
                output_lines.extend(result.stdout.strip().split("\n"))
            if result.stderr:
                output_lines.extend(
                    [f"[STDERR] {line}" for line in result.stderr.strip().split("\n")]
                )

            output = (
                "\n".join(output_lines)
                if output_lines
                else "Command completed (no output)"
            )

            # Check if command was successful
            success = result.returncode == 0

            if not success:
                output += f"\n[EXIT CODE] {result.returncode}"

            return success, output

        except subprocess.TimeoutExpired:
            return False, "❌ Command timed out (30 seconds)"
        except FileNotFoundError:
            command_name = command.split()[0] if command.split() else command
            return False, f"❌ Command not found: {command_name}"
        except PermissionError:
            return False, f"❌ Permission denied executing: {command}"
        except Exception as e:
            return False, f"❌ Error executing command: {e}"

    def execute_dynamic_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Dynamically execute a tool by its name with the given arguments.

        Args:
            tool_name: The name of the tool to execute.
            arguments: A dictionary of arguments for the tool.

        Returns:
            A string representation of the tool's execution result.
        """
        if self.debug:
            print(f"🛠️ Attempting to execute tool: {tool_name} with args: {arguments}")

        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    # Ensure the command is prepared with venv if it's a terminal command tool
                    if (
                        tool_name == "execute_terminal_command"
                        and "command" in arguments
                    ):
                        arguments["command"] = self._prepare_command_with_venv(
                            arguments["command"]
                        )

                    target_func = tool.func
                    if target_func is None:
                        return f"Error: Tool '{tool_name}' has no associated function."

                    # Langchain's Tool.func expects arguments as a single string or dict
                    # depending on how it was defined. We need to inspect the target function.
                    sig = inspect.signature(target_func) if target_func else None
                    if not sig:
                        return f"Error: Tool '{tool_name}' has no valid function signature."

                    if len(sig.parameters) == 1 and list(sig.parameters.keys())[0] in [
                        "tool_input",
                        "input_str",
                        "code_data",
                        "query",
                        "file_path",
                        "command",
                        "diff_content",
                    ]:  # Common single arg names
                        # If the tool function expects a single argument, and we have multiple,
                        # we might need to pass them as a JSON string or a dict if the tool handles it.
                        # For now, if arguments is a dict and the func takes one arg, pass the dict.
                        # This assumes the tool function is designed to unpack it or handle it.
                        # A more robust solution would involve better schema matching.
                        if isinstance(arguments, dict) and len(arguments) == 1:
                            result = target_func(list(arguments.values())[0])
                        else:
                            # Pass arguments as a JSON string if the tool expects a single string input
                            if any(
                                p.annotation == str for p in sig.parameters.values()
                            ):
                                result = target_func(json.dumps(arguments))
                            else:  # Pass the dict directly
                                result = target_func(arguments)

                    else:  # Tool function expects arguments to be unpacked
                        result = target_func(**arguments)

                    return str(result) if not isinstance(result, str) else result
                except TypeError as e:
                    if self.debug:
                        sig_info = inspect.signature(tool.func) if tool.func else "None"
                        print(
                            f"❌ TypeError executing tool {tool_name}: {e}. Args: {arguments}, Sig: {sig_info}"
                        )
                    return f"Error: TypeError in tool '{tool_name}'. Check arguments. Details: {e}"
                except Exception as e:
                    if self.debug:
                        print(f"❌ Exception executing tool {tool_name}: {e}")
                    return f"Error executing tool '{tool_name}': {e}"

        return f"Error: Tool '{tool_name}' not found."


# Create a global instance for easy access
def create_symbiote_tools(
    gemini_api_key: Optional[str] = None,
    debug: bool = False,
    workspace_path: Optional[Path] = None,
) -> SymbioteTools:
    """Factory function to create SymbioteTools instance."""
    return SymbioteTools(
        gemini_api_key=gemini_api_key, debug=debug, workspace_path=workspace_path
    )
