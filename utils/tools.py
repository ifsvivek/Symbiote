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

Author: Vivek Sharma
License: MIT
"""

import os
import ast
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
            # File System Tools
            Tool(
                name="read_file_content",
                description="Read and analyze the content of a specific file",
                func=self.read_file_content,
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

    # Helper Methods

    def _analyze_naming_pattern(self, names: List[str]) -> Dict[str, float]:
        """Analyze naming patterns in a list of names."""
        patterns = {
            "snake_case": 0,
            "camelCase": 0,
            "PascalCase": 0,
            "CONSTANT_CASE": 0,
        }

        for name in names:
            if not name:
                continue

            if "_" in name and name.islower():
                patterns["snake_case"] += 1
            elif (
                name[0].islower()
                and any(c.isupper() for c in name[1:])
                and "_" not in name
            ):
                patterns["camelCase"] += 1
            elif name[0].isupper() and not "_" in name:
                patterns["PascalCase"] += 1
            elif "_" in name and name.isupper():
                patterns["CONSTANT_CASE"] += 1

        total = sum(patterns.values())
        if total > 0:
            return {pattern: count / total for pattern, count in patterns.items()}
        return {pattern: 0.0 for pattern in patterns}

    def _score_to_grade(self, score: float) -> str:
        """Convert a score to a letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"

    def _get_maintainability_recommendations(
        self, scores: Dict[str, float]
    ) -> List[str]:
        """Get recommendations based on maintainability scores."""
        recommendations = []

        if scores["complexity_score"] < 0.7:
            recommendations.append("Reduce function complexity through refactoring")

        if scores["documentation_score"] < 0.7:
            recommendations.append("Improve documentation coverage")

        if scores["naming_score"] < 0.8:
            recommendations.append("Standardize naming conventions")

        if scores["organization_score"] < 0.7:
            recommendations.append("Break down large functions into smaller units")

        return recommendations

    # Terminal Command Execution Methods

    def execute_terminal_command(self, command: str) -> tuple[bool, str]:
        """
        Execute a terminal command safely and return the result.

        Args:
            command: The shell command to execute

        Returns:
            Tuple of (success: bool, output: str)
        """
        try:
            if self.debug:
                print(f"🔧 Executing terminal command: {command}")

            # Parse command safely using shlex
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

    def handle_terminal_commands(self, ai_response: str) -> None:
        """Handle terminal command execution requests from AI responses."""
        # Look for EXECUTE_COMMAND: pattern in AI response
        command_pattern = r"EXECUTE_COMMAND:\s*(.+?)(?:\n|$)"
        matches = re.findall(command_pattern, ai_response, re.IGNORECASE | re.MULTILINE)

        for command in matches:
            command = command.strip()
            if command:
                success, output = self.execute_terminal_command(command)

                # Display the result
                if success:
                    print(f"\n✅ Command output:\n{output}")
                else:
                    print(f"\n❌ Command failed:\n{output}")

    # File Operation Methods

    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in workspace."""
        try:
            full_path = self.workspace_path / file_path
            return full_path.exists()
        except:
            return False

    def create_file(self, file_path: str, content: str) -> bool:
        """Create a new file with given content."""
        try:
            full_path = self.workspace_path / file_path

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"✅ Created {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error creating file: {e}")
            return False

    def handle_file_creation(self, file_path: str, user_request: str) -> bool:
        """Handle creation of a new file."""
        try:
            full_path = self.workspace_path / file_path

            if full_path.exists():
                print(f"❌ File {file_path} already exists!")
                return False

            # Use AI to generate file content
            if self.gemini_client:
                creation_prompt = f"""
Create a new file based on this request: {user_request}

File path: {file_path}

Generate appropriate content for this file based on its name and the user's request.
Return ONLY the file content without any explanations or markdown formatting.
"""

                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash-preview-05-20", contents=creation_prompt
                )

                if response and response.text:
                    content = response.text.strip()

                    print(f"\n📝 Generated content for {file_path}:")
                    print("=" * 50)
                    print(content[:500] + ("..." if len(content) > 500 else ""))
                    print("=" * 50)

                    confirmation = (
                        input(f"\n🤔 Create {file_path} with this content? (yes/no): ")
                        .strip()
                        .lower()
                    )

                    if confirmation in ["yes", "y"]:
                        return self.create_file(file_path, content)
                    else:
                        print("❌ File creation cancelled")
                        return False

            print("❌ AI file generation requires Gemini API key")
            return False

        except Exception as e:
            print(f"❌ Error creating file: {e}")
            return False

    def read_file_with_lines(
        self, file_path: str, start_line: int = 1, end_line: Optional[int] = None
    ) -> str:
        """Read file content with line numbers for easy reference."""
        try:
            full_path = self.workspace_path / file_path
            if not full_path.exists():
                return f"❌ File not found: {file_path}"

            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            if end_line is None:
                end_line = len(lines)

            # Adjust for 1-based indexing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)

            output = [f"📄 {file_path} (lines {start_line}-{end_idx}):", ""]

            for i, line in enumerate(lines[start_idx:end_idx], start=start_line):
                output.append(f"{i:4d} | {line.rstrip()}")

            return "\n".join(output)

        except Exception as e:
            return f"❌ Error reading file: {e}"

    def show_file_diff(
        self, file_path: str, original_content: str, new_content: str
    ) -> str:
        """Generate and display a unified diff between original and new content."""
        try:
            original_lines = original_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)

            diff = list(
                difflib.unified_diff(
                    original_lines,
                    new_lines,
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                    lineterm="",
                )
            )

            if not diff:
                return "📝 No changes detected."

            output = [f"🔄 Diff for {file_path}:", ""]

            for line in diff:
                if line.startswith("+++") or line.startswith("---"):
                    output.append(f"📁 {line.rstrip()}")
                elif line.startswith("@@"):
                    output.append(f"📍 {line.rstrip()}")
                elif line.startswith("+"):
                    output.append(f"✅ {line.rstrip()}")
                elif line.startswith("-"):
                    output.append(f"❌ {line.rstrip()}")
                else:
                    output.append(f"   {line.rstrip()}")

            return "\n".join(output)

        except Exception as e:
            return f"❌ Error generating diff: {e}"

    def apply_file_changes(
        self, file_path: str, new_content: str, backup: bool = True
    ) -> bool:
        """Apply changes to a file with optional backup."""
        try:
            full_path = self.workspace_path / file_path

            # Create backup if requested
            if backup and full_path.exists():
                backup_path = full_path.with_suffix(full_path.suffix + ".backup")
                with open(full_path, "r", encoding="utf-8") as src:
                    with open(backup_path, "w", encoding="utf-8") as dst:
                        dst.write(src.read())
                print(f"💾 Backup created: {backup_path.name}")

            # Write new content
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"✅ Changes applied to {file_path}")
            return True

        except Exception as e:
            print(f"❌ Error applying changes: {e}")
            return False

    def execute_file_operation(self, operation: str, file_path: str, **kwargs) -> str:
        """Execute file operations based on AI requests."""
        try:
            if operation == "read":
                start_line = kwargs.get("start_line", 1)
                end_line = kwargs.get("end_line", None)
                return self.read_file_with_lines(file_path, start_line, end_line)

            elif operation == "modify":
                original_content = kwargs.get("original_content", "")
                new_content = kwargs.get("new_content", "")

                # Show diff first
                diff_output = self.show_file_diff(
                    file_path, original_content, new_content
                )
                print(diff_output)

                # Ask for confirmation
                print(f"\n🤔 Apply these changes to {file_path}?")
                response = (
                    input("   Type 'yes' to apply, 'no' to cancel: ").strip().lower()
                )

                if response in ["yes", "y"]:
                    success = self.apply_file_changes(file_path, new_content)
                    return (
                        "✅ Changes applied successfully!"
                        if success
                        else "❌ Failed to apply changes"
                    )
                else:
                    return "❌ Changes cancelled by user"

            elif operation == "create":
                content = kwargs.get("content", "")
                full_path = self.workspace_path / file_path

                if full_path.exists():
                    print(f"⚠️  File {file_path} already exists!")
                    response = input("   Overwrite? (yes/no): ").strip().lower()
                    if response not in ["yes", "y"]:
                        return "❌ File creation cancelled"

                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)

                return f"✅ Created file: {file_path}"

            else:
                return f"❌ Unknown operation: {operation}"

        except Exception as e:
            return f"❌ Error executing file operation: {e}"

    def parse_ai_file_commands(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response for file operation commands."""
        commands = []

        # Look for file operation patterns in AI response
        patterns = {
            "read_file": r"(?:read|show|display)\s+(?:file\s+)?([^\s]+\.(?:py|js|ts|java|cpp|c|h|txt|md))",
            "modify_file": r"(?:modify|change|edit|update)\s+(?:file\s+)?([^\s]+\.(?:py|js|ts|java|cpp|c|h|txt|md))",
            "create_file": r"(?:create|make|new)\s+(?:file\s+)?([^\s]+\.(?:py|js|ts|java|cpp|c|h|txt|md))",
        }

        for operation, pattern in patterns.items():
            matches = re.finditer(pattern, ai_response, re.IGNORECASE)
            for match in matches:
                file_path = match.group(1)
                commands.append(
                    {
                        "operation": operation.split("_")[0],
                        "file_path": file_path,
                        "context": match.group(0),
                    }
                )

        return commands

    def smart_command_parser(
        self,
        user_input: str,
        ai_response: str,
        workspace_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Intelligently parse user requests and AI responses for file operations."""
        operations = []

        # Combine user input and AI response for better context
        combined_text = f"{user_input} {ai_response}".lower()

        # Enhanced patterns for more natural language
        file_patterns = [
            # Reading files
            (
                r"(?:show|read|display|open|view|check|examine|look at)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_./]+\.(?:py|js|ts|java|cpp|c|h|txt|md|json|yaml|yml))",
                "read",
            ),
            (
                r"(?:what\'s|whats)\s+in\s+([a-zA-Z0-9_./]+\.(?:py|js|ts|java|cpp|c|h|txt|md|json|yaml|yml))",
                "read",
            ),
            (
                r"(?:content|contents)\s+of\s+([a-zA-Z0-9_./]+\.(?:py|js|ts|java|cpp|c|h|txt|md|json|yaml|yml))",
                "read",
            ),
            # Modifying files
            (
                r"(?:fix|modify|change|edit|update|alter)\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_./]+\.(?:py|js|ts|java|cpp|c|h|txt|md|json|yaml|yml))",
                "modify",
            ),
            (
                r"(?:add|insert)\s+.*(?:to|in)\s+([a-zA-Z0-9_./]+\.(?:py|js|ts|java|cpp|c|h|txt|md|json|yaml|yml))",
                "modify",
            ),
            (
                r"(?:remove|delete)\s+.*(?:from)\s+([a-zA-Z0-9_./]+\.(?:py|js|ts|java|cpp|c|h|txt|md|json|yaml|yml))",
                "modify",
            ),
            # Creating files
            (
                r"(?:create|make|generate|new)\s+(?:a\s+)?(?:file\s+)?(?:called\s+)?([a-zA-Z0-9_./]+\.(?:py|js|ts|java|cpp|c|h|txt|md|json|yaml|yml))",
                "create",
            ),
        ]

        # Find workspace files that might be referenced without extension
        workspace_files = []
        if workspace_context:
            recent_files = workspace_context.get("recent_files", [])
            file_structure = workspace_context.get("file_structure", [])
            workspace_files = recent_files + [
                f.get("name", "") for f in file_structure if f.get("type") == "file"
            ]

        # Check for file patterns
        for pattern, operation in file_patterns:
            matches = re.finditer(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                file_path = match.group(1)
                operations.append(
                    {
                        "operation": operation,
                        "file_path": file_path,
                        "context": match.group(0),
                        "confidence": 0.8,
                    }
                )

        # Check for references to known files without full paths
        words = combined_text.split()
        for word in words:
            clean_word = word.strip('.,!?()[]{}":;')
            for workspace_file in workspace_files:
                if (
                    clean_word in workspace_file.lower()
                    or workspace_file.lower() in clean_word
                ):
                    # Determine operation based on context
                    operation = "read"  # default
                    if any(
                        verb in combined_text
                        for verb in ["fix", "modify", "change", "edit", "update"]
                    ):
                        operation = "modify"
                    elif any(
                        verb in combined_text for verb in ["create", "make", "new"]
                    ):
                        operation = "create"

                    operations.append(
                        {
                            "operation": operation,
                            "file_path": workspace_file,
                            "context": f"Reference to {workspace_file}",
                            "confidence": 0.6,
                        }
                    )
                    break

        # Remove duplicates and sort by confidence
        unique_operations = []
        seen_files = set()
        for op in sorted(operations, key=lambda x: x["confidence"], reverse=True):
            if op["file_path"] not in seen_files:
                unique_operations.append(op)
                seen_files.add(op["file_path"])

        return unique_operations[:3]  # Limit to top 3 operations

    def execute_smart_operations(self, operations: List[Dict[str, Any]]) -> str:
        """Execute file operations with smart handling."""
        results = []

        for op in operations:
            operation = op["operation"]
            file_path = op["file_path"]
            confidence = op["confidence"]

            print(
                f"\n🔧 Detected: {operation} operation on '{file_path}' (confidence: {confidence:.1f})"
            )

            if confidence < 0.5:
                print("   ⚠️ Low confidence, skipping...")
                continue

            if operation == "read":
                result = self.read_file_with_lines(file_path)
                print(result)
                results.append(f"Read {file_path}")

            elif operation == "modify":
                print(
                    f"   📝 To modify {file_path}, I need to know what changes you want."
                )
                print(
                    f"   Please specify the exact changes or let me analyze the issue first."
                )
                # First read the file to understand it
                current_content = self.read_file_with_lines(file_path)
                print(current_content)
                results.append(f"Prepared to modify {file_path}")

            elif operation == "create":
                print(
                    f"   ✨ To create {file_path}, I need to know what content you want."
                )
                print(f"   Please specify the type of file and its purpose.")
                results.append(f"Prepared to create {file_path}")

        return " | ".join(results) if results else "No operations executed"

    def interactive_file_modification(
        self, file_path: str, modification_request: str, auto_confirm: bool = False
    ) -> bool:
        """Handle interactive file modification with AI assistance."""
        try:
            full_path = self.workspace_path / file_path
            if not full_path.exists():
                print(f"❌ File not found: {file_path}")
                return False

            # Read current content
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                current_content = f.read()

            print(f"📄 Current content of {file_path}:")
            print(self.read_file_with_lines(file_path))

            # Use AI to generate the modification
            if self.gemini_client:
                modification_prompt = f"""
You are helping modify a file. Here's the current content:

```
{current_content}
```

User's modification request: {modification_request}

Please provide the complete modified file content. Make the requested changes while preserving the existing structure and style. Return only the complete file content without any explanations or markdown formatting.
"""

                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash-preview-05-20", contents=modification_prompt
                )

                if response and response.text:
                    new_content = response.text.strip()

                    # Show diff
                    diff_output = self.show_file_diff(
                        file_path, current_content, new_content
                    )
                    print(f"\n{diff_output}")

                    # Handle confirmation
                    if auto_confirm:
                        print(
                            f"\n✅ Auto-applying changes to {file_path} (already confirmed)"
                        )
                        return self.apply_file_changes(file_path, new_content)
                    else:
                        # Ask for confirmation
                        print(f"\n🤔 Apply these changes to {file_path}?")
                        confirmation = (
                            input("   Type 'yes' to apply, 'no' to cancel: ")
                            .strip()
                            .lower()
                        )

                        if confirmation in ["yes", "y"]:
                            return self.apply_file_changes(file_path, new_content)
                        else:
                            print("❌ Changes cancelled")
                            return False
                else:
                    print("❌ Failed to generate modifications")
                    return False
            else:
                print("❌ AI modification requires Gemini API key")
                return False

        except Exception as e:
            print(f"❌ Error during interactive modification: {e}")
            return False


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
