"""
🧬 Symbiote Code Parser

This module provides comprehensive code analysis and pattern extraction capabilities
for building personalized coding style graphs. It serves as the foundation for the
Symbiote AI coding assistant.

Key Features:
- Multi-language code parsing (Python, JavaScript, TypeScript, Java, etc.)
- Syntax pattern extraction and analysis
- Code structure and organization analysis
- API usage pattern detection
- Coding style and preference identification
- AI-powered pattern recognition using Gemini API

Author: Vivek Sharma
License: MIT
"""

import ast
import re
import os
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from google import genai
from google.genai import types


class Language(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


@dataclass
class CodePattern:
    """Represents a specific coding pattern found in the codebase."""

    pattern_type: str
    pattern_value: str
    frequency: int
    confidence: float
    context: Dict[str, Any]
    file_paths: List[str]


@dataclass
class AIInsight:
    """Represents AI-generated insights about coding patterns."""

    insight_type: (
        str  # 'style_preference', 'best_practice', 'anti_pattern', 'suggestion'
    )
    description: str
    confidence: float
    evidence: List[str]
    recommendation: Optional[str]


@dataclass
class FunctionSignature:
    """Represents a function signature pattern."""

    name: str
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    decorators: List[str]
    complexity: int
    line_count: int
    file_path: str


@dataclass
class ClassStructure:
    """Represents a class structure pattern."""

    name: str
    base_classes: List[str]
    methods: List[str]
    properties: List[str]
    decorators: List[str]
    docstring: Optional[str]
    file_path: str


@dataclass
class ImportPattern:
    """Represents import/dependency patterns."""

    module_name: str
    import_type: str  # 'import', 'from_import', 'relative'
    alias: Optional[str]
    frequency: int
    files: Set[str]


@dataclass
class NamingConvention:
    """Represents naming convention patterns."""

    convention_type: str  # 'snake_case', 'camelCase', 'PascalCase', etc.
    pattern: str
    frequency: int
    context: str  # 'function', 'variable', 'class', 'constant'


@dataclass
class CodeAnalysisResult:
    """Complete analysis result for a codebase."""

    language: Language
    total_files: int
    total_lines: int
    functions: List[FunctionSignature]
    classes: List[ClassStructure]
    imports: List[ImportPattern]
    naming_conventions: List[NamingConvention]
    patterns: List[CodePattern]
    complexity_metrics: Dict[str, float]
    api_usage: Dict[str, int]
    style_preferences: Dict[str, Any]
    ai_insights: List[AIInsight]  # New field for AI insights


class CodeParser:
    """
    Enhanced code parser with AI-powered analysis capabilities.
    """

    def __init__(self, gemini_api_key: Optional[str] = None, debug: bool = False):
        self.debug = debug
        self.supported_extensions = {
            ".py": Language.PYTHON,
            ".js": Language.JAVASCRIPT,
            ".ts": Language.TYPESCRIPT,
            ".tsx": Language.TYPESCRIPT,
            ".jsx": Language.JAVASCRIPT,
            ".java": Language.JAVA,
            ".cpp": Language.CPP,
            ".cc": Language.CPP,
            ".cxx": Language.CPP,
            ".c": Language.C,
            ".go": Language.GO,
            ".rs": Language.RUST,
        }

        # Pattern recognition settings
        self.min_pattern_frequency = 3
        self.confidence_threshold = 0.7

        # Cache for parsed results
        self._file_cache = {}
        self._analysis_cache = {}

        # Initialize Gemini client if available and API key provided
        self.gemini_client = None
        if gemini_api_key:
            try:
                self.gemini_client = genai.Client(api_key=gemini_api_key)
                print("🤖 Gemini AI integration enabled")
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini client: {e}")
        else:
            # Try to get API key from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    self.gemini_client = genai.Client(api_key=api_key)
                    print("🤖 Gemini AI integration enabled (from env)")
                except Exception as e:
                    print(f"⚠️ Failed to initialize Gemini client: {e}")

    def analyze_with_ai(self, code_sample: str, language: Language) -> List[AIInsight]:
        """Use Gemini AI to analyze code patterns and provide insights."""
        if not self.gemini_client:
            return []

        try:
            prompt = f"""
            Analyze this {language.value} code sample and provide insights about:
            1. Coding style preferences and patterns
            2. Best practices being followed or violated
            3. Code organization and structure
            4. Potential improvements or suggestions

            Code sample:
            ```{language.value}
            {code_sample[:2000]}  # Limit to first 2000 chars
            ```

            Please provide insights in JSON format with the following structure:
            {{
                "insights": [
                    {{
                        "type": "style_preference|best_practice|anti_pattern|suggestion",
                        "description": "Brief description of the insight",
                        "confidence": 0.8,
                        "evidence": ["line of code or pattern that supports this insight"],
                        "recommendation": "optional recommendation for improvement"
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
                    max_output_tokens=1000,
                ),
            )

            # Parse the JSON response
            try:
                response_text = response.text or "{}"
                result = json.loads(response_text)
                insights = []

                for insight_data in result.get("insights", []):
                    insights.append(
                        AIInsight(
                            insight_type=insight_data.get("type", "suggestion"),
                            description=insight_data.get("description", ""),
                            confidence=float(insight_data.get("confidence", 0.5)),
                            evidence=insight_data.get("evidence", []),
                            recommendation=insight_data.get("recommendation"),
                        )
                    )

                return insights

            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse AI response as JSON")
                return []

        except Exception as e:
            print(f"⚠️ AI analysis failed: {e}")
            return []

    def detect_advanced_patterns(
        self, content: str, language: Language, file_path: str
    ) -> List[CodePattern]:
        """Detect advanced coding patterns using regex and heuristics."""
        patterns = []

        if language == Language.PYTHON:
            patterns.extend(self._detect_python_patterns(content, file_path))
        elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
            patterns.extend(self._detect_javascript_patterns(content, file_path))

        return patterns

    def _detect_python_patterns(
        self, content: str, file_path: str
    ) -> List[CodePattern]:
        """Detect Python-specific patterns."""
        patterns = []

        # Detect decorator usage patterns
        decorator_matches = re.findall(r"@(\w+)", content)
        for decorator in set(decorator_matches):
            frequency = decorator_matches.count(decorator)
            patterns.append(
                CodePattern(
                    pattern_type="decorator_usage",
                    pattern_value=decorator,
                    frequency=frequency,
                    confidence=0.9,
                    context={"language": "python", "usage_type": "decorator"},
                    file_paths=[file_path],
                )
            )

        # Detect comprehension patterns
        list_comp_count = len(re.findall(r"\[.*for.*in.*\]", content))
        if list_comp_count > 0:
            patterns.append(
                CodePattern(
                    pattern_type="comprehension_usage",
                    pattern_value="list_comprehension",
                    frequency=list_comp_count,
                    confidence=0.8,
                    context={"language": "python", "comprehension_type": "list"},
                    file_paths=[file_path],
                )
            )

        # Detect exception handling patterns
        try_except_pattern = re.findall(r"try:(.*?)except\s+(\w+)", content, re.DOTALL)
        if try_except_pattern:
            patterns.append(
                CodePattern(
                    pattern_type="exception_handling",
                    pattern_value="try_except",
                    frequency=len(try_except_pattern),
                    confidence=0.9,
                    context={
                        "language": "python",
                        "pattern": "structured_error_handling",
                    },
                    file_paths=[file_path],
                )
            )

        # Detect type hints usage
        type_hints = re.findall(
            r":\s*([A-Za-z_][A-Za-z0-9_]*(?:\[.*?\])?)\s*=?", content
        )
        if type_hints:
            patterns.append(
                CodePattern(
                    pattern_type="type_annotations",
                    pattern_value="python_type_hints",
                    frequency=len(type_hints),
                    confidence=0.8,
                    context={"language": "python", "modern_python": True},
                    file_paths=[file_path],
                )
            )

        return patterns

    def _detect_javascript_patterns(
        self, content: str, file_path: str
    ) -> List[CodePattern]:
        """Detect JavaScript/TypeScript-specific patterns."""
        patterns = []

        # Detect arrow function usage
        arrow_functions = re.findall(r"=>", content)
        if arrow_functions:
            patterns.append(
                CodePattern(
                    pattern_type="function_style",
                    pattern_value="arrow_functions",
                    frequency=len(arrow_functions),
                    confidence=0.8,
                    context={"language": "javascript", "modern_syntax": True},
                    file_paths=[file_path],
                )
            )

        # Detect async/await pattern
        async_count = len(re.findall(r"\basync\b", content))
        await_count = len(re.findall(r"\bawait\b", content))
        if async_count > 0 or await_count > 0:
            patterns.append(
                CodePattern(
                    pattern_type="async_pattern",
                    pattern_value="async_await",
                    frequency=max(async_count, await_count),
                    confidence=0.9,
                    context={"language": "javascript", "async_programming": True},
                    file_paths=[file_path],
                )
            )

        return patterns

    def detect_language(self, file_path: str) -> Language:
        """Detect the programming language of a file based on its extension."""
        extension = Path(file_path).suffix.lower()
        return self.supported_extensions.get(extension, Language.UNKNOWN)

    def parse_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse a single file and extract patterns."""
        try:
            # Check cache first
            file_hash = self._get_file_hash(file_path)
            if (
                file_path in self._file_cache
                and self._file_cache[file_path]["hash"] == file_hash
            ):
                return self._file_cache[file_path]["data"]

            language = self.detect_language(file_path)
            if language == Language.UNKNOWN:
                return None

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            file_data = {
                "path": file_path,
                "language": language,
                "content": content,
                "size": len(content),
                "lines": len(content.splitlines()),
                "patterns": [],
                "functions": [],
                "classes": [],
                "imports": [],
            }

            # Language-specific parsing
            if language == Language.PYTHON:
                file_data.update(self._parse_python_file(content, file_path))

            # Detect advanced patterns
            file_data["patterns"] = self.detect_advanced_patterns(
                content, language, file_path
            )

            # Cache the result
            self._file_cache[file_path] = {"hash": file_hash, "data": file_data}

            return file_data

        except Exception as e:
            if self.debug:
                print(f"⚠️ Error parsing {file_path}: {e}")
            return None

    def _parse_python_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse Python-specific elements from a file."""
        result = {
            "functions": [],
            "classes": [],
            "imports": [],
            "complexity_metrics": {},
            "naming_patterns": [],
        }

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    result["functions"].append(
                        self._extract_function_info(node, file_path)
                    )
                elif isinstance(node, ast.ClassDef):
                    result["classes"].append(self._extract_class_info(node, file_path))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    result["imports"].extend(self._extract_import_info(node))

            # Calculate complexity metrics
            result["complexity_metrics"] = self._calculate_complexity_metrics(tree)

            # Analyze naming patterns
            result["naming_patterns"] = self._analyze_naming_patterns(tree)

        except SyntaxError as e:
            if self.debug:
                print(f"⚠️ Syntax error in {file_path}: {e}")
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error parsing Python file {file_path}: {e}")

        return result

    def _extract_function_info(
        self, node: ast.FunctionDef, file_path: str
    ) -> FunctionSignature:
        """Extract function signature information."""
        return FunctionSignature(
            name=node.name,
            parameters=[arg.arg for arg in node.args.args],
            return_type=ast.unparse(node.returns) if node.returns else None,
            docstring=ast.get_docstring(node),
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            complexity=self._calculate_cyclomatic_complexity(node),
            line_count=node.end_lineno - node.lineno + 1 if node.end_lineno else 1,
            file_path=file_path,
        )

    def _extract_class_info(self, node: ast.ClassDef, file_path: str) -> ClassStructure:
        """Extract class structure information."""
        methods = []
        properties = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        properties.append(target.id)

        return ClassStructure(
            name=node.name,
            base_classes=[ast.unparse(base) for base in node.bases],
            methods=methods,
            properties=properties,
            decorators=[ast.unparse(dec) for dec in node.decorator_list],
            docstring=ast.get_docstring(node),
            file_path=file_path,
        )

    def _extract_import_info(self, node) -> List[Dict[str, Any]]:
        """Extract import information."""
        imports = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {"module": alias.name, "alias": alias.asname, "type": "import"}
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(
                    {
                        "module": f"{module}.{alias.name}" if module else alias.name,
                        "alias": alias.asname,
                        "type": "from_import",
                        "level": node.level,
                    }
                )

        return imports

    def _calculate_complexity_metrics(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate various complexity metrics."""
        metrics = {
            "cyclomatic_complexity": 0.0,
            "cognitive_complexity": 0.0,
            "lines_of_code": 0.0,
            "logical_lines": 0.0,
        }

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                metrics["cyclomatic_complexity"] += 1.0
            elif isinstance(node, ast.FunctionDef):
                metrics["logical_lines"] += 1.0

        return metrics

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _analyze_naming_patterns(self, tree: ast.AST) -> List[NamingConvention]:
        """Analyze naming patterns in the code."""
        patterns = []
        names = {"functions": [], "variables": [], "classes": [], "constants": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                names["classes"].append(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id.isupper():
                    names["constants"].append(node.id)
                else:
                    names["variables"].append(node.id)

        # Analyze patterns for each category
        for category, name_list in names.items():
            if name_list:
                snake_case = sum(
                    1 for name in name_list if "_" in name and name.islower()
                )
                camel_case = sum(
                    1
                    for name in name_list
                    if any(c.isupper() for c in name[1:]) and "_" not in name
                )
                pascal_case = sum(
                    1
                    for name in name_list
                    if name[0].isupper() and any(c.isupper() for c in name[1:])
                )

                total = len(name_list)
                if snake_case / total > 0.6:
                    patterns.append(
                        NamingConvention(
                            "snake_case", "snake_case", snake_case, category
                        )
                    )
                elif camel_case / total > 0.6:
                    patterns.append(
                        NamingConvention("camelCase", "camelCase", camel_case, category)
                    )
                elif pascal_case / total > 0.6:
                    patterns.append(
                        NamingConvention(
                            "PascalCase", "PascalCase", pascal_case, category
                        )
                    )

        return patterns

    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash for caching."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

    def parse_codebase(self, directory_path: str) -> CodeAnalysisResult:
        """Parse an entire codebase and return comprehensive analysis."""
        print(f"🔍 Analyzing codebase: {directory_path}")

        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Collect all files
        all_files = []
        for pattern in [
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.tsx",
            "**/*.jsx",
            "**/*.java",
            "**/*.cpp",
            "**/*.c",
            "**/*.go",
            "**/*.rs",
        ]:
            all_files.extend(directory.glob(pattern))

        # Filter out common ignore patterns
        ignore_patterns = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "build",
            "dist",
        }
        filtered_files = [
            f for f in all_files if not any(part in ignore_patterns for part in f.parts)
        ]

        print(f"📊 Found {len(filtered_files)} files to analyze")

        # Analyze files
        all_functions = []
        all_classes = []
        all_imports = []
        all_patterns = []
        all_naming_conventions = []
        api_usage = Counter()
        complexity_metrics = defaultdict(list)

        language_counter = Counter()
        total_lines = 0
        processed_files = 0

        for file_path in filtered_files:
            file_data = self.parse_file(str(file_path))
            if file_data:
                processed_files += 1
                language_counter[file_data["language"].value] += 1
                total_lines += file_data["lines"]

                all_functions.extend(file_data.get("functions", []))
                all_classes.extend(file_data.get("classes", []))
                all_imports.extend(file_data.get("imports", []))
                all_patterns.extend(file_data.get("patterns", []))
                all_naming_conventions.extend(file_data.get("naming_patterns", []))

                # Update complexity metrics
                for metric, value in file_data.get("complexity_metrics", {}).items():
                    complexity_metrics[metric].append(value)

        # Determine primary language
        primary_language = (
            Language(language_counter.most_common(1)[0][0])
            if language_counter
            else Language.UNKNOWN
        )

        # Process imports into ImportPattern objects
        import_patterns = []
        import_counter = Counter()
        for imp in all_imports:
            module = imp["module"]
            import_counter[module] += 1

        for module, freq in import_counter.items():
            import_patterns.append(
                ImportPattern(
                    module_name=module,
                    import_type="import",
                    alias=None,
                    frequency=freq,
                    files=set(),
                )
            )

        # Calculate average complexity metrics
        avg_complexity = {}
        for metric, values in complexity_metrics.items():
            avg_complexity[metric] = sum(values) / len(values) if values else 0

        # Detect style preferences
        style_preferences = self._detect_style_preferences(filtered_files)

        # Generate AI insights if available
        ai_insights = []
        if self.gemini_client and all_functions:
            # Create a sample of code for AI analysis
            sample_code = self._create_code_sample(all_functions[:5], all_classes[:3])
            ai_insights = self.analyze_with_ai(sample_code, primary_language)

        print(
            f"✅ Analysis complete: {processed_files} files, {total_lines} lines of code"
        )

        return CodeAnalysisResult(
            language=primary_language,
            total_files=processed_files,
            total_lines=total_lines,
            functions=all_functions,
            classes=all_classes,
            imports=import_patterns,
            naming_conventions=all_naming_conventions,
            patterns=all_patterns,
            complexity_metrics=avg_complexity,
            api_usage=dict(api_usage),
            style_preferences=style_preferences,
            ai_insights=ai_insights,
        )

    def _detect_style_preferences(self, files: List[Path]) -> Dict[str, Any]:
        """Detect coding style preferences from the files."""
        preferences = {
            "indentation": {"spaces": 0, "tabs": 0, "mixed": 0},
            "quotes": {"single": 0, "double": 0},
            "line_length": [],
            "import_style": {"grouped": 0, "sorted": 0},
        }

        for file_path in files[:10]:  # Sample first 10 files
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.splitlines()

                    for line in lines:
                        if line.strip():
                            # Check indentation
                            if line.startswith("    "):
                                preferences["indentation"]["spaces"] += 1
                            elif line.startswith("\t"):
                                preferences["indentation"]["tabs"] += 1

                            # Check quotes
                            preferences["quotes"]["single"] += line.count("'")
                            preferences["quotes"]["double"] += line.count('"')

                            # Track line length
                            preferences["line_length"].append(len(line))
            except:
                continue

        # Calculate averages and preferences
        total_indent = sum(preferences["indentation"].values())
        if total_indent > 0:
            preferences["preferred_indentation"] = max(
                preferences["indentation"], key=preferences["indentation"].get
            )

        total_quotes = sum(preferences["quotes"].values())
        if total_quotes > 0:
            preferences["preferred_quotes"] = max(
                preferences["quotes"], key=preferences["quotes"].get
            )

        if preferences["line_length"]:
            preferences["average_line_length"] = sum(preferences["line_length"]) / len(
                preferences["line_length"]
            )

        return preferences

    def _create_code_sample(
        self, functions: List[FunctionSignature], classes: List[ClassStructure]
    ) -> str:
        """Create a representative code sample for AI analysis."""
        sample_parts = []

        # Add a few function samples
        for func in functions[:3]:
            sample_parts.append(f"def {func.name}({', '.join(func.parameters)}):")
            if func.docstring:
                sample_parts.append(f'    """{func.docstring[:100]}"""')
            sample_parts.append("    # function body...")
            sample_parts.append("")

        # Add a class sample
        for cls in classes[:2]:
            sample_parts.append(f"class {cls.name}:")
            if cls.docstring:
                sample_parts.append(f'    """{cls.docstring[:100]}"""')
            for method in cls.methods[:2]:
                sample_parts.append(f"    def {method}(self):")
                sample_parts.append("        pass")
            sample_parts.append("")

        return "\n".join(sample_parts)

    def export_to_json(self, result: CodeAnalysisResult, output_file: str) -> None:
        """Export analysis results to JSON file."""
        try:
            # Convert dataclass to dict with proper serialization
            def convert_sets_to_lists(obj):
                """Convert sets to lists for JSON serialization."""
                if isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {k: convert_sets_to_lists(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_sets_to_lists(item) for item in obj]
                return obj

            data = {
                "language": result.language.value,
                "total_files": result.total_files,
                "total_lines": result.total_lines,
                "functions": [
                    convert_sets_to_lists(asdict(f)) for f in result.functions
                ],
                "classes": [convert_sets_to_lists(asdict(c)) for c in result.classes],
                "imports": [convert_sets_to_lists(asdict(i)) for i in result.imports],
                "naming_conventions": [
                    convert_sets_to_lists(asdict(n)) for n in result.naming_conventions
                ],
                "patterns": [convert_sets_to_lists(asdict(p)) for p in result.patterns],
                "complexity_metrics": result.complexity_metrics,
                "api_usage": result.api_usage,
                "style_preferences": result.style_preferences,
                "ai_insights": [
                    convert_sets_to_lists(asdict(ai)) for ai in result.ai_insights
                ],
                "timestamp": str(Path(output_file).stem),
                "analysis_version": "2.0",
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ Error exporting to JSON: {e}")
            raise
