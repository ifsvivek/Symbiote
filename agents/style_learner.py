"""
🧬 Symbiote Style Learning Agent

This module implements the intelligent style learning agent that uses LangChain
to analyze code patterns and learn developer preferences over time.

Key Features:
- Pattern-based learning using LangChain agents
- Memory and persistence for learned preferences
- Integration with Gemini API for intelligent insights
- Adaptive recommendations based on historical patterns

Author: Vivek Sharma
License: MIT
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from datetime import datetime
import json
import pickle

# LangChain imports
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import Document

# Google GenAI imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

# Vector store for pattern memory
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

# Local imports
from utils.code_parser import (
    CodeAnalysisResult,
    CodePattern,
    FunctionSignature,
    ClassStructure,
)
from utils.tools import SymbioteTools


@dataclass
class StyleMemory:
    """Represents learned style preferences and patterns."""

    developer_id: str
    preferences: Dict[str, Any]
    patterns: List[CodePattern]
    confidence_scores: Dict[str, float]
    learning_sessions: int
    last_updated: datetime


@dataclass
class LearningInsight:
    """Represents an insight learned from code analysis."""

    category: str
    pattern: str
    confidence: float
    frequency: int
    evidence: List[str]
    recommendation: str


class StyleLearnerAgent:
    """
    Intelligent agent that learns coding style preferences using LangChain.

    This agent:
    1. Analyzes code patterns across multiple sessions
    2. Builds a memory of developer preferences
    3. Generates intelligent recommendations
    4. Adapts based on feedback and new code samples
    """

    def __init__(
        self,
        gemini_api_key: str,
        memory_path: str = "symbiote_memory",
        debug: bool = False,
    ):
        self.debug = debug
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(exist_ok=True)

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=SecretStr(gemini_api_key),
            temperature=0.2,
            max_output_tokens=2048,
        )

        # Initialize embeddings for pattern similarity
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-exp-03-07", google_api_key=SecretStr(gemini_api_key)
        )

        # Initialize tools system
        self.tools = SymbioteTools(gemini_api_key, debug=debug)

        # Initialize conversation memory (using simple in-memory storage)
        self.conversation_history = []

        # Load or initialize style memory
        self.style_memory = self._load_style_memory()

        # Initialize vector store for pattern similarity
        self.pattern_store = self._initialize_pattern_store()

        # Create the learning agent
        self.agent = self._create_learning_agent()

        if self.debug:
            print(f"🧠 Style Learning Agent initialized")
            print(f"📚 Memory path: {self.memory_path}")
            print(f"🔄 Learning sessions: {self.style_memory.learning_sessions}")
            print(f"🛠️ Tools available: {len(self.tools.tools)}")

    def _serialize_style_memory(self, memory: StyleMemory) -> Dict[str, Any]:
        """Convert StyleMemory to JSON-serializable format."""
        return {
            "developer_id": memory.developer_id,
            "preferences": memory.preferences,
            "patterns": [asdict(pattern) for pattern in memory.patterns],
            "confidence_scores": memory.confidence_scores,
            "learning_sessions": memory.learning_sessions,
            "last_updated": memory.last_updated.isoformat(),
        }

    def _deserialize_style_memory(self, data: Dict[str, Any]) -> StyleMemory:
        """Convert JSON data back to StyleMemory object."""
        return StyleMemory(
            developer_id=data["developer_id"],
            preferences=data["preferences"],
            patterns=[CodePattern(**pattern) for pattern in data["patterns"]],
            confidence_scores=data["confidence_scores"],
            learning_sessions=data["learning_sessions"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
        )

    def _load_style_memory(self) -> StyleMemory:
        """Load existing style memory or create new one."""
        json_file = self.memory_path / "style_memory.json"
        pkl_file = self.memory_path / "style_memory.pkl"

        # Try to load from JSON first (new format)
        if json_file.exists():
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                return self._deserialize_style_memory(data)
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Failed to load JSON memory: {e}")

        # Fallback to pickle (old format) with warning suppression
        if pkl_file.exists():
            try:
                with open(pkl_file, "rb") as f:
                    memory = pickle.load(f)
                # Convert to JSON format for future use
                self._save_style_memory_json(memory)
                if self.debug:
                    print("🔄 Converted pickle memory to JSON format")
                return memory
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Failed to load pickle memory: {e}")

        # Create new memory
        return StyleMemory(
            developer_id="default",
            preferences={},
            patterns=[],
            confidence_scores={},
            learning_sessions=0,
            last_updated=datetime.now(),
        )

    def _save_style_memory_json(self, memory: Optional[StyleMemory] = None) -> None:
        """Save style memory to JSON format."""
        try:
            memory_to_save = memory or self.style_memory
            json_file = self.memory_path / "style_memory.json"

            with open(json_file, "w") as f:
                json.dump(self._serialize_style_memory(memory_to_save), f, indent=2)

            if self.debug:
                print(f"💾 Style memory saved to JSON")

        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to save JSON memory: {e}")

    def _save_style_memory(self) -> None:
        """Save current style memory to disk (using JSON)."""
        self._save_style_memory_json()

    def _initialize_pattern_store(self) -> Optional[FAISS]:
        """Initialize vector store for pattern similarity search."""
        try:
            vector_store_path = self.memory_path / "pattern_vectors"

            if vector_store_path.exists():
                # Load existing vector store with safe deserialization
                return FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,  # We trust our own saved files
                )
            else:
                # Create new vector store with dummy data
                dummy_docs = [Document(page_content="dummy", metadata={})]
                store = FAISS.from_documents(dummy_docs, self.embeddings)
                store.save_local(str(vector_store_path))
                return store

        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to initialize pattern store: {e}")
            return None

    def _create_learning_agent(self) -> AgentExecutor:
        """Create the LangChain learning agent with tools."""

        # Use comprehensive tools from SymbioteTools
        tools = self.tools.tools

        # Create the learning prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert code style analyst and learning agent. Your job is to:

1. Analyze code patterns and extract meaningful insights about developer preferences
2. Compare new patterns with previously learned preferences
3. Generate adaptive recommendations based on accumulated knowledge
4. Update learned preferences when you find consistent new patterns

You have access to the following powerful tools:
{tools}

Tool names: {tool_names}

Use these tools systematically to build a comprehensive understanding of the developer's style.
Focus on using:
- analyze_naming_conventions for naming pattern analysis
- evaluate_code_structure for structural patterns
- assess_code_complexity for complexity insights
- extract_style_preferences for preference learning
- generate_ai_insights for AI-powered analysis

Current learned preferences: {learned_preferences}
Learning sessions completed: {learning_sessions}

Always be thorough in your analysis and provide specific, actionable insights.""",
                ),
                ("human", "{input}"),
                ("assistant", "{agent_scratchpad}"),
            ]
        )

        # Create the agent
        agent = create_react_agent(self.llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.debug,
            max_iterations=15,
            handle_parsing_errors=True,
        )

    def _analyze_naming_patterns(self, code_analysis: str) -> str:
        """Tool: Analyze naming convention patterns."""
        try:
            # Parse the analysis if it's a JSON string
            if isinstance(code_analysis, str):
                data = json.loads(code_analysis)
            else:
                data = code_analysis

            naming_analysis = {
                "function_names": [f["name"] for f in data.get("functions", [])],
                "class_names": [c["name"] for c in data.get("classes", [])],
                "naming_conventions": data.get("naming_conventions", []),
            }

            # Analyze patterns
            function_patterns = self._detect_naming_pattern(
                naming_analysis["function_names"]
            )
            class_patterns = self._detect_naming_pattern(naming_analysis["class_names"])

            result = {
                "function_naming": function_patterns,
                "class_naming": class_patterns,
                "consistency_score": self._calculate_naming_consistency(
                    naming_analysis
                ),
                "recommendations": self._generate_naming_recommendations(
                    function_patterns, class_patterns
                ),
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"Error analyzing naming patterns: {str(e)}"

    def _evaluate_code_structure(self, code_analysis: str) -> str:
        """Tool: Evaluate code organization and structure."""
        try:
            if isinstance(code_analysis, str):
                data = json.loads(code_analysis)
            else:
                data = code_analysis

            structure_metrics = {
                "avg_function_length": self._calculate_avg_function_length(
                    data.get("functions", [])
                ),
                "class_complexity": self._calculate_class_complexity(
                    data.get("classes", [])
                ),
                "import_organization": self._analyze_import_organization(
                    data.get("imports", [])
                ),
                "file_organization": self._analyze_file_organization(data),
            }

            return json.dumps(structure_metrics, indent=2)

        except Exception as e:
            return f"Error evaluating code structure: {str(e)}"

    def _assess_complexity_patterns(self, code_analysis: str) -> str:
        """Tool: Assess complexity and design patterns."""
        try:
            if isinstance(code_analysis, str):
                data = json.loads(code_analysis)
            else:
                data = data

            complexity_assessment = {
                "avg_complexity": data.get("complexity_metrics", {}).get(
                    "cyclomatic_complexity", 0
                ),
                "complexity_distribution": self._analyze_complexity_distribution(
                    data.get("functions", [])
                ),
                "design_patterns": self._detect_design_patterns(data),
                "code_smells": self._detect_code_smells(data),
            }

            return json.dumps(complexity_assessment, indent=2)

        except Exception as e:
            return f"Error assessing complexity patterns: {str(e)}"

    def _find_similar_patterns(self, pattern_description: str) -> str:
        """Tool: Find similar patterns from previous sessions."""
        try:
            if not self.pattern_store:
                return "No pattern store available"

            # Search for similar patterns
            similar_docs = self.pattern_store.similarity_search(
                pattern_description, k=5
            )

            similar_patterns = []
            for doc in similar_docs:
                similar_patterns.append(
                    {"pattern": doc.page_content, "metadata": doc.metadata}
                )

            return json.dumps(similar_patterns, indent=2)

        except Exception as e:
            return f"Error finding similar patterns: {str(e)}"

    def _update_preferences(self, preference_update: str) -> str:
        """Tool: Update learned preferences."""
        try:
            update_data = json.loads(preference_update)

            # Update style memory
            for category, preference in update_data.items():
                if category in self.style_memory.preferences:
                    # Update existing preference with weighted average
                    current = self.style_memory.preferences[category]
                    if isinstance(current, dict) and isinstance(preference, dict):
                        for key, value in preference.items():
                            if key in current:
                                # Weighted update (new evidence gets 30% weight)
                                if isinstance(value, (int, float)):
                                    current[key] = current[key] * 0.7 + value * 0.3
                                else:
                                    current[key] = value  # Override non-numeric values
                            else:
                                current[key] = value
                    else:
                        self.style_memory.preferences[category] = preference
                else:
                    self.style_memory.preferences[category] = preference

            # Update learning session count and timestamp
            self.style_memory.learning_sessions += 1
            self.style_memory.last_updated = datetime.now()

            # Save updated memory
            self._save_style_memory()

            return f"Preferences updated successfully. Learning sessions: {self.style_memory.learning_sessions}"

        except Exception as e:
            return f"Error updating preferences: {str(e)}"

    def _detect_naming_pattern(self, names: List[str]) -> Dict[str, Any]:
        """Detect naming convention patterns."""
        patterns = {
            "snake_case": 0,
            "camelCase": 0,
            "PascalCase": 0,
            "CONSTANT_CASE": 0,
        }

        for name in names:
            if "_" in name and name.islower():
                patterns["snake_case"] += 1
            elif name[0].islower() and any(c.isupper() for c in name[1:]):
                patterns["camelCase"] += 1
            elif name[0].isupper():
                if "_" in name and name.isupper():
                    patterns["CONSTANT_CASE"] += 1
                else:
                    patterns["PascalCase"] += 1

        total = sum(patterns.values())
        if total > 0:
            return {pattern: count / total for pattern, count in patterns.items()}
        return patterns

    def _calculate_naming_consistency(self, naming_analysis: Dict[str, Any]) -> float:
        """Calculate naming consistency score."""
        # Implementation for consistency scoring
        return 0.8  # Placeholder

    def _generate_naming_recommendations(
        self, function_patterns: Dict, class_patterns: Dict
    ) -> List[str]:
        """Generate naming recommendations."""
        recommendations = []

        if function_patterns:
            func_dominant = max(function_patterns, key=lambda x: function_patterns[x])
            if function_patterns[func_dominant] > 0.7:
                recommendations.append(
                    f"Maintain {func_dominant} for functions ({function_patterns[func_dominant]:.1%} consistency)"
                )

        if class_patterns:
            class_dominant = max(class_patterns, key=lambda x: class_patterns[x])
            if class_patterns[class_dominant] > 0.7:
                recommendations.append(
                    f"Maintain {class_dominant} for classes ({class_patterns[class_dominant]:.1%} consistency)"
                )

        return recommendations

    def _calculate_avg_function_length(self, functions: List[Dict]) -> float:
        """Calculate average function length."""
        if not functions:
            return 0.0
        return sum(f.get("line_count", 0) for f in functions) / len(functions)

    def _calculate_class_complexity(self, classes: List[Dict]) -> Dict[str, Any]:
        """Calculate class complexity metrics."""
        if not classes:
            return {"avg_methods": 0, "avg_properties": 0}

        return {
            "avg_methods": sum(len(c.get("methods", [])) for c in classes)
            / len(classes),
            "avg_properties": sum(len(c.get("properties", [])) for c in classes)
            / len(classes),
        }

    def _analyze_import_organization(self, imports: List[Dict]) -> Dict[str, Any]:
        """Analyze import organization patterns."""
        return {
            "total_imports": len(imports),
            "unique_modules": len(set(imp.get("module_name", "") for imp in imports)),
            "relative_imports": sum(
                1 for imp in imports if imp.get("type") == "from_import"
            ),
        }

    def _analyze_file_organization(self, data: Dict) -> Dict[str, Any]:
        """Analyze file organization patterns."""
        return {
            "functions_per_file": len(data.get("functions", []))
            / max(data.get("total_files", 1), 1),
            "classes_per_file": len(data.get("classes", []))
            / max(data.get("total_files", 1), 1),
        }

    def _analyze_complexity_distribution(self, functions: List[Dict]) -> Dict[str, int]:
        """Analyze complexity distribution."""
        distribution = {"low": 0, "medium": 0, "high": 0}

        for func in functions:
            complexity = func.get("complexity", 1)
            if complexity <= 5:
                distribution["low"] += 1
            elif complexity <= 10:
                distribution["medium"] += 1
            else:
                distribution["high"] += 1

        return distribution

    def _detect_design_patterns(self, data: Dict) -> List[str]:
        """Detect common design patterns."""
        patterns = []

        # Simple pattern detection based on class names and structure
        class_names = [c.get("name", "") for c in data.get("classes", [])]

        if any("Factory" in name for name in class_names):
            patterns.append("Factory Pattern")
        if any("Observer" in name for name in class_names):
            patterns.append("Observer Pattern")
        if any("Builder" in name for name in class_names):
            patterns.append("Builder Pattern")

        return patterns

    def _detect_code_smells(self, data: Dict) -> List[str]:
        """Detect potential code smells."""
        smells = []

        # Check for large functions
        large_functions = [
            f for f in data.get("functions", []) if f.get("line_count", 0) > 50
        ]
        if large_functions:
            smells.append(
                f"Large functions detected: {len(large_functions)} functions > 50 lines"
            )

        # Check for high complexity
        complex_functions = [
            f for f in data.get("functions", []) if f.get("complexity", 0) > 10
        ]
        if complex_functions:
            smells.append(
                f"High complexity functions: {len(complex_functions)} functions > 10 complexity"
            )

        return smells

    def learn_from_analysis(
        self, analysis_result: CodeAnalysisResult
    ) -> Dict[str, Any]:
        """
        Learn from a code analysis result.

        Args:
            analysis_result: The analysis result to learn from

        Returns:
            Dictionary containing learning insights and updated preferences
        """
        try:
            if self.debug:
                print("🧠 Starting learning session...")

            # Convert analysis result to JSON for the agent
            analysis_data = {
                "language": analysis_result.language.value,
                "total_files": analysis_result.total_files,
                "total_lines": analysis_result.total_lines,
                "functions": [asdict(f) for f in analysis_result.functions],
                "classes": [asdict(c) for c in analysis_result.classes],
                "imports": [asdict(i) for i in analysis_result.imports],
                "naming_conventions": [
                    asdict(nc) for nc in analysis_result.naming_conventions
                ],
                "patterns": [asdict(p) for p in analysis_result.patterns],
                "complexity_metrics": analysis_result.complexity_metrics,
                "style_preferences": analysis_result.style_preferences,
            }

            # Create learning prompt
            learning_prompt = f"""
            Analyze this codebase and learn from the developer's style patterns:
            
            {json.dumps(analysis_data, indent=2)}
            
            Please:
            1. Analyze the naming patterns and compare with previous preferences
            2. Evaluate the code structure and organization
            3. Assess complexity patterns and design choices
            4. Find similar patterns from previous learning sessions
            5. Update preferences based on consistent new evidence
            
            Provide comprehensive insights about the developer's evolving style preferences.
            """

            # Run the agent
            result = self.agent.invoke(
                {
                    "input": learning_prompt,
                    "learned_preferences": json.dumps(
                        self.style_memory.preferences, indent=2
                    ),
                    "learning_sessions": self.style_memory.learning_sessions,
                }
            )

            # Store patterns in vector store for future similarity search
            if self.pattern_store:
                pattern_docs = []
                for pattern in analysis_result.patterns:
                    doc = Document(
                        page_content=f"{pattern.pattern_type}: {pattern.pattern_value}",
                        metadata={
                            "frequency": pattern.frequency,
                            "confidence": pattern.confidence,
                            "session": self.style_memory.learning_sessions,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    pattern_docs.append(doc)

                if pattern_docs:
                    self.pattern_store.add_documents(pattern_docs)
                    vector_store_path = self.memory_path / "pattern_vectors"
                    self.pattern_store.save_local(str(vector_store_path))

            learning_result = {
                "agent_output": result.get("output", ""),
                "learned_preferences": self.style_memory.preferences,
                "learning_sessions": self.style_memory.learning_sessions,
                "patterns_stored": len(analysis_result.patterns),
                "timestamp": datetime.now().isoformat(),
            }

            if self.debug:
                print(
                    f"✅ Learning session complete! Total sessions: {self.style_memory.learning_sessions}"
                )

            return learning_result

        except Exception as e:
            if self.debug:
                print(f"❌ Learning session failed: {e}")
            return {
                "error": str(e),
                "learned_preferences": self.style_memory.preferences,
                "learning_sessions": self.style_memory.learning_sessions,
            }

    def get_style_recommendations(
        self, analysis_result: CodeAnalysisResult
    ) -> List[str]:
        """Get personalized style recommendations based on learned preferences."""
        try:
            # Compare current analysis with learned preferences
            recommendations = []

            # Naming convention recommendations
            if "naming" in self.style_memory.preferences:
                naming_prefs = self.style_memory.preferences["naming"]
                # Add logic to compare and recommend based on learned naming patterns

            # Structure recommendations
            if "structure" in self.style_memory.preferences:
                structure_prefs = self.style_memory.preferences["structure"]
                # Add logic for structure recommendations

            # Complexity recommendations
            if "complexity" in self.style_memory.preferences:
                complexity_prefs = self.style_memory.preferences["complexity"]
                # Add logic for complexity recommendations

            return recommendations

        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to generate recommendations: {e}")
            return []

    def export_learned_preferences(self, output_file: str) -> None:
        """Export learned preferences to a JSON file."""
        try:
            export_data = {
                "developer_id": self.style_memory.developer_id,
                "preferences": self.style_memory.preferences,
                "confidence_scores": self.style_memory.confidence_scores,
                "learning_sessions": self.style_memory.learning_sessions,
                "last_updated": self.style_memory.last_updated.isoformat(),
                "export_timestamp": datetime.now().isoformat(),
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)

            if self.debug:
                print(f"📊 Learned preferences exported to: {output_file}")

        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to export preferences: {e}")
