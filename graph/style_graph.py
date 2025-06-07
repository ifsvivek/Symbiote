"""
🧬 Symbiote Style Graph

This module defines the LangGraph-based style analysis graph that processes
code patterns and creates intelligent insights using multiple specialized agents.

Key Components:
- Style Analysis Graph using LangGraph
- Node definitions for different analysis phases
- Edge routing based on analysis results
- Integration with Gemini API for intelligent insights

Author: Vivek Sharma
License: MIT
"""

from typing import Dict, List, Any, Optional, Annotated, TypedDict
from dataclasses import dataclass
import json
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

try:
    from langgraph.checkpoint.memory import MemorySaver

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig

# Google GenAI imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from utils.code_parser import CodeAnalysisResult, CodePattern, AIInsight
from utils.tools import SymbioteTools


class StyleGraphState(TypedDict):
    """State definition for the style analysis graph."""

    analysis_result: CodeAnalysisResult
    current_patterns: List[CodePattern]
    insights: List[AIInsight]
    recommendations: List[str]
    confidence_score: float
    processing_stage: str
    messages: List[BaseMessage]
    errors: List[str]
    iteration_count: int
    max_iterations: int


@dataclass
class StyleRecommendation:
    """Represents a style recommendation with context."""

    category: str  # 'naming', 'structure', 'imports', 'complexity'
    title: str
    description: str
    confidence: float
    evidence: List[str]
    suggested_action: str
    priority: str  # 'high', 'medium', 'low'


class SymbioteStyleGraph:
    """
    LangGraph-based style analysis graph for Symbiote.

    This graph orchestrates multiple analysis phases:
    1. Pattern Analysis - Extract and categorize patterns
    2. Style Evaluation - Assess consistency and preferences
    3. Insight Generation - Generate AI-powered insights
    4. Recommendation Synthesis - Create actionable recommendations
    """

    def __init__(self, gemini_api_key: str, debug: bool = False):
        self.debug = debug
        self.gemini_api_key = gemini_api_key

        # Initialize memory as None first to ensure attribute exists
        self.memory = None

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=gemini_api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )

        # Initialize tools system
        self.tools = SymbioteTools(gemini_api_key, debug=debug)

        # Initialize graph
        self.graph = self._build_graph()

        # Memory for conversation continuity (optional)
        self.memory = None
        if MEMORY_AVAILABLE:
            try:
                from langgraph.checkpoint.memory import MemorySaver

                self.memory = MemorySaver()
                if self.debug:
                    print("💾 Memory saver initialized successfully")
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Memory saver initialization failed: {e}")
                self.memory = None
        else:
            if self.debug:
                print("⚠️ Memory saver not available - running without memory")

        if self.debug:
            print("🕸️ Style Graph initialized with Gemini integration")

    def _build_graph(self):
        """Build the LangGraph workflow for style analysis."""

        # Create the graph
        workflow = StateGraph(StyleGraphState)

        # Add nodes
        workflow.add_node("pattern_analyzer", self._analyze_patterns_node)
        workflow.add_node("style_evaluator", self._evaluate_style_node)
        workflow.add_node("insight_generator", self._generate_insights_node)
        workflow.add_node(
            "recommendation_synthesizer", self._synthesize_recommendations_node
        )
        workflow.add_node("quality_checker", self._check_quality_node)

        # Define the flow
        workflow.set_entry_point("pattern_analyzer")

        workflow.add_edge("pattern_analyzer", "style_evaluator")
        workflow.add_edge("style_evaluator", "insight_generator")
        workflow.add_edge("insight_generator", "recommendation_synthesizer")
        workflow.add_edge("recommendation_synthesizer", "quality_checker")

        # Conditional edges from quality checker
        workflow.add_conditional_edges(
            "quality_checker",
            self._should_continue,
            {
                "continue": "insight_generator",  # Re-process if quality is low
                "end": END,
            },
        )

        return workflow.compile(checkpointer=self.memory if self.memory else None)

    def _analyze_patterns_node(self, state: StyleGraphState) -> StyleGraphState:
        """Analyze and categorize code patterns."""
        try:
            if self.debug:
                print("🔍 Analyzing patterns...")

            result = state["analysis_result"]
            patterns = result.patterns

            # Categorize patterns by type
            categorized_patterns = {
                "naming": [p for p in patterns if "naming" in p.pattern_type],
                "structure": [p for p in patterns if "structure" in p.pattern_type],
                "imports": [p for p in patterns if "import" in p.pattern_type],
                "complexity": [p for p in patterns if "complexity" in p.pattern_type],
                "style": [p for p in patterns if "style" in p.pattern_type],
            }

            # Update state
            state["current_patterns"] = patterns
            state["processing_stage"] = "pattern_analysis_complete"
            state["messages"].append(
                HumanMessage(
                    content=f"Analyzed {len(patterns)} patterns across {len(categorized_patterns)} categories"
                )
            )

            return state

        except Exception as e:
            state["errors"].append(f"Pattern analysis error: {str(e)}")
            return state

    def _evaluate_style_node(self, state: StyleGraphState) -> StyleGraphState:
        """Evaluate style consistency and preferences."""
        try:
            if self.debug:
                print("🎨 Evaluating style consistency...")

            result = state["analysis_result"]

            # Create evaluation prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a code style expert. Analyze the provided code metrics and style preferences to evaluate consistency and quality.
                
                Focus on:
                1. Naming convention consistency
                2. Code organization patterns
                3. Import style consistency
                4. Overall style cohesion
                
                Provide insights in JSON format with confidence scores.""",
                    ),
                    (
                        "human",
                        """Analyze this codebase style data:
                
                Style Preferences: {style_preferences}
                Naming Conventions: {naming_conventions}
                Import Patterns: {import_patterns}
                Total Files: {total_files}
                
                Provide a comprehensive style evaluation.""",
                    ),
                ]
            )

            # Format the data
            style_data = {
                "style_preferences": result.style_preferences,
                "naming_conventions": [
                    nc.convention_type for nc in result.naming_conventions
                ],
                "import_patterns": [
                    ip.module_name for ip in result.imports[:10]
                ],  # Sample
                "total_files": result.total_files,
            }

            # Get LLM evaluation
            chain = prompt | self.llm | JsonOutputParser()
            evaluation = chain.invoke(style_data)

            # Update state
            state["processing_stage"] = "style_evaluation_complete"
            state["messages"].append(
                AIMessage(
                    content=f"Style evaluation complete: {evaluation.get('summary', 'No summary')}"
                )
            )

            return state

        except Exception as e:
            state["errors"].append(f"Style evaluation error: {str(e)}")
            return state

    def _generate_insights_node(self, state: StyleGraphState) -> StyleGraphState:
        """Generate AI-powered insights about the codebase."""
        try:
            if self.debug:
                print("🧠 Generating AI insights...")

            result = state["analysis_result"]

            # Create insight generation prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are an expert code analyst with deep knowledge of software engineering best practices.
                
                Your task is to generate actionable insights about code quality, style consistency, and potential improvements.
                
                Analyze the provided codebase metrics and generate insights in the following categories:
                1. Style Preferences - What patterns indicate the developer's preferred style
                2. Best Practices - What good practices are being followed
                3. Anti-patterns - What problematic patterns should be addressed
                4. Suggestions - Specific recommendations for improvement
                
                Return insights as JSON array with type, description, confidence, evidence, and recommendation fields.""",
                    ),
                    (
                        "human",
                        """Analyze this codebase:
                
                Language: {language}
                Files: {total_files}
                Lines of Code: {total_lines}
                Functions: {function_count}
                Classes: {class_count}
                
                Complexity Metrics: {complexity_metrics}
                Style Preferences: {style_preferences}
                
                Generate comprehensive insights and recommendations.""",
                    ),
                ]
            )

            # Prepare data for analysis
            analysis_data = {
                "language": result.language.value,
                "total_files": result.total_files,
                "total_lines": result.total_lines,
                "function_count": len(result.functions),
                "class_count": len(result.classes),
                "complexity_metrics": result.complexity_metrics,
                "style_preferences": result.style_preferences,
            }

            # Generate insights
            chain = prompt | self.llm | JsonOutputParser()
            insights_data = chain.invoke(analysis_data)

            # Convert to AIInsight objects
            insights = []
            for insight_data in insights_data.get("insights", []):
                insights.append(
                    AIInsight(
                        insight_type=insight_data.get("type", "suggestion"),
                        description=insight_data.get("description", ""),
                        confidence=float(insight_data.get("confidence", 0.5)),
                        evidence=insight_data.get("evidence", []),
                        recommendation=insight_data.get("recommendation"),
                    )
                )

            # Update state
            state["insights"] = insights
            state["processing_stage"] = "insight_generation_complete"
            state["messages"].append(
                AIMessage(content=f"Generated {len(insights)} AI insights")
            )

            return state

        except Exception as e:
            state["errors"].append(f"Insight generation error: {str(e)}")
            return state

    def _synthesize_recommendations_node(
        self, state: StyleGraphState
    ) -> StyleGraphState:
        """Synthesize actionable recommendations from insights."""
        try:
            if self.debug:
                print("💡 Synthesizing recommendations...")

            insights = state["insights"]

            # Create recommendation synthesis prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a software development mentor who provides clear, actionable recommendations.
                
                Based on the provided insights, create specific recommendations that help developers improve their code quality and consistency.
                
                Each recommendation should:
                1. Be specific and actionable
                2. Include rationale and benefits
                3. Have a clear priority level
                4. Provide concrete steps to implement
                
                Return recommendations as a JSON array.""",
                    ),
                    (
                        "human",
                        """Based on these insights, create actionable recommendations:
                
                {insights}
                
                Focus on the most impactful improvements that align with the existing style patterns.""",
                    ),
                ]
            )

            # Format insights for the prompt
            insights_text = "\n".join(
                [
                    f"- {insight.insight_type}: {insight.description} (confidence: {insight.confidence})"
                    for insight in insights
                ]
            )

            # Generate recommendations
            chain = prompt | self.llm | JsonOutputParser()
            recommendations_data = chain.invoke({"insights": insights_text})

            # Extract recommendations
            recommendations = [
                rec.get("description", "")
                for rec in recommendations_data.get("recommendations", [])
            ]

            # Update state
            state["recommendations"] = recommendations
            state["processing_stage"] = "recommendation_synthesis_complete"
            state["messages"].append(
                AIMessage(content=f"Synthesized {len(recommendations)} recommendations")
            )

            return state

        except Exception as e:
            state["errors"].append(f"Recommendation synthesis error: {str(e)}")
            return state

    def _check_quality_node(self, state: StyleGraphState) -> StyleGraphState:
        """Check the quality of generated insights and recommendations."""
        try:
            if self.debug:
                print("✅ Checking output quality...")

            insights = state["insights"]
            recommendations = state["recommendations"]

            # Increment iteration counter
            state["iteration_count"] = state.get("iteration_count", 0) + 1

            # Calculate overall confidence score
            if insights:
                avg_confidence = sum(insight.confidence for insight in insights) / len(
                    insights
                )
            else:
                avg_confidence = 0.0

            # Quality criteria
            min_insights = 3
            min_recommendations = 2
            min_confidence = 0.6

            quality_score = 0.0
            if len(insights) >= min_insights:
                quality_score += 0.4
            if len(recommendations) >= min_recommendations:
                quality_score += 0.3
            if avg_confidence >= min_confidence:
                quality_score += 0.3

            state["confidence_score"] = avg_confidence
            state["processing_stage"] = (
                f"quality_check_complete_iteration_{state['iteration_count']}"
            )

            # Add quality assessment message
            state["messages"].append(
                AIMessage(
                    content=f"Quality check iteration {state['iteration_count']}: {quality_score:.2f} (confidence: {avg_confidence:.2f})"
                )
            )

            return state

        except Exception as e:
            state["errors"].append(f"Quality check error: {str(e)}")
            return state

    def _should_continue(self, state: StyleGraphState) -> str:
        """Determine if the graph should continue processing."""
        confidence = state.get("confidence_score", 0.0)
        insights_count = len(state.get("insights", []))
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)

        # Stop if we've reached max iterations to prevent infinite loops
        if iteration_count >= max_iterations:
            if self.debug:
                print(
                    f"🛑 Stopping after {iteration_count} iterations to prevent recursion"
                )
            return "end"

        # Continue if quality is too low and we haven't exceeded max iterations
        if confidence < 0.5 or insights_count < 3:
            if self.debug:
                print(
                    f"🔄 Quality check failed (confidence: {confidence:.2f}, insights: {insights_count}), continuing iteration {iteration_count + 1}"
                )
            return "continue"

        if self.debug:
            print(
                f"✅ Quality check passed (confidence: {confidence:.2f}, insights: {insights_count})"
            )
        return "end"

    def analyze_codebase(self, analysis_result: CodeAnalysisResult) -> Dict[str, Any]:
        """
        Run the complete style analysis graph on a codebase.

        Args:
            analysis_result: The parsed codebase analysis

        Returns:
            Dictionary containing insights, recommendations, and metadata
        """
        try:
            if self.debug:
                print("🚀 Starting style graph analysis...")

            # Initialize state
            initial_state = StyleGraphState(
                analysis_result=analysis_result,
                current_patterns=[],
                insights=[],
                recommendations=[],
                confidence_score=0.0,
                processing_stage="initialized",
                messages=[],
                errors=[],
                iteration_count=0,
                max_iterations=3,
            )
            # Run the graph
            thread_config = RunnableConfig(configurable={"thread_id": "style_analysis"})
            final_state = self.graph.invoke(initial_state, thread_config)

            # Extract results
            result = {
                "insights": [
                    {
                        "type": insight.insight_type,
                        "description": insight.description,
                        "confidence": insight.confidence,
                        "evidence": insight.evidence,
                        "recommendation": insight.recommendation,
                    }
                    for insight in final_state["insights"]
                ],
                "recommendations": final_state["recommendations"],
                "confidence_score": final_state["confidence_score"],
                "processing_stages": [msg.content for msg in final_state["messages"]],
                "errors": final_state["errors"],
                "metadata": {
                    "graph_version": "1.0",
                    "total_patterns": len(final_state["current_patterns"]),
                    "final_stage": final_state["processing_stage"],
                },
            }

            if self.debug:
                print(
                    f"✅ Style graph analysis complete! Generated {len(result['insights'])} insights"
                )

            return result

        except Exception as e:
            if self.debug:
                print(f"❌ Style graph analysis failed: {e}")
            return {
                "insights": [],
                "recommendations": [],
                "confidence_score": 0.0,
                "errors": [str(e)],
                "metadata": {"status": "failed"},
            }

    def export_graph_visualization(self, output_file: str) -> None:
        """Export a visualization of the graph structure."""
        try:
            # Get the graph as a PNG/mermaid
            graph_image = self.graph.get_graph().draw_mermaid_png()

            with open(output_file, "wb") as f:
                f.write(graph_image)

            if self.debug:
                print(f"📊 Graph visualization saved to: {output_file}")

        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to export graph visualization: {e}")
