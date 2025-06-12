"""
🧬 Symbiote Smart Context Manager

This module provides intelligent context management for better AI decision making.
It tracks user patterns, workspace state, and provides dynamic context updates.

Author: Vivek Sharma
License: MIT
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque


@dataclass
class UserInteraction:
    """Represents a user interaction with the system."""

    timestamp: datetime
    query: str
    intent: str  # 'analysis', 'modification', 'exploration', 'debugging'
    files_accessed: List[str]
    tools_used: List[str]
    success: bool
    session_id: str


@dataclass
class WorkspaceState:
    """Current state of the workspace."""

    active_files: Set[str]
    recent_modifications: List[str]
    git_status: Dict[str, Any]
    last_analysis: Optional[datetime]
    complexity_score: float
    primary_language: str
    estimated_session_focus: str


class SmartContextManager:
    """
    Intelligent context manager that learns user patterns and optimizes AI responses.

    Features:
    - Tracks user interaction patterns
    - Manages workspace state
    - Provides intelligent tool suggestions
    - Learns from successful interactions
    - Adapts to user preferences
    """

    def __init__(self, workspace_path: Path, debug: bool = False):
        self.workspace_path = workspace_path
        self.debug = debug

        # User interaction tracking
        self.interaction_history: deque = deque(maxlen=100)
        self.user_patterns: Dict[str, Any] = defaultdict(int)
        self.successful_tools: Dict[str, int] = defaultdict(int)

        # Workspace state
        self.workspace_state = WorkspaceState(
            active_files=set(),
            recent_modifications=[],
            git_status={},
            last_analysis=None,
            complexity_score=0.0,
            primary_language="unknown",
            estimated_session_focus="exploration",
        )

        # Context cache
        self.context_cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}

        # Session tracking
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()

        if self.debug:
            print(f"🧠 SmartContextManager initialized for {workspace_path}")

    def track_interaction(
        self,
        query: str,
        intent: str,
        files_accessed: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        success: bool = True,
    ):
        """Track a user interaction for learning."""
        interaction = UserInteraction(
            timestamp=datetime.now(),
            query=query,
            intent=intent,
            files_accessed=files_accessed or [],
            tools_used=tools_used or [],
            success=success,
            session_id=self.current_session_id,
        )

        self.interaction_history.append(interaction)

        # Update patterns
        self.user_patterns[intent] += 1
        for tool in tools_used or []:
            if success:
                self.successful_tools[tool] += 1

        # Update workspace state
        if files_accessed:
            self.workspace_state.active_files.update(files_accessed)

        if self.debug:
            print(f"📊 Tracked interaction: {intent} -> {success}")

    def infer_user_intent(self, query: str) -> str:
        """Infer user intent from query using patterns."""
        query_lower = query.lower()

        # Intent keywords mapping
        intent_patterns = {
            "analysis": [
                "analyze",
                "check",
                "examine",
                "review",
                "assess",
                "evaluate",
                "what does",
                "how does",
                "explain",
                "understand",
                "complexity",
            ],
            "modification": [
                "fix",
                "change",
                "update",
                "modify",
                "add",
                "remove",
                "refactor",
                "implement",
                "create",
                "delete",
                "edit",
            ],
            "exploration": [
                "show",
                "list",
                "find",
                "search",
                "browse",
                "navigate",
                "where is",
                "what files",
                "structure",
            ],
            "debugging": [
                "debug",
                "error",
                "bug",
                "issue",
                "problem",
                "why",
                "not working",
                "failed",
                "broken",
            ],
        }

        # Score each intent
        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] = score

        # Return highest scoring intent or default
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=lambda x: intent_scores[x])
        return "exploration"

    def get_context_suggestions(self, query: str, intent: str) -> Dict[str, Any]:
        """Get intelligent context suggestions for the query."""
        suggestions = {
            "recommended_tools": [],
            "relevant_files": [],
            "context_hints": [],
            "confidence": 0.0,
        }

        # Tool suggestions based on intent and history
        tool_recommendations = {
            "analysis": [
                "analyze_naming_conventions",
                "assess_code_complexity",
                "evaluate_code_structure",
                "generate_ai_insights",
            ],
            "exploration": [
                "list_directory_files",
                "read_file_content",
                "get_git_info",
            ],
            "debugging": [
                "identify_code_smells",
                "assess_code_complexity",
                "read_file_content",
            ],
            "modification": ["read_file_content", "create_file", "modify_file"],
        }

        base_tools = tool_recommendations.get(intent, [])

        # Filter based on successful history
        if self.successful_tools:
            # Sort by success rate
            sorted_tools = sorted(
                self.successful_tools.items(), key=lambda x: x[1], reverse=True
            )
            successful_tool_names = [tool for tool, _ in sorted_tools[:5]]

            # Prefer historically successful tools
            suggestions["recommended_tools"] = [
                tool for tool in base_tools if tool in successful_tool_names
            ] + [tool for tool in base_tools if tool not in successful_tool_names]
        else:
            suggestions["recommended_tools"] = base_tools

        # File suggestions based on query content and recent activity
        query_lower = query.lower()
        potential_files = []

        # Check for explicit file mentions
        for file_ext in [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"]:
            if file_ext in query_lower:
                # Look for file patterns in the query
                words = query_lower.split()
                for word in words:
                    if file_ext in word:
                        clean_file = word.strip('.,!?()[]{}":;')
                        if self._file_exists(clean_file):
                            potential_files.append(clean_file)

        # Add recently active files
        potential_files.extend(list(self.workspace_state.active_files)[:3])

        suggestions["relevant_files"] = list(set(potential_files))

        # Context hints based on patterns
        if intent == "analysis" and self.workspace_state.last_analysis:
            time_since = datetime.now() - self.workspace_state.last_analysis
            if time_since < timedelta(minutes=30):
                suggestions["context_hints"].append("Recent analysis available")

        if self.workspace_state.recent_modifications:
            suggestions["context_hints"].append(
                f"Recent changes in: {', '.join(self.workspace_state.recent_modifications[:3])}"
            )

        # Calculate confidence based on available context
        confidence_factors = [
            len(suggestions["recommended_tools"]) > 0,
            len(suggestions["relevant_files"]) > 0,
            len(self.interaction_history) > 5,  # Have some history
            intent in self.user_patterns and self.user_patterns[intent] > 1,
        ]

        suggestions["confidence"] = sum(confidence_factors) / len(confidence_factors)

        return suggestions

    def update_workspace_state(self, **kwargs):
        """Update workspace state with new information."""
        for key, value in kwargs.items():
            if hasattr(self.workspace_state, key):
                setattr(self.workspace_state, key, value)

        if self.debug:
            print(f"🔄 Updated workspace state: {list(kwargs.keys())}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        session_duration = datetime.now() - self.session_start
        recent_interactions = [
            i
            for i in self.interaction_history
            if i.session_id == self.current_session_id
        ]

        return {
            "session_id": self.current_session_id,
            "duration_minutes": session_duration.total_seconds() / 60,
            "total_interactions": len(recent_interactions),
            "primary_intent": (
                max(self.user_patterns, key=lambda x: self.user_patterns[x])
                if self.user_patterns
                else "unknown"
            ),
            "success_rate": sum(1 for i in recent_interactions if i.success)
            / max(len(recent_interactions), 1),
            "tools_used": list(
                set(tool for i in recent_interactions for tool in i.tools_used)
            ),
            "files_accessed": list(self.workspace_state.active_files),
            "estimated_focus": self.workspace_state.estimated_session_focus,
        }

    def _file_exists(self, file_path: str) -> bool:
        """Check if a file exists in the workspace."""
        try:
            full_path = self.workspace_path / file_path
            return full_path.exists()
        except Exception:
            return False

    def export_learning_data(self, output_file: str):
        """Export learning data for analysis."""
        data = {
            "user_patterns": dict(self.user_patterns),
            "successful_tools": dict(self.successful_tools),
            "workspace_state": asdict(self.workspace_state),
            "interaction_count": len(self.interaction_history),
            "session_summary": self.get_session_summary(),
            "export_timestamp": datetime.now().isoformat(),
        }

        # Convert sets to lists for JSON serialization
        if "active_files" in data["workspace_state"]:
            data["workspace_state"]["active_files"] = list(
                data["workspace_state"]["active_files"]
            )

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        if self.debug:
            print(f"📊 Learning data exported to {output_file}")


def create_smart_context_manager(
    workspace_path: Path, debug: bool = False
) -> SmartContextManager:
    """Factory function to create SmartContextManager."""
    return SmartContextManager(workspace_path, debug)
