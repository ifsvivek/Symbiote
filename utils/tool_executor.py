"""
🧬 Symbiote Autonomous Tool Executor

This module implements an intelligent tool execution system that allows LLMs to
autonomously decide and execute tools based on structured responses.

Key Features:
- Parse structured tool calls from LLM responses
- Autonomous tool execution with safety checks
- Dynamic tool discovery and registration
- Context-aware tool recommendations
- Virtual environment integration

Author: Vivek Sharma
License: MIT
"""

import json
import re
import ast
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from utils.tools import SymbioteTools


@dataclass
class ToolCall:
    """Represents a tool call request from an LLM."""

    tool_name: str
    arguments: Dict[str, Any]
    reasoning: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolExecutionResult:
    """Represents the result of a tool execution."""

    tool_call: ToolCall
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call": self.tool_call.to_dict(),
            "success": self.success,
            "result": str(self.result) if self.result is not None else None,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
        }


class AutonomousToolExecutor:
    """
    Intelligent tool executor that can parse and execute tool calls from LLM responses.
    """

    def __init__(
        self,
        tools: Optional[SymbioteTools] = None,
        debug: bool = False,
        workspace_path: Optional[Path] = None,
    ):
        self.debug = debug
        self.workspace_path = workspace_path or Path.cwd()
        self.tools = tools
        self.execution_history: List[ToolExecutionResult] = []

        # Safety settings
        self.max_executions_per_session = 10
        self.current_executions = 0

        # Tool call patterns for parsing LLM responses
        self._tool_call_patterns = [
            # JSON format: {"tool": "name", "args": {...}}
            r'```json\s*(\{[^}]*"tool"[^}]*\})\s*```',
            r'```\s*(\{[^}]*"tool"[^}]*\})\s*```',
            # Function call format: tool_name(arg1="value", arg2="value")
            r"(\w+)\s*\(\s*([^)]*)\s*\)",
            # Structured format: TOOL_CALL: name ARGS: {...}
            r"TOOL_CALL:\s*(\w+)\s+ARGS:\s*(\{[^}]*\})",
            # Natural language format: "I'll use the X tool with Y arguments"
            r"I'll use (?:the\s+)?(\w+)\s+tool\s+with\s+(.+?)(?:\.|$)",
            r"Let me (?:use\s+)?(?:the\s+)?(\w+)\s+(?:tool\s+)?(?:to\s+)?(.+?)(?:\.|$)",
        ]

        if self.debug:
            print(f"🔧 AutonomousToolExecutor initialized")
            print(f"   📁 Workspace: {self.workspace_path}")
            print(
                f"   🛠️  Tools available: {len(self.tools.tools) if self.tools else 0}"
            )

    def parse_tool_calls(self, llm_response: str) -> List[ToolCall]:
        """
        Parse tool calls from an LLM response.

        Args:
            llm_response: The raw response from the LLM

        Returns:
            List of parsed tool calls
        """
        tool_calls = []

        # Method 1: Look for explicit JSON tool calls
        tool_calls.extend(self._parse_json_tool_calls(llm_response))

        # Method 2: Look for function-style calls
        tool_calls.extend(self._parse_function_style_calls(llm_response))

        # Method 3: Look for structured format calls
        tool_calls.extend(self._parse_structured_calls(llm_response))

        # Method 4: Intelligent inference based on content
        if not tool_calls:
            tool_calls.extend(self._infer_tool_calls(llm_response))

        if self.debug and tool_calls:
            print(f"🔍 Parsed {len(tool_calls)} tool calls from LLM response")
            for call in tool_calls:
                print(f"   🛠️  {call.tool_name}: {call.arguments}")

        return tool_calls

    def _parse_json_tool_calls(self, response: str) -> List[ToolCall]:
        """Parse JSON-formatted tool calls."""
        tool_calls = []

        # Look for JSON blocks
        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r'(\{[^{}]*"tool"[^{}]*\})',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, dict) and "tool" in data:
                        tool_call = ToolCall(
                            tool_name=data["tool"],
                            arguments=data.get("args", data.get("arguments", {})),
                            reasoning=data.get("reasoning"),
                            confidence=data.get("confidence", 1.0),
                        )
                        tool_calls.append(tool_call)
                except (json.JSONDecodeError, KeyError):
                    continue

        return tool_calls

    def _parse_function_style_calls(self, response: str) -> List[ToolCall]:
        """Parse function-style tool calls like: tool_name(arg1="value")."""
        tool_calls = []

        # Pattern to match function calls
        pattern = r"(\w+)\s*\(\s*([^)]*)\s*\)"
        matches = re.findall(pattern, response)

        for tool_name, args_str in matches:
            # Skip if it looks like regular code rather than a tool call
            if tool_name in ["print", "len", "str", "int", "float", "list", "dict"]:
                continue

            # Check if this is a known tool
            if self.tools and tool_name in [t.name for t in self.tools.tools]:
                args = self._parse_function_arguments(args_str)
                tool_call = ToolCall(
                    tool_name=tool_name, arguments=args, confidence=0.8
                )
                tool_calls.append(tool_call)

        return tool_calls

    def _parse_structured_calls(self, response: str) -> List[ToolCall]:
        """Parse structured format: TOOL_CALL: name ARGS: {...}."""
        tool_calls = []

        pattern = r"TOOL_CALL:\s*(\w+)\s+ARGS:\s*(\{[^}]*\})"
        matches = re.findall(pattern, response, re.IGNORECASE)

        for tool_name, args_str in matches:
            try:
                args = json.loads(args_str)
                tool_call = ToolCall(
                    tool_name=tool_name, arguments=args, confidence=0.9
                )
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _infer_tool_calls(self, response: str) -> List[ToolCall]:
        """Intelligently infer tool calls based on response content."""
        tool_calls = []

        if not self.tools:
            return tool_calls

        response_lower = response.lower()

        # Map common phrases to tool calls
        inference_map = {
            # File operations
            "read file": ("read_file_content", {}),
            "read the file": ("read_file_content", {}),
            "show file": ("read_file_content", {}),
            "examine file": ("read_file_content", {}),
            "list files": ("list_directory_files", {}),
            "show directory": ("list_directory_files", {}),
            # Code analysis
            "analyze code": ("generate_ai_insights", {}),
            "check code quality": ("identify_code_smells", {}),
            "assess complexity": ("assess_code_complexity", {}),
            "analyze naming": ("analyze_naming_conventions", {}),
            "check structure": ("evaluate_code_structure", {}),
            "detect patterns": ("detect_design_patterns", {}),
            # Git operations
            "check git": ("get_git_info", {}),
            "git history": ("analyze_git_history", {}),
            "git status": ("get_git_info", {}),
        }

        for phrase, (tool_name, base_args) in inference_map.items():
            if phrase in response_lower:
                # Check if this tool exists
                if tool_name in [t.name for t in self.tools.tools]:
                    # Try to extract file/directory arguments from context
                    args = self._extract_context_arguments(response, base_args)

                    tool_call = ToolCall(
                        tool_name=tool_name,
                        arguments=args,
                        reasoning=f"Inferred from phrase: '{phrase}'",
                        confidence=0.7,
                    )
                    tool_calls.append(tool_call)

        return tool_calls

    def _parse_function_arguments(self, args_str: str) -> Dict[str, Any]:
        """Parse function-style arguments string into a dictionary."""
        args = {}

        if not args_str.strip():
            return args

        # Simple parsing for key=value pairs
        pairs = re.findall(r"(\w+)\s*=\s*([^,]+)", args_str)
        for key, value in pairs:
            # Clean up the value
            value = value.strip().strip("\"'")
            args[key] = value

        # If no key=value pairs found, treat as positional arguments
        if not args and args_str.strip():
            # For single argument, use a common parameter name
            args["input"] = args_str.strip().strip("\"'")

        return args

    def _extract_context_arguments(
        self, response: str, base_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract contextual arguments like file paths from the response."""
        args = base_args.copy()

        # Look for file references
        file_patterns = [
            r"(\w+\.py)",
            r"(\w+\.js)",
            r"(\w+\.ts)",
            r"(\w+\.java)",
            r"(main\.py)",
            r"(utils/\w+\.py)",
            r"(src/\w+\.py)",
        ]

        for pattern in file_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                args["file_path"] = matches[0]
                break

        # Look for directory references
        if "directory" in response.lower() or "folder" in response.lower():
            dir_matches = re.findall(r"([./\w]+)", response)
            for match in dir_matches:
                if Path(match).exists() or match in [".", "..", "./"]:
                    args["directory_path"] = match
                    break

        return args

    def execute_tool_calls(
        self, tool_calls: List[ToolCall]
    ) -> List[ToolExecutionResult]:
        """
        Execute a list of tool calls safely.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of execution results
        """
        results = []

        if not self.tools:
            for call in tool_calls:
                result = ToolExecutionResult(
                    tool_call=call,
                    success=False,
                    result=None,
                    error_message="No tools available for execution",
                )
                results.append(result)
            return results

        for tool_call in tool_calls:
            # Safety check
            if self.current_executions >= self.max_executions_per_session:
                result = ToolExecutionResult(
                    tool_call=tool_call,
                    success=False,
                    result=None,
                    error_message="Maximum executions per session reached",
                )
                results.append(result)
                continue

            result = self._execute_single_tool_call(tool_call)
            results.append(result)
            self.execution_history.append(result)
            self.current_executions += 1

        return results

    def _execute_single_tool_call(self, tool_call: ToolCall) -> ToolExecutionResult:
        """Execute a single tool call."""
        import time

        start_time = time.time()

        try:
            if self.debug:
                print(f"🔧 Executing tool: {tool_call.tool_name}")
                print(f"   📝 Arguments: {tool_call.arguments}")

            # Check if tools are available
            if not self.tools:
                raise RuntimeError("No tools available for execution")

            # Execute the tool
            result = self.tools.execute_dynamic_tool(
                tool_call.tool_name, tool_call.arguments
            )

            execution_time = time.time() - start_time

            return ToolExecutionResult(
                tool_call=tool_call,
                success=True,
                result=result,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return ToolExecutionResult(
                tool_call=tool_call,
                success=False,
                result=None,
                error_message=str(e),
                execution_time=execution_time,
            )

    def process_llm_response(
        self, llm_response: str
    ) -> Tuple[str, List[ToolExecutionResult]]:
        """
        Process an LLM response, execute any tool calls, and return the enhanced response.

        Args:
            llm_response: The raw LLM response

        Returns:
            Tuple of (enhanced_response, execution_results)
        """
        # Parse tool calls from the response
        tool_calls = self.parse_tool_calls(llm_response)

        if not tool_calls:
            return llm_response, []

        # Execute the tool calls
        results = self.execute_tool_calls(tool_calls)

        # Create enhanced response with tool results
        enhanced_response = llm_response + "\n\n🔧 **Tool Execution Results:**\n"

        for result in results:
            if result.success:
                enhanced_response += f"\n✅ **{result.tool_call.tool_name}**:\n"
                enhanced_response += f"```\n{result.result}\n```\n"
            else:
                enhanced_response += f"\n❌ **{result.tool_call.tool_name}** failed:\n"
                enhanced_response += f"Error: {result.error_message}\n"

        return enhanced_response, results

    def get_tool_call_suggestions(
        self, user_query: str, context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Get tool call suggestions based on user query and context.

        Args:
            user_query: The user's query
            context: Optional context information

        Returns:
            List of suggested tool calls
        """
        suggestions = []

        if not self.tools:
            return suggestions

        query_lower = user_query.lower()

        # Suggest tools based on query content
        if any(
            word in query_lower for word in ["analyze", "check", "examine", "review"]
        ):
            if "file" in query_lower or "code" in query_lower:
                suggestions.append("generate_ai_insights")
                suggestions.append("identify_code_smells")
                suggestions.append("assess_code_complexity")

        if any(word in query_lower for word in ["read", "show", "display", "see"]):
            if "file" in query_lower:
                suggestions.append("read_file_content")
            elif "directory" in query_lower or "folder" in query_lower:
                suggestions.append("list_directory_files")

        if any(
            word in query_lower for word in ["git", "commit", "history", "repository"]
        ):
            suggestions.append("get_git_info")
            suggestions.append("analyze_git_history")

        if any(word in query_lower for word in ["naming", "convention", "style"]):
            suggestions.append("analyze_naming_conventions")
            suggestions.append("extract_style_preferences")

        return suggestions

    def create_enhanced_system_prompt(self) -> str:
        """
        Create an enhanced system prompt that guides the LLM on tool usage.

        Returns:
            Enhanced system prompt string
        """
        if not self.tools:
            return "No tools available."

        tool_descriptions = []
        for tool in self.tools.tools:
            tool_descriptions.append(f"- **{tool.name}**: {tool.description}")

        tools_list = "\n".join(tool_descriptions)

        return f"""You are an intelligent coding assistant with access to powerful analysis tools. You can autonomously decide when and how to use these tools to provide better assistance.

## Available Tools:
{tools_list}

## Tool Usage Guidelines:

### When to Use Tools:
- **File Analysis**: When user asks about code quality, structure, or patterns
- **Code Review**: When user wants to understand or improve code
- **File Operations**: When user wants to read, examine, or list files
- **Git Operations**: When user asks about repository status or history

### How to Request Tool Execution:
You can request tool execution in several ways:

1. **JSON Format** (Recommended):
```json
{{
    "tool": "tool_name",
    "args": {{"argument": "value"}},
    "reasoning": "Why this tool is needed"
}}
```

2. **Function Style**:
```
tool_name(argument="value")
```

3. **Structured Format**:
```
TOOL_CALL: tool_name ARGS: {{"argument": "value"}}
```

4. **Natural Language**:
"I'll use the analyze_naming_conventions tool to check the code style patterns."

### Best Practices:
- Use tools proactively when they would provide valuable insights
- Combine multiple tools for comprehensive analysis
- Always explain why you're using a specific tool
- Use the most specific tool for the task at hand

### Examples:
- User asks "How's the code quality?" → Use `identify_code_smells` and `assess_code_complexity`
- User asks "Show me main.py" → Use `read_file_content` with file_path="main.py"
- User asks "What files are here?" → Use `list_directory_files`
- User asks "Check git status" → Use `get_git_info`

Remember: You have the autonomy to decide when tools would be helpful. Use them proactively to provide comprehensive, insightful responses."""

    def reset_session(self):
        """Reset the execution session."""
        self.current_executions = 0
        self.execution_history.clear()

        if self.debug:
            print("🔄 Tool execution session reset")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about tool execution in this session."""
        successful_executions = sum(1 for r in self.execution_history if r.success)
        failed_executions = len(self.execution_history) - successful_executions

        tool_usage = {}
        for result in self.execution_history:
            tool_name = result.tool_call.tool_name
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        return {
            "total_executions": len(self.execution_history),
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "current_session_executions": self.current_executions,
            "tool_usage": tool_usage,
            "average_execution_time": sum(
                r.execution_time for r in self.execution_history
            )
            / max(len(self.execution_history), 1),
        }


def create_autonomous_tool_executor(
    tools: Optional[SymbioteTools] = None,
    debug: bool = False,
    workspace_path: Optional[Path] = None,
) -> AutonomousToolExecutor:
    """Factory function to create AutonomousToolExecutor instance."""
    return AutonomousToolExecutor(
        tools=tools, debug=debug, workspace_path=workspace_path
    )
