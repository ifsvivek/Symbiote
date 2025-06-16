"""
LLaMA 3.3 Orchestrator Agent - Handles user input, tool calls, and workflow management.
Enhanced with modern tool calling patterns and proper Groq tool use integration.
"""

import json
import os
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr, BaseModel, Field

from ..agents.base_agent import BaseAgent, AgentMessage, AgentConfig, SassLevel
from ..tools.tool_registry import ToolRegistry
from ..tools.base_tool import ToolResult


class ToolCallRequest(BaseModel):
    """Pydantic model for tool call requests following OpenAI/Anthropic format."""
    id: str = Field(description="Unique identifier for the tool call")
    name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(description="Arguments to pass to the tool")


class LlamaAgent(BaseAgent):
    """
    LLaMA 3.3 agent for orchestration and tool management using LangChain ChatGroq.
    Enhanced with modern tool calling patterns and proper Groq tool use integration.
    """

    def __init__(self, config: AgentConfig, tool_registry: ToolRegistry):
        super().__init__(config)
        self.tool_registry = tool_registry
        self.gemini_agent = None  # Will be set by the orchestrator
        self.llm = None
        self.langchain_history: List[BaseMessage] = []

    async def initialize(self) -> bool:
        """Initialize the LLaMA agent with LangChain ChatGroq and proper tool binding."""
        try:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                print(f"[{self.name}] Warning: GROQ_API_KEY not set. Tool use will not work.")
                return False
                
            # Initialize the ChatGroq client with tool support
            self.llm = ChatGroq(
                api_key=SecretStr(self.groq_api_key),
                model="llama-3.3-70b-versatile",  # Supports tool use and parallel calls
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            # Bind tools to the LLM using LangChain's tool binding
            tools = self._get_langchain_tools()
            if tools:
                # Use bind_tools for proper function calling format
                self.llm = self.llm.bind_tools(tools, tool_choice="auto")
                print(f"[{self.name}] Bound {len(tools)} tools to LLM")
            
            print(f"[{self.name}] LLaMA agent initialized with proper tool calling support")
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to initialize: {e}")
            return False

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process user message using modern workflow with proper tool calling.
        Supports both single and multi-turn conversations with tool use.
        """
        self.add_to_history(message)
        
        try:
            if not self.llm:
                return self._create_response(
                    message,
                    "I'm not fully initialized. Please check if GROQ_API_KEY is set.",
                    {}
                )

            # Build system prompt with modern patterns
            system_prompt = self._build_system_prompt()
            user_input = message.content.get("text", "")
            
            # Build message history for multi-turn conversations
            messages = self._build_message_history(system_prompt, user_input)
            
            # Get LLM response with tool calling support
            response = await self._get_llm_response(messages)
            
            # Process any tool calls with modern patterns
            final_response, tool_results = await self._process_tool_calls(response, messages)
            
            # Update conversation history
            self.langchain_history.extend(messages[1:])  # Skip system message
            if hasattr(response, 'tool_calls') and response.tool_calls:
                self.langchain_history.append(response)
                if tool_results:
                    # Add tool results to history
                    for tool_result in tool_results:
                        tool_msg = ToolMessage(
                            content=json.dumps(tool_result),
                            tool_call_id=getattr(tool_result, 'tool_call_id', 'unknown')
                        )
                        self.langchain_history.append(tool_msg)
            
            return self._create_response(message, final_response, tool_results)
            
        except Exception as e:
            error_response = f"Oops! Something went wrong: {str(e)}"
            if self.sass_level.value >= 7:
                error_response = f"Well, that's embarrassing... I broke something: {str(e)} 😅"
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                message_type="error",
                content={"text": error_response, "error": str(e)},
                timestamp=datetime.now().isoformat(),
            )

    async def _get_llm_response(self, messages: List[BaseMessage]) -> Any:
        """Get response from LLM with proper error handling."""
        try:
            if self.llm is None:
                raise Exception("LLM not initialized")
            # Use ainvoke for async operation
            response = await self.llm.ainvoke(messages)
            return response
        except Exception as e:
            print(f"[{self.name}] Error getting LLM response: {e}")
            raise

    async def _process_tool_calls(self, response: Any, messages: List[BaseMessage]) -> tuple[str, Dict[str, Any]]:
        """
        Process tool calls with modern patterns.
        Returns (final_response_text, tool_results_dict)
        """
        tool_results = {}
        
        # Check if response contains tool calls (LangChain format)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"[{self.name}] Processing {len(response.tool_calls)} tool call(s)...")
            
            # Execute all tool calls (supports parallel execution)
            tool_messages = []
            for i, tool_call in enumerate(response.tool_calls):
                try:
                    result = await self._execute_single_tool_call(tool_call, i)
                    tool_results[f"{tool_call['name']}_{i}"] = result
                    
                    # Create tool message for conversation continuity
                    tool_msg = ToolMessage(
                        content=json.dumps(result),
                        tool_call_id=tool_call.get('id', f"call_{i}")
                    )
                    tool_messages.append(tool_msg)
                    
                except Exception as e:
                    error_result = {
                        "success": False,
                        "error": str(e),
                        "tool_call_id": tool_call.get('id', f"call_{i}")
                    }
                    tool_results[f"{tool_call['name']}_{i}"] = error_result
                    print(f"[{self.name}] ❌ Tool call failed: {e}")
            
            # If we have tool results, get final response from LLM
            if tool_messages:
                # Add assistant message and tool results to conversation
                messages.append(response)
                messages.extend(tool_messages)
                
                # Get final response from LLM
                if self.llm is not None:
                    final_response = await self.llm.ainvoke(messages)
                    return self._extract_response_text(final_response), tool_results
                else:
                    return "LLM not initialized", tool_results
        
        # No tool calls, return response text directly
        return self._extract_response_text(response), tool_results

    async def _execute_single_tool_call(self, tool_call: Dict[str, Any], call_index: int) -> Dict[str, Any]:
        """Execute a single tool call and return formatted result."""
        tool_name = tool_call['name']
        tool_args = tool_call.get('args', {})
        tool_id = tool_call.get('id', f"call_{call_index}")
        
        print(f"[{self.name}] Executing {tool_name} with args: {tool_args}")
        
        # Handle special tools
        if tool_name == "request_code_generation":
            return await self._handle_code_generation(tool_args, tool_id)
        
        # Handle registry tools
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "tool_call_id": tool_id
            }
        
        # Execute the tool
        result = await tool.execute(tool_args)
        
        # Format result with modern patterns
        formatted_result = {
            "success": result.success,
            "tool_call_id": tool_id,
            "tool_name": tool_name
        }
        
        if result.success:
            formatted_result["data"] = result.data
            print(f"[{self.name}] ✅ {tool_name} executed successfully")
        else:
            formatted_result["error"] = result.error
            print(f"[{self.name}] ❌ {tool_name} failed: {result.error}")
        
        return formatted_result

    async def _handle_code_generation(self, args: Dict[str, Any], tool_id: str) -> Dict[str, Any]:
        """Handle code generation requests to Gemini agent."""
        if not self.gemini_agent:
            return {
                "success": False,
                "error": "Gemini agent not connected",
                "tool_call_id": tool_id
            }
        
        try:
            # Create request message
            code_request = AgentMessage(
                sender=self.name,
                recipient="gemini_agent",
                message_type="code_generation_request",
                content={
                    "user_request": args.get("user_request", ""),
                    "sass_level": self.sass_level.value
                },
                timestamp=datetime.now().isoformat(),
            )
            
            # Process request
            code_response = await self.gemini_agent.process_message(code_request)
            
            return {
                "success": True,
                "data": {
                    "generated_code": code_response.content.get("generated_code", ""),
                    "explanation": code_response.content.get("explanation", "")
                },
                "tool_call_id": tool_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_call_id": tool_id
            }

    def _build_system_prompt(self) -> str:
        """Build system prompt with modern patterns."""
        available_tools_info = self._get_tools_description()
        
        return f"""You are Symbiote, an agentic coding assistant with access to various tools.
Personality: {self.get_sass_prompt()}

You have access to the following tools:
{available_tools_info}

IMPORTANT GUIDELINES:
1. When users ask about "this project", "the README", "the codebase", etc., use file_manager to read relevant files first
   - For README questions: Use file_manager with "README.md" as the file_path
   - For project questions: Use file_manager to read README.md, then other relevant files
2. For file operations, always specify the operation and file_path clearly
3. Use tools proactively to provide accurate, file-based information
4. Always provide specific, helpful information based on actual file contents
5. If you can't find a file, try common variations (README.md, readme.md, README.txt, etc.)

Tool Usage Examples:
- To read README: Use file_manager with file_path="README.md"
- To list files: Use file_manager with operation="list" and directory="."
- To read any file: Use file_manager with file_path="path/to/file"

When you need to use tools, the system will automatically handle the tool calls.
Focus on providing helpful, accurate responses based on the tool results."""

    def _build_message_history(self, system_prompt: str, user_input: str) -> List[BaseMessage]:
        """Build message history for multi-turn conversations."""
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        
        # Add relevant conversation history (last 10 messages to avoid context overflow)
        if self.langchain_history:
            messages.extend(self.langchain_history[-10:])
        
        # Add current user message
        messages.append(HumanMessage(content=user_input))
        
        return messages

    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from LLM response."""
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)

    def _get_langchain_tools(self) -> List[Tool]:
        """
        Create LangChain tools with proper schema definitions for function calling.
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        
        tools = []
        
        # File Manager Tool
        class FileManagerInput(BaseModel):
            file_path: str = Field(description="Path to the file to read or operate on")
            operation: str = Field(default="read", description="Operation to perform: read, write, create, delete, list, exists")
            content: Optional[str] = Field(default=None, description="Content to write (for write operation)")
        
        async def file_manager_func(file_path: str, operation: str = "read", content: Optional[str] = None):
            """Read, write, or manage files. Default operation is read."""
            try:
                file_tool = self.tool_registry.get_tool("file_manager")
                if not file_tool:
                    return "File manager tool not available"
                
                params = {"operation": operation, "file_path": file_path}
                if content is not None:
                    params["content"] = content
                
                result = await file_tool.execute(params)
                if result.success:
                    return json.dumps(result.data)
                else:
                    return f"Error: {result.error}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        tools.append(StructuredTool.from_function(
            func=file_manager_func,
            name="file_manager",
            description="Read, write, create, delete, or list files. Use file_path to specify the target file.",
            args_schema=FileManagerInput
        ))
        
        # Git Manager Tool
        class GitManagerInput(BaseModel):
            operation: str = Field(description="Git operation: status, add, commit, push, pull, log, diff")
            path: Optional[str] = Field(default=".", description="Path for git operations")
            message: Optional[str] = Field(default=None, description="Commit message (for commit operation)")
        
        async def git_manager_func(operation: str, path: str = ".", message: Optional[str] = None):
            """Execute git operations like status, add, commit, etc."""
            try:
                git_tool = self.tool_registry.get_tool("git_manager")
                if not git_tool:
                    return "Git manager tool not available"
                
                params = {"operation": operation, "path": path}
                if message:
                    params["message"] = message
                
                result = await git_tool.execute(params)
                if result.success:
                    return json.dumps(result.data)
                else:
                    return f"Error: {result.error}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        tools.append(StructuredTool.from_function(
            func=git_manager_func,
            name="git_manager",
            description="Execute git operations like status, add, commit, push, pull, log, diff",
            args_schema=GitManagerInput
        ))
        
        # Shell Executor Tool
        class ShellExecutorInput(BaseModel):
            command: str = Field(description="Shell command to execute")
            timeout: Optional[int] = Field(default=30, description="Timeout in seconds")
        
        async def shell_executor_func(command: str, timeout: int = 30):
            """Execute shell commands safely."""
            try:
                shell_tool = self.tool_registry.get_tool("shell_executor")
                if not shell_tool:
                    return "Shell executor tool not available"
                
                result = await shell_tool.execute({"command": command, "timeout": timeout})
                if result.success:
                    return json.dumps(result.data)
                else:
                    return f"Error: {result.error}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        tools.append(StructuredTool.from_function(
            func=shell_executor_func,
            name="shell_executor",
            description="Execute shell commands safely with timeout protection",
            args_schema=ShellExecutorInput
        ))
        
        # Code Generation Tool
        class CodeGenerationInput(BaseModel):
            user_request: str = Field(description="Description of what code to generate")
        
        async def code_generation_func(user_request: str):
            """Generate code using the Gemini agent."""
            if not self.gemini_agent:
                return json.dumps({"error": "Gemini agent not connected"})
                
            try:
                # Create request message
                code_request = AgentMessage(
                    sender=self.name,
                    recipient="gemini_agent",
                    message_type="code_generation_request",
                    content={"user_request": user_request, "sass_level": self.sass_level.value},
                    timestamp=datetime.now().isoformat(),
                )
                
                # Process request
                code_response = await self.gemini_agent.process_message(code_request)
                
                return json.dumps({
                    "generated_code": code_response.content.get("generated_code", ""),
                    "explanation": code_response.content.get("explanation", "")
                })
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        tools.append(StructuredTool.from_function(
            func=code_generation_func,
            name="request_code_generation",
            description="Request code generation from the Gemini agent. Provide a clear description of what code you need.",
            args_schema=CodeGenerationInput
        ))
        
        return tools

    def _create_response(
        self,
        original_message: AgentMessage,
        main_content: str,
        tool_results: Dict[str, Any],
    ) -> AgentMessage:
        """Create a response message."""
        # Handle both ToolResult objects and dictionaries
        processed_results = {}
        for k, v in tool_results.items():
            if hasattr(v, 'data'):
                # It's a ToolResult object
                processed_results[k] = v.data
            else:
                # It's already a dictionary
                processed_results[k] = v
        
        return AgentMessage(
            sender=self.name,
            recipient=original_message.sender,
            message_type="response",
            content={
                "text": main_content,
                "tool_results": processed_results,
                "sass_level": self.sass_level.value,
            },
            timestamp=datetime.now().isoformat(),
        )

    def set_gemini_agent(self, gemini_agent):
        """Set the Gemini agent for code generation requests."""
        self.gemini_agent = gemini_agent

    def _format_tool_output(self, tool_result: ToolResult) -> str:
        """Format tool output for display."""
        if not tool_result.success:
            return f"Error: {tool_result.error}"

        data = tool_result.data
        if not data:
            return "Success"

        # Format based on data type
        if "output" in data:
            output = data["output"]
            if len(output) > 200:
                return f"{output[:200]}..."
            return output
        elif "items" in data:
            items = data["items"]
            if isinstance(items, list):
                return f"Found {len(items)} items: {', '.join(items[:5])}{'...' if len(items) > 5 else ''}"
            return str(items)
        elif "generated_code" in data:
            code = data["generated_code"]
            if len(code) > 200:
                return f"Generated code ({len(code)} chars): {code[:200]}..."
            return f"Generated code: {code}"
        else:
            return str(data)[:200]

    def _get_tools_description(self) -> str:
        """Generate a description of available tools for the LLM."""
        tools_info = []
        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                tool_info = tool.get_info()
                tools_info.append(f"• {tool_info.name}: {tool_info.description}")
        
        return "\n".join(tools_info) if tools_info else "No tools available"
