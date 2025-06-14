"""
LLaMA 3.3 Orchestrator Agent - Handles user input, tool calls, and workflow management.
"""

import json
import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

from ..agents.base_agent import BaseAgent, AgentMessage, AgentConfig, SassLevel
from ..tools.tool_registry import ToolRegistry
from ..tools.base_tool import ToolResult


class LlamaAgent(BaseAgent):
    """LLaMA 3.3 agent for orchestration and tool management using LangChain ChatGroq."""

    def __init__(self, config: AgentConfig, tool_registry: ToolRegistry):
        super().__init__(config)
        self.tool_registry = tool_registry
        self.gemini_agent = None  # Will be set by the orchestrator
        self.llm = None

    async def initialize(self) -> bool:
        """Initialize the LLaMA agent with LangChain ChatGroq."""
        try:
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            if not self.groq_api_key:
                print(f"[{self.name}] Warning: GROQ_API_KEY not set. Tool use will not work.")
            self.llm = ChatGroq(
                api_key=SecretStr(self.groq_api_key) if self.groq_api_key else None,
                model="llama-3.3-70b-versatile",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            print(f"[{self.name}] LLaMA agent initialized (LangChain ChatGroq mode)")
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to initialize: {e}")
            return False

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process user message and orchestrate the workflow using LangChain ChatGroq tool calling."""
        self.add_to_history(message)
        try:
            if not self.llm:
                return self._create_response(
                    message,
                    "I'm not fully initialized. Please check if GROQ_API_KEY is set.",
                    {}
                )
            system_prompt = (
                "You are Symbiote, an agentic coding assistant. "
                f"Personality: {self.get_sass_prompt()} "
                "You can call tools to interact with files, git, shell, and more. "
                "If the user requests code generation, call the 'request_code_generation' tool."
            )
            lc_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message.content.get("text", "")),
            ]
            user_input = message.content.get("text", "")
            # Detect if code generation is requested
            if any(word in user_input.lower() for word in ["generate", "create", "code", "calculator", "script"]):
                # Actually call the code generation tool
                code_tool = None
                for tool in self._get_langchain_tools():
                    if tool.name == "request_code_generation":
                        code_tool = tool
                        break
                if code_tool:
                    code_result = code_tool.func(user_request=user_input)
                    # If async, run it in event loop
                    if asyncio.iscoroutine(code_result):
                        code_result = await code_result
                    response_text = f"[TOOL CALLED: request_code_generation]\n{code_result}"
                    return self._create_response(message, response_text, {"request_code_generation": code_result})
            # Use the direct tool execution approach based on keywords
            tool_results = await self._execute_tools_for_request(user_input)
            try:
                # We'll use the raw LLM without tool calling for now
                response = self.llm.invoke(
                    f"System: {system_prompt}\nUser: {user_input}"
                )
                # Extract string content from various possible response types
                if hasattr(response, 'content'):
                    response_text = str(response.content)
                else:
                    response_text = str(response)
            except Exception:
                # Fallback response
                response_text = "I understand your request. Let me help with that."
            # Format response with tool results if any tools were executed
            if tool_results:
                formatted_response = self._format_response_with_tools(response_text, tool_results)
                return self._create_response(message, formatted_response, tool_results)
            else:
                # Return the original LLM response if no tools were executed
                return self._create_response(message, response_text, {})
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
        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            message_type="error",
            content={"text": "No response generated.", "error": "No response generated."},
            timestamp=datetime.now().isoformat(),
        )

    def _get_langchain_tools(self):
        """Wrap registered tools as LangChain tools."""
        from langchain_core.tools import Tool
        
        lc_tools = []
        
        # Create a wrapper for all registered tools
        for tool_name in self.tool_registry.list_tools():
            symbiote_tool = self.tool_registry.get_tool(tool_name)
            if not symbiote_tool:
                continue
            
            # Create a wrapper function for this tool
            def create_tool_wrapper(tool):
                def wrapper(**kwargs):
                    # We need to run the async tool in a synchronous context
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(tool.execute(kwargs))
                    if result.success:
                        return json.dumps(result.data)
                    else:
                        return f"Error: {result.error}"
                return wrapper
            
            # Create and add this tool
            lc_tool = Tool(
                name=symbiote_tool.name,
                description=symbiote_tool.description,
                func=create_tool_wrapper(symbiote_tool)
            )
            lc_tools.append(lc_tool)
        
        # Add Gemini code generation as a tool
        def code_gen_wrapper(user_request: str):
            if not self.gemini_agent:
                return "Gemini agent not connected."
                
            # Create request message
            code_request = AgentMessage(
                sender=self.name,
                recipient="gemini_agent",
                message_type="code_generation_request",
                content={"user_request": user_request, "sass_level": self.sass_level.value},
                timestamp=datetime.now().isoformat(),
            )
            
            # Process request synchronously through asyncio
            loop = asyncio.get_event_loop()
            code_response = loop.run_until_complete(
                self.gemini_agent.process_message(code_request)
            )
            
            return code_response.content.get("generated_code", "")
        
        # Add the code generation tool
        lc_tools.append(
            Tool(
                name="request_code_generation",
                description="Request code generation from the Gemini agent.",
                func=code_gen_wrapper
            )
        )
        
        return lc_tools

    def _create_response(
        self,
        original_message: AgentMessage,
        main_content: str,
        tool_results: Dict[str, ToolResult],
    ) -> AgentMessage:
        """Create a response message."""
        return AgentMessage(
            sender=self.name,
            recipient=original_message.sender,
            message_type="response",
            content={
                "text": main_content,
                "tool_results": {k: v.data for k, v in tool_results.items()},
                "sass_level": self.sass_level.value,
            },
            timestamp=datetime.now().isoformat(),
        )

    def set_gemini_agent(self, gemini_agent):
        """Set the Gemini agent for code generation requests."""
        self.gemini_agent = gemini_agent

    async def _execute_tools_for_request(self, user_input: str) -> Dict[str, Any]:
        """Execute tools based on user request analysis."""
        tool_results = {}
        user_lower = user_input.lower()

        try:
            # Determine which tools to execute based on keywords
            if any(
                keyword in user_lower
                for keyword in ["list", "files", "directory", "ls", "dir"]
            ):
                file_tool = self.tool_registry.get_tool("file_manager")
                if file_tool:
                    result = await file_tool.execute(
                        {"operation": "list", "directory": "."}
                    )
                    tool_results["file_list"] = result

            if any(
                keyword in user_lower
                for keyword in ["git", "status", "commit", "push", "pull"]
            ):
                git_tool = self.tool_registry.get_tool("git_manager")
                if git_tool:
                    # Determine git operation
                    if "status" in user_lower or "git" in user_lower:
                        result = await git_tool.execute({"operation": "status"})
                        tool_results["git_status"] = result
                    if "commit" in user_lower:
                        result = await git_tool.execute(
                            {"operation": "add", "files": "."}
                        )
                        tool_results["git_add"] = result
                        if "message" in user_lower or "commit" in user_lower:
                            commit_msg = "Auto commit via Symbiote"
                            result = await git_tool.execute(
                                {"operation": "commit", "message": commit_msg}
                            )
                            tool_results["git_commit"] = result
                    if "push" in user_lower:
                        result = await git_tool.execute({"operation": "push"})
                        tool_results["git_push"] = result

            if any(
                keyword in user_lower
                for keyword in ["run", "execute", "command", "shell"]
            ):
                shell_tool = self.tool_registry.get_tool("shell_executor")
                if shell_tool:
                    # Extract command or use default
                    if "ls" in user_lower:
                        result = await shell_tool.execute({"command": "ls -la"})
                        tool_results["shell_ls"] = result
                    elif "pwd" in user_lower:
                        result = await shell_tool.execute({"command": "pwd"})
                        tool_results["shell_pwd"] = result

        except Exception as e:
            tool_results["error"] = {"error": str(e)}

        return tool_results

    def _format_response_with_tools(
        self, base_response: str, tool_results: Dict[str, Any]
    ) -> str:
        """Format response with tool execution results."""
        if not tool_results:
            return base_response

        response_parts = [base_response]

        for tool_name, result in tool_results.items():
            # Handle various result formats
            if isinstance(result, dict) and "error" in result:
                # Handle error dict
                response_parts.append(f"\n❌ {tool_name}: {result.get('error', 'Unknown error')}")
            elif isinstance(result, ToolResult):
                # Handle ToolResult object
                if result.success:
                    response_parts.append(f"\n✅ {tool_name}: {self._summarize_tool_result(result)}")
                else:
                    response_parts.append(f"\n❌ {tool_name}: {result.error}")
            else:
                # Generic handling for any other type
                response_parts.append(f"\n✅ {tool_name}: {str(result)[:200]}")

        return "\n".join(response_parts)

    def _summarize_tool_result(self, result) -> str:
        """Create a summary of tool execution result."""
        if hasattr(result, "data"):
            data = result.data
            if "items" in data:  # File listing
                return f"Found {len(data['items'])} items"
            elif "output" in data:  # Git/shell output
                output = data["output"]
                return output[:100] + "..." if len(output) > 100 else output
            else:
                return str(data)[:100]
        return str(result)[:100]

    async def _execute_tool_call(
        self, function_name: str, function_args: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool call and return the result."""
        try:
            if function_name == "request_code_generation":
                # Handle code generation request
                if self.gemini_agent:
                    user_request = function_args.get("user_request", "")
                    # Create a message for Gemini
                    gemini_message = AgentMessage(
                        sender=self.name,
                        recipient="gemini_coder",
                        message_type="code_request",
                        content={"text": user_request},
                        timestamp=datetime.now().isoformat(),
                    )
                    response = await self.gemini_agent.process_message(gemini_message)
                    return ToolResult(
                        success=True,
                        data={"generated_code": response.content.get("text", "")},
                        error="",
                    )
                else:
                    return ToolResult(
                        success=False,
                        data={},
                        error="Gemini agent not available for code generation",
                    )
            else:
                # Execute regular tool
                tool = self.tool_registry.get_tool(function_name)
                if tool:
                    return await tool.execute(function_args)
                else:
                    return ToolResult(
                        success=False,
                        data={},
                        error=f"Tool '{function_name}' not found",
                    )
        except Exception as e:
            return ToolResult(
                success=False, data={}, error=f"Tool execution failed: {str(e)}"
            )

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
