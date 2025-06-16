"""
Symbiote orchestrator with full tech stack integration.
Implements advanced workflows with LangGraph, FAISS memory, and AST analysis.
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .agents.base_agent import AgentMessage, AgentConfig, SassLevel
from .agents.llama_agent import LlamaAgent
from .agents.gemini_agent import GeminiAgent
from .tools.tool_registry import ToolRegistry
from .tools.file_manager import FileManagerTool
from .tools.code_analysis import CodeAnalysisTool
from .memory.vector_store import VectorMemoryStore
from .workflows.langgraph_orchestrator import SymbioteWorkflow


class SymbioteOrchestrator:
    """
    orchestrator with full tech stack integration.
    Manages multi-agent workflows, memory, and intelligent interactions.
    """

    def __init__(self, use_langgraph: bool = True, verbose: bool = True):
        # Core components
        self.tool_registry = ToolRegistry()
        self.memory_store: Optional[VectorMemoryStore] = None

        # Agents
        self.llama_agent: Optional[LlamaAgent] = None
        self.gemini_agent: Optional[GeminiAgent] = None

        # Workflow management
        self.use_langgraph = use_langgraph
        self.workflow: Optional[SymbioteWorkflow] = None
        self.verbose = verbose

        # State
        self.current_sass_level = SassLevel.FRIENDLY
        self.conversation_id = 0
        self.is_initialized = False

    async def initialize(
        self,
        llama_config: Optional[AgentConfig] = None,
        gemini_config: Optional[AgentConfig] = None,
        memory_persist_dir: str = "./memory_store",
    ) -> bool:
        """Initialize the orchestrator with full tech stack."""
        print("🧬 Initializing Symbiote with full tech stack...")

        try:
            # Initialize vector memory store
            print("🧠 Initializing FAISS vector memory...")
            try:
                self.memory_store = VectorMemoryStore(memory_persist_dir)
            except ValueError as e:
                if "GOOGLE_API_KEY" in str(e):
                    print("⚠️ GOOGLE_API_KEY not set - memory features will be limited")
                    self.memory_store = None
                else:
                    raise e

            # Set up default configs if not provided
            if not llama_config:
                llama_config = AgentConfig(
                    name="llama_orchestrator",
                    model_name="llama-3.3-70b-versatile",
                    sass_level=self.current_sass_level,
                )

            if not gemini_config:
                gemini_config = AgentConfig(
                    name="gemini_coder",
                    model_name="gemini-2.0-flash-exp",
                    sass_level=self.current_sass_level,
                )

            # Register tools including AST analysis
            await self._register_enhanced_tools()

            # Initialize agents
            self.llama_agent = LlamaAgent(llama_config, self.tool_registry, verbose=self.verbose)
            self.gemini_agent = GeminiAgent(gemini_config)

            # Connect agents
            self.llama_agent.set_gemini_agent(self.gemini_agent)

            # Initialize both agents
            llama_success = await self.llama_agent.initialize()
            gemini_success = await self.gemini_agent.initialize()

            if not (llama_success and gemini_success):
                print("❌ Failed to initialize agents")
                return False

            # Initialize LangGraph workflow if enabled
            if self.use_langgraph and self.memory_store:
                print("🔄 Initializing LangGraph workflow...")
                self.workflow = SymbioteWorkflow(self.tool_registry, self.memory_store, verbose=self.verbose)
                self.workflow.set_agents(self.llama_agent, self.gemini_agent)
                print("✅ LangGraph workflow initialized")

            self.is_initialized = True
            print("✅ Symbiote initialized successfully!")
            print(
                f"🧠 Memory store: {len(self.memory_store.patterns) if self.memory_store else 0} patterns loaded"
            )
            print(f"🔧 Tools registered: {len(self.tool_registry.list_tools())}")
            print(
                f"🎭 Current sass level: {self.current_sass_level.value} ({self.current_sass_level.name})"
            )
            print(
                f"🔄 LangGraph workflow: {'Enabled' if self.use_langgraph else 'Disabled'}"
            )
            return True

        except Exception as e:
            print(f"❌ Failed to initialize Symbiote: {e}")
            return False

    async def _register_enhanced_tools(self):
        """Register tools including AST analysis."""
        print("🔧 Registering tools with full tech stack...")

        # Core tools
        file_tool = FileManagerTool()
        self.tool_registry.register_tool(file_tool)

        from .tools.git_manager import GitManagerTool

        git_tool = GitManagerTool()
        self.tool_registry.register_tool(git_tool)

        from .tools.shell_executor import ShellExecutorTool

        shell_tool = ShellExecutorTool()
        self.tool_registry.register_tool(shell_tool)

        from .tools.code_search import CodeSearchTool

        search_tool = CodeSearchTool()
        self.tool_registry.register_tool(search_tool)

        # AST-based code analysis
        code_analysis_tool = CodeAnalysisTool()
        self.tool_registry.register_tool(code_analysis_tool)

        print(f"✅ Registered {len(self.tool_registry.list_tools())} tools")

    async def _register_core_tools(self):
        """Register core tools with the tool registry (legacy method)."""
        await self._register_enhanced_tools()

    def set_verbose(self, verbose: bool):
        """Set verbose mode for debugging output."""
        self.verbose = verbose
        # Pass verbose setting to workflow if available
        if self.workflow:
            self.workflow.set_verbose(verbose)

    async def process_user_input(self, user_input: str) -> str:
        """
        Process user input with workflow capabilities.
        Uses LangGraph workflow for advanced interactions when available.
        """
        if not self.is_initialized:
            return "❌ Symbiote not initialized. Please run initialize() first."

        self.conversation_id += 1

        # Check for special commands
        if user_input.startswith("/"):
            return await self._handle_special_command(user_input)

        # Use LangGraph workflow if available for interactions
        if self.use_langgraph and self.workflow:
            if self.verbose:
                print("🔄 Using LangGraph workflow for processing...")
            try:
                response = await self.workflow.process_query(
                    user_query=user_input, thread_id=f"conv_{self.conversation_id}"
                )

                # Learn from this interaction
                if self.memory_store and "generated code" in response.lower():
                    await self._learn_from_interaction(user_input, response)

                return response
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ LangGraph workflow failed, falling back to direct agent: {e}")

        # Fallback to direct agent processing
        if not self.llama_agent:
            return "❌ Symbiote not initialized properly."

        # Create message for LLaMA agent
        message = AgentMessage(
            sender="user",
            recipient="llama_orchestrator",
            message_type="user_input",
            content={"text": user_input},
            timestamp=datetime.now().isoformat(),
            context={"conversation_id": self.conversation_id},
        )

        # Process with LLaMA agent
        response = await self.llama_agent.process_message(message)

        # Extract response text
        if response.message_type == "error":
            return f"❌ {response.content.get('text', 'An error occurred')}"
        else:
            response_text = response.content.get("text", "No response generated")

            # Learn from interaction if memory is available
            if self.memory_store and "code" in response_text.lower():
                await self._learn_from_interaction(user_input, response_text)

            return response_text

    async def _handle_special_command(self, command: str) -> str:
        """Handle special slash commands."""
        parts = command.strip().split()
        cmd = parts[0].lower()

        if cmd == "/sass":
            if len(parts) != 2:
                return "Usage: /sass <0-10>"

            try:
                level = int(parts[1])
                if 0 <= level <= 10:
                    self.current_sass_level = SassLevel(level)
                    await self._update_agent_sass_levels(level)
                    return f"🎭 Sass level updated to {level} ({self.current_sass_level.name})"
                else:
                    return "❌ Sass level must be between 0 and 10"
            except ValueError:
                return "❌ Invalid sass level. Must be a number between 0 and 10"

        elif cmd == "/tools":
            tools = self.tool_registry.list_tools()
            if not tools:
                return "No tools registered"

            tool_list = "\n".join([f"  • {tool}" for tool in tools])
            return f"🔧 Available tools:\n{tool_list}"

        elif cmd == "/status":
            stats = self.tool_registry.get_registry_stats()
            llama_status = "✅" if self.llama_agent else "❌"
            gemini_status = "✅" if self.gemini_agent else "❌"
            memory_status = "✅" if self.memory_store else "❌"
            workflow_status = "✅" if self.workflow else "❌"

            memory_info = ""
            if self.memory_store:
                pattern_stats = self.memory_store.get_pattern_stats()
                memory_info = (
                    f"\n  • Memory Patterns: {pattern_stats.get('total_patterns', 0)}"
                )

            return f"""🧬 Symbiote Status:
  • LLaMA Agent: {llama_status}
  • Gemini Agent: {gemini_status}
  • Vector Memory: {memory_status}{memory_info}
  • LangGraph Workflow: {workflow_status}
  • Sass Level: {self.current_sass_level.value} ({self.current_sass_level.name})
  • Total Tools: {stats['total_tools']}
  • Conversations: {self.conversation_id}"""

        elif cmd == "/memory":
            if not self.memory_store:
                return "❌ Memory store not initialized"

            stats = self.memory_store.get_pattern_stats()
            return f"""🧠 Memory Store Statistics:
  • Total Patterns: {stats.get('total_patterns', 0)}
  • By Language: {stats.get('by_language', {})}
  • By Type: {stats.get('by_type', {})}
  • Most Used: {len(stats.get('most_used', []))} patterns
  • Recent Patterns: {len(stats.get('recent_patterns', []))} patterns"""

        elif cmd == "/help":
            return """🧬 Symbiote Commands:
  • /sass <0-10>  - Set sass/personality level
  • /tools       - List available tools  
  • /status      - Show system status
  • /memory      - Show memory store statistics
  • /analyze     - Analyze current project structure
  • /workflow    - Toggle LangGraph workflow mode
  • /help        - Show this help message
  
  Regular Usage:
  Just type what you want me to do! I can help with:
  • Reading and writing files with deep analysis
  • Generating code with pattern learning
  • Analyzing your codebase with AST
  • Running shell commands intelligently
  • Git operations with smart suggestions
  • Learning from your coding patterns
  • Multi-agent workflows for complex tasks
  • And much more with advanced AI intelligence!"""

        elif cmd == "/analyze":
            return await self._analyze_project()

        elif cmd == "/workflow":
            self.use_langgraph = not self.use_langgraph
            status = "enabled" if self.use_langgraph else "disabled"
            return f"🔄 LangGraph workflow {status}"

        else:
            return f"❌ Unknown command: {cmd}. Type /help for available commands."

    async def _learn_from_interaction(self, user_input: str, response: str):
        """Learn from user interactions and store patterns."""
        if not self.memory_store:
            return

        try:
            # Extract code from response if present
            code_blocks = []
            lines = response.split("\n")
            in_code_block = False
            current_code = []
            current_language = "python"

            for line in lines:
                if line.strip().startswith("```"):
                    if in_code_block:
                        # End of code block
                        if current_code:
                            code_blocks.append(
                                ("\n".join(current_code), current_language)
                            )
                        current_code = []
                        in_code_block = False
                    else:
                        # Start of code block
                        in_code_block = True
                        # Extract language if specified
                        parts = line.strip().split("```")
                        if len(parts) > 1 and parts[1]:
                            current_language = parts[1].strip()
                elif in_code_block:
                    current_code.append(line)

            # Learn from each code block
            for code, language in code_blocks:
                if code.strip():
                    await self.memory_store.learn_from_interaction(
                        user_query=user_input, code_generated=code, language=language
                    )

        except Exception as e:
            print(f"⚠️ Error learning from interaction: {e}")

    async def _analyze_project(self) -> str:
        """Analyze the current project structure using AST analysis."""
        try:
            if not self.tool_registry.get_tool("code_analysis"):
                return "❌ Code analysis tool not available"

            # Analyze current directory for Python files
            import os
            from pathlib import Path

            python_files = []
            for root, dirs, files in os.walk("."):
                # Skip common directories
                dirs[:] = [
                    d
                    for d in dirs
                    if d not in {".git", "__pycache__", ".venv", "venv", "node_modules"}
                ]

                for file in files:
                    if file.endswith(".py"):
                        python_files.append(os.path.join(root, file))

            if not python_files:
                return "📁 No Python files found in current directory"

            analysis_results = []
            total_complexity = 0
            total_functions = 0
            total_classes = 0

            # Analyze first 10 files to avoid overwhelming output
            for file_path in python_files[:10]:
                try:
                    code_tool = self.tool_registry.get_tool("code_analysis")
                    if code_tool:
                        result = await code_tool.execute(
                            {"operation": "analyze_file", "file_path": file_path}
                        )

                        if result.success:
                            analysis = result.data.get("analysis", {})
                            complexity = analysis.get("complexity", {})

                            total_complexity += complexity.get(
                                "cyclomatic_complexity", 0
                            )
                            total_functions += len(analysis.get("functions", []))
                            total_classes += len(analysis.get("classes", []))

                            analysis_results.append(
                                {
                                    "file": file_path,
                                    "functions": len(analysis.get("functions", [])),
                                    "classes": len(analysis.get("classes", [])),
                                    "complexity": complexity.get(
                                        "cyclomatic_complexity", 0
                                    ),
                                }
                            )

                except Exception as e:
                    print(f"⚠️ Error analyzing {file_path}: {e}")

            # Generate summary
            avg_complexity = (
                total_complexity / len(analysis_results) if analysis_results else 0
            )

            summary = f"""📊 Project Analysis Summary:
  • Files Analyzed: {len(analysis_results)} of {len(python_files)} Python files
  • Total Functions: {total_functions}
  • Total Classes: {total_classes}
  • Average Complexity: {avg_complexity:.1f}
  
  Top Complex Files:"""

            # Sort by complexity and show top 5
            sorted_files = sorted(
                analysis_results, key=lambda x: x["complexity"], reverse=True
            )
            for i, file_info in enumerate(sorted_files[:5]):
                summary += f"\n  {i+1}. {file_info['file']} (complexity: {file_info['complexity']})"

            return summary

        except Exception as e:
            return f"❌ Error analyzing project: {e}"

    async def _update_agent_sass_levels(self, level: int):
        """Update sass levels for all agents."""
        if self.llama_agent:
            self.llama_agent.update_sass_level(level)

        if self.gemini_agent:
            self.gemini_agent.update_sass_level(level)

    def get_conversation_history(self) -> list:
        """Get conversation history from LLaMA agent."""
        if self.llama_agent:
            return self.llama_agent.conversation_history
        return []

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        if self.memory_store:
            return self.memory_store.get_pattern_stats()
        return {"error": "Memory store not initialized"}

    async def clear_memory(self):
        """Clear memory store (use with caution!)."""
        if self.memory_store:
            self.memory_store.clear_memory()
            return "🗑️ Memory cleared"
        return "❌ Memory store not available"

    async def shutdown(self):
        """Gracefully shutdown the orchestrator."""
        print("🧬 Shutting down Symbiote...")

        # Save memory if available
        if self.memory_store:
            self.memory_store._save_memory()
            print("💾 Memory saved")

        self.is_initialized = False
        print("✅ Symbiote shutdown complete")
