"""
Main orchestrator for Symbiote multi-agent system.
"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .agents.base_agent import AgentMessage, AgentConfig, SassLevel
from .agents.llama_agent import LlamaAgent
from .agents.gemini_agent import GeminiAgent
from .tools.tool_registry import ToolRegistry
from .tools.file_manager import FileManagerTool


class SymbioteOrchestrator:
    """Main orchestrator that manages agents and handles user interactions."""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.llama_agent: Optional[LlamaAgent] = None
        self.gemini_agent: Optional[GeminiAgent] = None
        self.current_sass_level = SassLevel.FRIENDLY
        self.conversation_id = 0
        
    async def initialize(
        self, 
        llama_config: Optional[AgentConfig] = None,
        gemini_config: Optional[AgentConfig] = None
    ) -> bool:
        """Initialize the orchestrator and all agents."""
        print("🧬 Initializing Symbiote...")
        
        # Set up default configs if not provided
        if not llama_config:
            llama_config = AgentConfig(
                name="llama_orchestrator",
                model_name="llama-3.3-70b",
                sass_level=self.current_sass_level
            )
        
        if not gemini_config:
            gemini_config = AgentConfig(
                name="gemini_coder",
                model_name="gemini-2.0-flash-exp",
                sass_level=self.current_sass_level
            )
        
        # Register core tools
        await self._register_core_tools()
        
        # Initialize agents
        self.llama_agent = LlamaAgent(llama_config, self.tool_registry)
        self.gemini_agent = GeminiAgent(gemini_config)
        
        # Connect agents
        self.llama_agent.set_gemini_agent(self.gemini_agent)
        
        # Initialize both agents
        llama_success = await self.llama_agent.initialize()
        gemini_success = await self.gemini_agent.initialize()
        
        if llama_success and gemini_success:
            print("✅ Symbiote initialized successfully!")
            print(f"🎭 Current sass level: {self.current_sass_level.value} ({self.current_sass_level.name})")
            return True
        else:
            print("❌ Failed to initialize Symbiote")
            return False
    
    async def _register_core_tools(self):
        """Register core tools with the tool registry."""
        print("🔧 Registering core tools...")
        
        # Register file manager tool
        from .tools.file_manager import FileManagerTool
        file_tool = FileManagerTool()
        self.tool_registry.register_tool(file_tool)
        
        # Register git manager tool
        from .tools.git_manager import GitManagerTool
        git_tool = GitManagerTool()
        self.tool_registry.register_tool(git_tool)
        
        # Register shell executor tool
        from .tools.shell_executor import ShellExecutorTool
        shell_tool = ShellExecutorTool()
        self.tool_registry.register_tool(shell_tool)
        
        # Register code search tool
        from .tools.code_search import CodeSearchTool
        search_tool = CodeSearchTool()
        self.tool_registry.register_tool(search_tool)
        
        print(f"✅ Registered {len(self.tool_registry.list_tools())} tools")
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input and return response."""
        self.conversation_id += 1
        
        # Check for special commands
        if user_input.startswith('/'):
            return await self._handle_special_command(user_input)
        
        # Create message for LLaMA agent
        message = AgentMessage(
            sender="user",
            recipient="llama_orchestrator",
            message_type="user_input",
            content={"text": user_input},
            timestamp=datetime.now().isoformat(),
            context={"conversation_id": self.conversation_id}
        )
        
        # Process with LLaMA agent
        if not self.llama_agent:
            return "❌ Symbiote not initialized. Please run initialize() first."
        
        response = await self.llama_agent.process_message(message)
        
        # Extract response text
        if response.message_type == "error":
            return f"❌ {response.content.get('text', 'An error occurred')}"
        else:
            return response.content.get('text', 'No response generated')
    
    async def _handle_special_command(self, command: str) -> str:
        """Handle special slash commands."""
        parts = command.strip().split()
        cmd = parts[0].lower()
        
        if cmd == '/sass':
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
        
        elif cmd == '/tools':
            tools = self.tool_registry.list_tools()
            if not tools:
                return "No tools registered"
            
            tool_list = "\n".join([f"  • {tool}" for tool in tools])
            return f"🔧 Available tools:\n{tool_list}"
        
        elif cmd == '/status':
            stats = self.tool_registry.get_registry_stats()
            llama_status = "✅" if self.llama_agent else "❌"
            gemini_status = "✅" if self.gemini_agent else "❌"
            
            return f"""🧬 Symbiote Status:
  • LLaMA Agent: {llama_status}
  • Gemini Agent: {gemini_status}
  • Sass Level: {self.current_sass_level.value} ({self.current_sass_level.name})
  • Total Tools: {stats['total_tools']}
  • Conversations: {self.conversation_id}"""
        
        elif cmd == '/help':
            return """🧬 Symbiote Commands:
  • /sass <0-10>  - Set sass/personality level
  • /tools       - List available tools
  • /status      - Show system status
  • /help        - Show this help message
  
  Regular Usage:
  Just type what you want me to do! I can help with:
  • Reading and writing files
  • Generating code
  • Analyzing your codebase
  • Running shell commands
  • Git operations
  • And much more!"""
        
        else:
            return f"❌ Unknown command: {cmd}. Type /help for available commands."
    
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
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator."""
        print("🧬 Shutting down Symbiote...")
        # TODO: Implement proper shutdown procedures
        print("✅ Symbiote shutdown complete")
