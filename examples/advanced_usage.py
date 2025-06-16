#!/usr/bin/env python3
"""
Example usage of the enhanced LLaMA agent with modern tool calling patterns.
This demonstrates how the agent now follows OpenAI/Anthropic patterns for tool use.
"""

import asyncio
import os
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.llama_agent import LlamaAgent
from agents.base_agent import AgentConfig, SassLevel, AgentMessage
from tools.tool_registry import ToolRegistry
from tools.file_manager import FileManagerTool
from tools.shell_executor import ShellExecutorTool
from tools.git_manager import GitManagerTool
from tools.code_search import CodeSearchTool


async def demonstrate_advanced_workflow():
    """Demonstrate advanced workflow with proper tool calling."""
    
    # Set up the environment
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Please set GROQ_API_KEY environment variable")
        return
    
    # Create tool registry with all tools
    tool_registry = ToolRegistry()
    
    # Register tools (following Claude patterns)
    await tool_registry.register_tool(FileManagerTool())
    await tool_registry.register_tool(ShellExecutorTool())
    await tool_registry.register_tool(GitManagerTool())
    await tool_registry.register_tool(CodeSearchTool())
    
    # Create agent configuration
    config = AgentConfig(
        name="llama_agent",
        model="llama-3.3-70b-versatile",
        temperature=0.1,  # Lower for more deterministic tool use
        max_tokens=4096,
        sass_level=SassLevel.PROFESSIONAL
    )
    
    # Initialize the enhanced agent
    agent = LlamaAgent(config, tool_registry)
    success = await agent.initialize()
    
    if not success:
        print("❌ Failed to initialize agent")
        return
    
    print("✅ Enhanced LLaMA Agent initialized with Claude-style tool calling")
    print("🔧 Available tools:", ", ".join(tool_registry.list_tools()))
    print()
    
    # Test scenarios following Claude patterns
    test_scenarios = [
        {
            "name": "File Analysis",
            "message": "Please read the README.md file and tell me what this project is about.",
            "expected_tools": ["file_manager"]
        },
        {
            "name": "Code Exploration", 
            "message": "Show me the structure of the src directory and analyze the main Python files.",
            "expected_tools": ["file_manager", "code_search"]
        },
        {
            "name": "Git Status Check",
            "message": "What's the current git status of this repository?",
            "expected_tools": ["git_manager"]
        },
        {
            "name": "Multi-tool Workflow",
            "message": "Check if there are any Python files with TODO comments, and show me the git history for the last 3 commits.",
            "expected_tools": ["code_search", "git_manager"]
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"🧪 Test {i}: {scenario['name']}")
        print(f"📝 Query: {scenario['message']}")
        print(f"🔧 Expected tools: {', '.join(scenario['expected_tools'])}")
        print("-" * 60)
        
        # Create message
        message = AgentMessage(
            sender="user",
            recipient="llama_agent",
            message_type="user_input",
            content={"text": scenario["message"]},
            timestamp="2025-06-16T12:00:00Z"
        )
        
        # Process message
        try:
            response = await agent.process_message(message)
            
            print(f"📤 Response: {response.content.get('text', 'No response')}")
            
            # Show tool results if any
            tool_results = response.content.get('tool_results', {})
            if tool_results:
                print(f"🔧 Tools used: {', '.join(tool_results.keys())}")
            
            print("✅ Test completed successfully")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        print("\n" + "="*80 + "\n")


async def demonstrate_streaming_workflow():
    """Demonstrate streaming responses (Claude-style)."""
    print("🔄 Streaming workflow demonstration")
    print("Note: Current implementation doesn't include streaming, but the architecture supports it")
    print("Following Claude patterns, streaming would work like this:")
    print()
    print("```python")
    print("async with agent.stream_response(message) as stream:")
    print("    async for chunk in stream:")
    print("        if chunk.type == 'text':")
    print("            print(chunk.content, end='', flush=True)")
    print("        elif chunk.type == 'tool_use':")
    print("            result = await execute_tool(chunk)")
    print("            await stream.send_tool_result(result)")
    print("```")
    print()


def compare_with_claude():
    """Show comparison between old implementation and Claude-style patterns."""
    print("📊 Comparison: Old vs Claude-Style Implementation")
    print("="*60)
    
    comparisons = [
        {
            "aspect": "Tool Call Format",
            "old": "Custom JSON with 'tool_calls' array",
            "new": "OpenAI/Claude standard with proper tool_calls objects"
        },
        {
            "aspect": "Tool Definitions", 
            "old": "Basic name/description strings",
            "new": "Proper JSON schemas with LangChain StructuredTool"
        },
        {
            "aspect": "Error Handling",
            "old": "Basic try/catch with custom formatting",
            "new": "Claude-style error responses with tool_result objects"
        },
        {
            "aspect": "Conversation Flow",
            "old": "Single-turn with manual history management",
            "new": "Multi-turn with proper message history and tool results"
        },
        {
            "aspect": "Tool Execution",
            "old": "Sequential execution with custom parsing",
            "new": "Parallel execution following Claude patterns"
        },
        {
            "aspect": "Response Format",
            "old": "Mixed text and tool output formatting",
            "new": "Structured responses with separate tool_result handling"
        }
    ]
    
    for comp in comparisons:
        print(f"🔧 {comp['aspect']}:")
        print(f"   Old: {comp['old']}")
        print(f"   New: {comp['new']}")
        print()


async def main():
    """Main demonstration function."""
    print("🚀 Claude-Style LLaMA Agent Demonstration")
    print("="*60)
    print()
    
    # Show comparison first
    compare_with_claude()
    print("\n" + "="*80 + "\n")
    
    # Run the main workflow demonstration
    await demonstrate_claude_style_workflow()
    
    # Show streaming example
    await demonstrate_streaming_workflow()
    
    print("🎉 Demonstration completed!")
    print()
    print("Key improvements implemented:")
    print("✅ Proper tool calling format (OpenAI/Claude compatible)")
    print("✅ Enhanced error handling and conversation flow")
    print("✅ Multi-turn conversation support with tool results")
    print("✅ Parallel tool execution capability")
    print("✅ Structured tool definitions with proper schemas")
    print("✅ LangChain integration with tool binding")


if __name__ == "__main__":
    asyncio.run(main())
