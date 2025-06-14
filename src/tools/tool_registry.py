"""
Tool registry for managing and organizing Symbiote tools.
"""
from typing import Dict, List, Optional
from .base_tool import BaseTool, ToolCategory, ToolInfo


class ToolRegistry:
    """Registry for managing all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }
    
    def register_tool(self, tool: BaseTool) -> bool:
        """Register a new tool."""
        try:
            if tool.name in self._tools:
                print(f"Warning: Tool '{tool.name}' already registered. Overwriting.")
            
            self._tools[tool.name] = tool
            self._categories[tool.category].append(tool.name)
            print(f"Registered tool: {tool.name} ({tool.category.value})")
            return True
        except Exception as e:
            print(f"Failed to register tool '{tool.name}': {e}")
            return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a specific category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_tool_info(self, name: str) -> Optional[ToolInfo]:
        """Get information about a specific tool."""
        tool = self.get_tool(name)
        return tool.get_info() if tool else None
    
    def get_all_tool_info(self) -> List[ToolInfo]:
        """Get information about all registered tools."""
        return [tool.get_info() for tool in self._tools.values()]
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name not in self._tools:
            return False
        
        tool = self._tools[name]
        del self._tools[name]
        
        if name in self._categories[tool.category]:
            self._categories[tool.category].remove(name)
        
        print(f"Unregistered tool: {name}")
        return True
    
    def clear_category(self, category: ToolCategory):
        """Clear all tools in a specific category."""
        tool_names = self._categories[category].copy()
        for name in tool_names:
            self.unregister_tool(name)
    
    def clear_all(self):
        """Clear all registered tools."""
        self._tools.clear()
        for category in self._categories:
            self._categories[category].clear()
        print("Cleared all tools from registry")
    
    def get_registry_stats(self) -> Dict[str, int]:
        """Get statistics about the tool registry."""
        stats = {
            "total_tools": len(self._tools),
            "categories": {}
        }
        
        for category, tool_names in self._categories.items():
            stats["categories"][category.value] = len(tool_names)
        
        return stats
