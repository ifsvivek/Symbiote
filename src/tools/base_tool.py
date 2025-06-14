"""
Base tool interface for Symbiote tool system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ToolCategory(Enum):
    """Categories for organizing tools."""
    FILE_OPERATIONS = "file_operations"
    GIT_OPERATIONS = "git_operations"
    SHELL_OPERATIONS = "shell_operations"
    CODE_ANALYSIS = "code_analysis"
    SEARCH = "search"
    UTILITY = "utility"


@dataclass
class ToolResult:
    """Standardized result format for tool execution."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolInfo:
    """Information about a tool."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    examples: Optional[Dict[str, Any]] = None


class BaseTool(ABC):
    """Base class for all Symbiote tools."""
    
    def __init__(self, name: str, description: str, category: ToolCategory):
        self.name = name
        self.description = description
        self.category = category
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_info(self) -> ToolInfo:
        """Get information about this tool."""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        pass
    
    def create_success_result(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Helper to create a successful result."""
        return ToolResult(success=True, data=data, metadata=metadata)
    
    def create_error_result(self, error: str, data: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Helper to create an error result."""
        return ToolResult(success=False, data=data or {}, error=error)
