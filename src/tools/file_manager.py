"""
File management tool for Symbiote.
"""
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from .base_tool import BaseTool, ToolResult, ToolInfo, ToolCategory


class FileManagerTool(BaseTool):
    """Tool for file operations like read, write, create, delete."""
    
    def __init__(self):
        super().__init__(
            name="file_manager",
            description="Manage files: read, write, create, delete, and list files",
            category=ToolCategory.FILE_OPERATIONS
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute file operation."""
        if not self.validate_parameters(parameters):
            return self.create_error_result("Invalid parameters")
        
        operation = parameters.get("operation")
        file_path = parameters.get("file_path")
        
        try:
            if operation == "read":
                return await self._read_file(str(file_path))
            elif operation == "write":
                content = parameters.get("content", "")
                return await self._write_file(str(file_path), content)
            elif operation == "create":
                return await self._create_file(str(file_path))
            elif operation == "delete":
                return await self._delete_file(str(file_path))
            elif operation == "list":
                directory = parameters.get("directory", ".")
                return await self._list_directory(str(directory))
            elif operation == "exists":
                return await self._check_exists(str(file_path))
            else:
                return self.create_error_result(f"Unknown operation: {operation}")
                
        except Exception as e:
            return self.create_error_result(f"File operation failed: {str(e)}")
    
    async def _read_file(self, file_path: str) -> ToolResult:
        """Read file contents."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_info = os.stat(file_path)
            metadata = {
                "size": file_info.st_size,
                "modified": file_info.st_mtime,
                "lines": len(content.splitlines())
            }
            
            return self.create_success_result(
                {"content": content, "file_path": file_path},
                metadata
            )
        except FileNotFoundError:
            return self.create_error_result(f"File not found: {file_path}")
        except PermissionError:
            return self.create_error_result(f"Permission denied: {file_path}")
    
    async def _write_file(self, file_path: str, content: str) -> ToolResult:
        """Write content to file."""
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_info = os.stat(file_path)
            metadata = {
                "size": file_info.st_size,
                "lines_written": len(content.splitlines())
            }
            
            return self.create_success_result(
                {"file_path": file_path, "bytes_written": len(content.encode('utf-8'))},
                metadata
            )
        except PermissionError:
            return self.create_error_result(f"Permission denied: {file_path}")
    
    async def _create_file(self, file_path: str) -> ToolResult:
        """Create empty file."""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            Path(file_path).touch(exist_ok=False)
            
            return self.create_success_result({"file_path": file_path, "created": True})
        except FileExistsError:
            return self.create_error_result(f"File already exists: {file_path}")
        except PermissionError:
            return self.create_error_result(f"Permission denied: {file_path}")
    
    async def _delete_file(self, file_path: str) -> ToolResult:
        """Delete file."""
        try:
            os.remove(file_path)
            return self.create_success_result({"file_path": file_path, "deleted": True})
        except FileNotFoundError:
            return self.create_error_result(f"File not found: {file_path}")
        except PermissionError:
            return self.create_error_result(f"Permission denied: {file_path}")
    
    async def _list_directory(self, directory: str) -> ToolResult:
        """List directory contents."""
        try:
            items = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                item_info = {
                    "name": item,
                    "path": item_path,
                    "is_file": os.path.isfile(item_path),
                    "is_directory": os.path.isdir(item_path),
                    "size": os.path.getsize(item_path) if os.path.isfile(item_path) else 0
                }
                items.append(item_info)
            
            return self.create_success_result(
                {"directory": directory, "items": items},
                {"item_count": len(items)}
            )
        except FileNotFoundError:
            return self.create_error_result(f"Directory not found: {directory}")
        except PermissionError:
            return self.create_error_result(f"Permission denied: {directory}")
    
    async def _check_exists(self, file_path: str) -> ToolResult:
        """Check if file/directory exists."""
        exists = os.path.exists(file_path)
        is_file = os.path.isfile(file_path) if exists else False
        is_directory = os.path.isdir(file_path) if exists else False
        
        return self.create_success_result({
            "file_path": file_path,
            "exists": exists,
            "is_file": is_file,
            "is_directory": is_directory
        })
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        required_params = ["operation"]
        
        for param in required_params:
            if param not in parameters:
                return False
        
        operation = parameters.get("operation")
        valid_operations = ["read", "write", "create", "delete", "list", "exists"]
        
        if operation not in valid_operations:
            return False
        
        # Check operation-specific requirements
        if operation in ["read", "write", "create", "delete", "exists"]:
            if "file_path" not in parameters:
                return False
        
        if operation == "write" and "content" not in parameters:
            return False
        
        return True
    
    def get_info(self) -> ToolInfo:
        """Get tool information."""
        return ToolInfo(
            name=self.name,
            description=self.description,
            category=self.category,
            parameters={
                "operation": {
                    "type": "string",
                    "required": True,
                    "options": ["read", "write", "create", "delete", "list", "exists"],
                    "description": "Type of file operation to perform"
                },
                "file_path": {
                    "type": "string",
                    "required": "for most operations",
                    "description": "Path to the file"
                },
                "content": {
                    "type": "string",
                    "required": "for write operation",
                    "description": "Content to write to file"
                },
                "directory": {
                    "type": "string",
                    "required": "for list operation",
                    "description": "Directory to list (defaults to current directory)"
                }
            },
            examples={
                "read_file": {
                    "operation": "read",
                    "file_path": "/path/to/file.txt"
                },
                "write_file": {
                    "operation": "write",
                    "file_path": "/path/to/file.txt",
                    "content": "Hello, World!"
                },
                "list_directory": {
                    "operation": "list",
                    "directory": "/path/to/directory"
                }
            }
        )
