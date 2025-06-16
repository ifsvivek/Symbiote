"""
Git management tool for Symbiote.
"""
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from .base_tool import BaseTool, ToolResult, ToolInfo, ToolCategory


class GitManagerTool(BaseTool):
    """Tool for git operations like status, commit, push, etc."""
    
    def __init__(self):
        super().__init__(
            name="git_manager",
            description="Manage git operations: status, add, commit, push, pull, diff, log",
            category=ToolCategory.GIT_OPERATIONS
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute git operation."""
        if not self.validate_parameters(parameters):
            return self.create_error_result("Invalid parameters")
        
        operation = parameters.get("operation")
        
        try:
            if operation == "status":
                return await self._git_status()
            elif operation == "add":
                files = parameters.get("files", ".")
                return await self._git_add(files)
            elif operation == "commit":
                message = parameters.get("message", "")
                return await self._git_commit(message)
            elif operation == "push":
                branch = parameters.get("branch", "")
                return await self._git_push(branch)
            elif operation == "pull":
                return await self._git_pull()
            elif operation == "diff":
                return await self._git_diff()
            elif operation == "log":
                count = parameters.get("count", 5)
                return await self._git_log(count)
            elif operation == "branch":
                return await self._git_branch()
            elif operation == "checkout":
                branch = parameters.get("branch", "")
                create_new = parameters.get("create_new", False)
                return await self._git_checkout(branch, create_new)
            else:
                return self.create_error_result(f"Unknown operation: {operation}")
                
        except Exception as e:
            return self.create_error_result(f"Git operation failed: {str(e)}")
    
    async def _run_git_command(self, args: list) -> ToolResult:
        """Run a git command and return the result."""
        try:
            process = await asyncio.create_subprocess_exec(
                "git", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="."
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                output = stdout.decode('utf-8').strip()
                return self.create_success_result(
                    {"output": output, "command": f"git {' '.join(args)}"},
                    {"return_code": process.returncode}
                )
            else:
                error = stderr.decode('utf-8').strip()
                return self.create_error_result(f"Git command failed: {error}")
                
        except FileNotFoundError:
            return self.create_error_result("Git is not installed or not in PATH")
        except Exception as e:
            return self.create_error_result(f"Error running git command: {str(e)}")
    
    async def _git_status(self) -> ToolResult:
        """Get git status."""
        return await self._run_git_command(["status", "--porcelain"])
    
    async def _git_add(self, files: str) -> ToolResult:
        """Add files to git staging area."""
        if isinstance(files, list):
            files = " ".join(files)
        return await self._run_git_command(["add"] + files.split())
    
    async def _git_commit(self, message: str) -> ToolResult:
        """Commit staged changes."""
        if not message:
            return self.create_error_result("Commit message is required")
        return await self._run_git_command(["commit", "-m", message])
    
    async def _git_push(self, branch: str = "") -> ToolResult:
        """Push changes to remote repository."""
        if branch:
            return await self._run_git_command(["push", "origin", branch])
        else:
            return await self._run_git_command(["push"])
    
    async def _git_pull(self) -> ToolResult:
        """Pull changes from remote repository."""
        return await self._run_git_command(["pull"])
    
    async def _git_diff(self) -> ToolResult:
        """Show git diff."""
        return await self._run_git_command(["diff"])
    
    async def _git_log(self, count: int) -> ToolResult:
        """Show git log."""
        return await self._run_git_command(["log", "--oneline", f"-{count}"])
    
    async def _git_branch(self) -> ToolResult:
        """Show git branches."""
        return await self._run_git_command(["branch", "-a"])
    
    async def _git_checkout(self, branch: str, create_new: bool = False) -> ToolResult:
        """Checkout or create a new branch."""
        if not branch:
            return self.create_error_result("Branch name is required")
        
        if create_new:
            return await self._run_git_command(["checkout", "-b", branch])
        else:
            return await self._run_git_command(["checkout", branch])

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if "operation" not in parameters:
            return False
        
        operation = parameters.get("operation")
        valid_operations = ["status", "add", "commit", "push", "pull", "diff", "log", "branch"]
        
        if operation not in valid_operations:
            return False
        
        # Check operation-specific requirements
        if operation == "commit" and "message" not in parameters:
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
                    "options": ["status", "add", "commit", "push", "pull", "diff", "log", "branch"],
                    "description": "Type of git operation to perform"
                },
                "files": {
                    "type": "string",
                    "required": False,
                    "description": "Files to add (for add operation)"
                },
                "message": {
                    "type": "string",
                    "required": "for commit operation",
                    "description": "Commit message"
                },
                "count": {
                    "type": "integer",
                    "required": False,
                    "description": "Number of log entries to show (default: 5)"
                }
            },
            examples={
                "git_status": {
                    "operation": "status"
                },
                "git_commit": {
                    "operation": "commit",
                    "message": "Add new feature"
                },
                "git_add": {
                    "operation": "add",
                    "files": "src/main.py"
                }
            }
        )
