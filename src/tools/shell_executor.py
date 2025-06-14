"""
Shell executor tool for Symbiote.
"""
import asyncio
import shlex
from typing import Dict, Any, Optional

from .base_tool import BaseTool, ToolResult, ToolInfo, ToolCategory


class ShellExecutorTool(BaseTool):
    """Tool for executing shell commands safely."""
    
    def __init__(self):
        super().__init__(
            name="shell_executor",
            description="Execute shell commands safely with timeout and output capture",
            category=ToolCategory.SHELL_OPERATIONS
        )
        # List of dangerous commands to block
        self.blocked_commands = {
            'rm', 'rmdir', 'del', 'delete', 'format', 'fdisk', 'mkfs',
            'dd', 'shutdown', 'reboot', 'halt', 'poweroff', 'init',
            'sudo', 'su', 'chmod', 'chown', 'passwd', 'useradd', 'userdel'
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute shell command."""
        if not self.validate_parameters(parameters):
            return self.create_error_result("Invalid parameters")
        
        command = parameters.get("command", "")
        timeout = parameters.get("timeout", 30)  # Default 30 second timeout
        working_dir = parameters.get("working_dir", ".")
        
        # Safety check
        if not self._is_command_safe(command):
            return self.create_error_result(f"Command blocked for safety: {command}")
        
        try:
            return await self._execute_command(command, timeout, working_dir)
        except Exception as e:
            return self.create_error_result(f"Shell execution failed: {str(e)}")
    
    def _is_command_safe(self, command: str) -> bool:
        """Check if command is safe to execute."""
        if not command.strip():
            return False
        
        # Parse command to get the base command
        try:
            parts = shlex.split(command)
            if not parts:
                return False
            
            base_command = parts[0].split('/')[-1]  # Get command name without path
            
            # Check against blocked commands
            if base_command.lower() in self.blocked_commands:
                return False
            
            # Block commands with dangerous flags
            dangerous_patterns = ['--force', '-f', '--recursive', '-r', '--delete']
            command_lower = command.lower()
            if any(pattern in command_lower for pattern in dangerous_patterns):
                if any(blocked in command_lower for blocked in ['rm', 'del', 'delete']):
                    return False
            
            return True
            
        except Exception:
            # If we can't parse it safely, block it
            return False
    
    async def _execute_command(self, command: str, timeout: int, working_dir: str) -> ToolResult:
        """Execute the command with proper error handling."""
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return self.create_error_result(f"Command timed out after {timeout} seconds")
            
            # Decode output
            stdout_text = stdout.decode('utf-8', errors='replace').strip()
            stderr_text = stderr.decode('utf-8', errors='replace').strip()
            
            # Prepare result
            result_data = {
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "working_dir": working_dir
            }
            
            metadata = {
                "execution_time": timeout,
                "success": process.returncode == 0
            }
            
            if process.returncode == 0:
                return self.create_success_result(result_data, metadata)
            else:
                return self.create_error_result(
                    f"Command failed with exit code {process.returncode}: {stderr_text}",
                    result_data
                )
                
        except FileNotFoundError:
            return self.create_error_result(f"Command not found: {command}")
        except PermissionError:
            return self.create_error_result(f"Permission denied: {command}")
        except Exception as e:
            return self.create_error_result(f"Execution error: {str(e)}")
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if "command" not in parameters:
            return False
        
        command = parameters.get("command", "")
        if not command or not command.strip():
            return False
        
        # Validate timeout if provided
        timeout = parameters.get("timeout", 30)
        if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 300:
            return False
        
        return True
    
    def get_info(self) -> ToolInfo:
        """Get tool information."""
        return ToolInfo(
            name=self.name,
            description=self.description,
            category=self.category,
            parameters={
                "command": {
                    "type": "string",
                    "required": True,
                    "description": "Shell command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "required": False,
                    "description": "Timeout in seconds (default: 30, max: 300)"
                },
                "working_dir": {
                    "type": "string",
                    "required": False,
                    "description": "Working directory for command execution (default: current directory)"
                }
            },
            examples={
                "list_files": {
                    "command": "ls -la",
                    "timeout": 10
                },
                "check_python": {
                    "command": "python --version"
                },
                "run_script": {
                    "command": "python script.py",
                    "working_dir": "/path/to/project",
                    "timeout": 60
                }
            }
        )
