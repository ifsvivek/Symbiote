"""
Gemini Code Generation Agent - Handles code generation, explanations, and refactoring.
"""
import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from google import genai
from google.genai import types

from ..agents.base_agent import BaseAgent, AgentMessage, AgentConfig, SassLevel


class GeminiAgent(BaseAgent):
    """Gemini agent specialized for code generation and analysis."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialize the Gemini agent."""
        try:
            # Get API key from environment or config
            api_key = os.getenv("GOOGLE_API_KEY") or self.config.api_key
            if not api_key:
                print(f"[{self.name}] Warning: GOOGLE_API_KEY not set. Code generation will not work.")
                return False
            
            # Initialize Gemini client
            self.client = genai.Client(api_key=api_key)
            print(f"[{self.name}] Gemini agent initialized")
            return True
        except Exception as e:
            print(f"[{self.name}] Failed to initialize: {e}")
            return False
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process code generation requests."""
        self.add_to_history(message)
        
        try:
            if message.message_type == "code_generation_request":
                return await self._handle_code_generation(message)
            elif message.message_type == "code_analysis_request":
                return await self._handle_code_analysis(message)
            elif message.message_type == "code_refactor_request":
                return await self._handle_code_refactor(message)
            else:
                return self._create_error_response(
                    message, 
                    f"Unknown message type: {message.message_type}"
                )
                
        except Exception as e:
            return self._create_error_response(message, str(e))
    
    async def _handle_code_generation(self, message: AgentMessage) -> AgentMessage:
        """Handle code generation requests from LLaMA agent."""
        content = message.content
        user_request = content.get('user_request', '')
        sass_level = content.get('sass_level', 4)
        
        # Build context for code generation
        context = self._build_generation_context(user_request, {}, {})
        
        # Generate code using Gemini
        generated_code = await self._generate_code_with_gemini(context, sass_level)
        
        # Add explanation based on sass level
        explanation = self._generate_explanation(generated_code, sass_level)
        
        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            message_type="code_generation_response",
            content={
                "generated_code": generated_code,
                "explanation": explanation,
                "sass_level": sass_level
            },
            timestamp=datetime.now().isoformat()
        )
    
    async def _generate_code_with_gemini(self, context: Dict[str, Any], sass_level: int) -> str:
        """Generate code using Gemini API."""
        if not self.client:
            return "# Error: Gemini client not initialized"
        
        try:
            # Build system instruction based on sass level
            sass_instruction = self._get_sass_instruction(sass_level)
            system_instruction = f"""You are Symbiote, an expert code generation agent. 
{sass_instruction}

Generate clean, well-documented, production-ready code that follows best practices.
Always include proper type hints, docstrings, and comments where appropriate.
Focus on readability, maintainability, and correctness."""

            # Build user prompt
            user_prompt = f"""Generate code for the following request:

User Request: {context.get('user_request', '')}
Requirements: {', '.join(context.get('requirements', []))}

Please provide clean, well-structured code with appropriate comments and documentation."""

            # Generate content with Gemini
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,  # Lower temperature for more consistent code
                    max_output_tokens=2048,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            return response.text or "# No code generated"
            
        except Exception as e:
            error_msg = f"Error generating code: {str(e)}"
            if sass_level >= 7:
                error_msg = f"Well, that didn't go as planned... {str(e)} 😅"
            return f"# {error_msg}"
    
    async def _handle_code_analysis(self, message: AgentMessage) -> AgentMessage:
        """Handle code analysis requests."""
        content = message.content
        code_to_analyze = content.get('code', '')
        sass_level = content.get('sass_level', 4)
        
        if not self.client:
            return self._create_error_response(message, "Gemini client not initialized")
        
        try:
            system_instruction = f"""You are Symbiote, a code analysis expert.
{self._get_sass_instruction(sass_level)}

Analyze the provided code and provide insights on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Suggestions for improvement
5. Security considerations (if applicable)"""

            user_prompt = f"""Please analyze this code:

```
{code_to_analyze}
```

Provide a comprehensive analysis with specific suggestions for improvement."""

            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.4,
                    max_output_tokens=1024
                )
            )
            
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                message_type="code_analysis_response",
                content={
                    "analysis": response.text,
                    "sass_level": sass_level
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return self._create_error_response(message, str(e))
    
    async def _handle_code_refactor(self, message: AgentMessage) -> AgentMessage:
        """Handle code refactoring requests."""
        content = message.content
        code_to_refactor = content.get('code', '')
        refactor_goals = content.get('goals', 'improve readability and maintainability')
        sass_level = content.get('sass_level', 4)
        
        if not self.client:
            return self._create_error_response(message, "Gemini client not initialized")
        
        try:
            system_instruction = f"""You are Symbiote, a code refactoring expert.
{self._get_sass_instruction(sass_level)}

Refactor the provided code while maintaining its functionality. Focus on:
1. Improving readability and maintainability
2. Following best practices and design patterns
3. Optimizing performance where possible
4. Adding proper documentation and type hints
5. Ensuring code is clean and well-structured"""

            user_prompt = f"""Please refactor this code with the goal of: {refactor_goals}

Original code:
```
{code_to_refactor}
```

Provide the refactored code with explanations of the changes made."""

            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=2048
                )
            )
            
            return AgentMessage(
                sender=self.name,
                recipient=message.sender,
                message_type="code_refactor_response",
                content={
                    "refactored_code": response.text,
                    "sass_level": sass_level
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return self._create_error_response(message, str(e))

    def _build_generation_context(
        self, 
        user_request: str, 
        intent_analysis: Dict[str, Any], 
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context for code generation."""
        context = {
            "user_request": user_request,
            "intent": intent_analysis.get('intent', 'unknown'),
            "requirements": self._extract_requirements(user_request),
            "available_data": tool_results,
            "style_preferences": self._get_style_preferences()
        }
        
        return context
    
    def _extract_requirements(self, user_request: str) -> List[str]:
        """Extract specific requirements from user request."""
        requirements = []
        
        # Simple keyword extraction for now
        if "function" in user_request.lower():
            requirements.append("Create a function")
        if "class" in user_request.lower():
            requirements.append("Create a class")
        if "test" in user_request.lower():
            requirements.append("Include tests")
        if "comment" in user_request.lower() or "document" in user_request.lower():
            requirements.append("Add documentation")
        if "async" in user_request.lower():
            requirements.append("Use async/await")
        if "type hint" in user_request.lower():
            requirements.append("Include type hints")
        
        return requirements
    
    def _get_style_preferences(self) -> Dict[str, Any]:
        """Get user's coding style preferences."""
        return {
            "language": "python",
            "naming_convention": "snake_case",
            "documentation_style": "docstring",
            "type_hints": True,
            "max_line_length": 88,
            "use_async": True
        }
    
    def _get_sass_instruction(self, sass_level: int) -> str:
        """Get sass-appropriate instruction for code generation."""
        if sass_level >= 9:
            return "Add comments with maximum sass and personality. Be brutally honest about code quality and don't hold back your opinions."
        elif sass_level >= 7:
            return "Add sarcastic but helpful comments. Point out potential issues with wit and attitude."
        elif sass_level >= 4:
            return "Add friendly, humorous comments where appropriate. Keep things light and engaging."
        else:
            return "Add professional, clear comments and documentation. Be helpful and courteous."
    
    def _generate_explanation(self, code: str, sass_level: int) -> str:
        """Generate explanation for the generated code."""
        base_explanation = f"I've generated the code you requested. "
        
        if sass_level >= 9:
            return base_explanation + "It's actually decent code, not that you asked for my opinion. Try not to mess it up! 😤"
        elif sass_level >= 7:
            return base_explanation + "Try not to break it immediately, okay? I put actual effort into this. 😏"
        elif sass_level >= 4:
            return base_explanation + "Hope it does what you had in mind! Let me know if you need any tweaks. 😊"
        else:
            return base_explanation + "The code follows best practices and should meet your requirements. Please review and test as needed."
    
    def _create_error_response(self, message: AgentMessage, error: str) -> AgentMessage:
        """Create an error response message."""
        sass_error = error
        if self.sass_level.value >= 7:
            sass_error = f"Well, this is awkward... {error} 😅"
        
        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            message_type="error",
            content={"error": sass_error, "original_error": error},
            timestamp=datetime.now().isoformat()
        )
    
    def _create_placeholder_response(self, message: AgentMessage, placeholder_text: str) -> AgentMessage:
        """Create a placeholder response for unimplemented features."""
        return AgentMessage(
            sender=self.name,
            recipient=message.sender,
            message_type="placeholder",
            content={"text": placeholder_text},
            timestamp=datetime.now().isoformat()
        )
