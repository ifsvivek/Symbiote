"""
LangGraph-based workflow orchestrator for multi-agent interactions.
Implements advanced agentic workflows with state management.
"""

import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from ..agents.base_agent import AgentMessage, AgentConfig
from ..agents.llama_agent import LlamaAgent
from ..agents.gemini_agent import GeminiAgent
from ..tools.tool_registry import ToolRegistry
from ..memory.vector_store import VectorMemoryStore


class WorkflowState(TypedDict):
    """State structure for LangGraph workflow."""

    messages: Annotated[List[BaseMessage], "conversation messages"]
    user_query: str
    current_agent: str
    tool_results: Dict[str, Any]
    code_context: Dict[str, Any]
    memory_patterns: List[Dict[str, Any]]
    workflow_step: str
    error_state: Optional[str]
    final_response: Optional[str]
    agent_response: Optional[str]


class SymbioteWorkflow:
    """
    LangGraph-based workflow orchestrator that manages multi-agent interactions
    with advanced AI patterns, with intelligent routing and state management.
    """

    def __init__(self, tool_registry: ToolRegistry, memory_store: VectorMemoryStore, verbose: bool = True):
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.llama_agent: Optional[LlamaAgent] = None
        self.gemini_agent: Optional[GeminiAgent] = None
        self.verbose = verbose

        # LangGraph components
        self.workflow = None
        self.checkpointer = MemorySaver()

        # Build the workflow graph
        self._build_workflow()

    def _vprint(self, message: str):
        """Print message only if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def set_verbose(self, verbose: bool):
        """Set verbose mode for workflow debugging output."""
        self.verbose = verbose

    def set_agents(self, llama_agent: LlamaAgent, gemini_agent: GeminiAgent):
        """Set the agents for the workflow."""
        self.llama_agent = llama_agent
        self.gemini_agent = gemini_agent

    def _build_workflow(self):
        """Build the LangGraph workflow with advanced agentic patterns."""
        # Create the state graph
        workflow = StateGraph(WorkflowState)

        # Add nodes (workflow steps)
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("retrieve_patterns", self._retrieve_patterns)
        workflow.add_node("route_to_agent", self._route_to_agent)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("generate_code", self._generate_code)
        workflow.add_node("write_file", self._write_file)
        workflow.add_node("get_agent_response", self._get_agent_response)
        workflow.add_node("review_and_refine", self._review_and_refine)
        workflow.add_node("learn_patterns", self._learn_patterns)
        workflow.add_node("finalize_response", self._finalize_response)

        # Define the workflow edges (routing logic)
        workflow.set_entry_point("analyze_query")

        workflow.add_edge("analyze_query", "retrieve_patterns")
        workflow.add_edge("retrieve_patterns", "route_to_agent")

        # Conditional routing based on query type
        workflow.add_conditional_edges(
            "route_to_agent",
            self._should_use_tools,
            {
                "use_tools": "execute_tools",
                "generate_code": "generate_code",
                "create_file": "generate_code",
                "direct_response": "get_agent_response",
            },
        )

        workflow.add_edge("execute_tools", "review_and_refine")
        
        # Conditional routing after code generation
        workflow.add_conditional_edges(
            "generate_code",
            self._should_write_file,
            {
                "write_file": "write_file",
                "skip_write": "review_and_refine",
            },
        )
        
        workflow.add_edge("write_file", "review_and_refine")
        workflow.add_edge("get_agent_response", "review_and_refine")
        workflow.add_edge("review_and_refine", "learn_patterns")
        workflow.add_edge("learn_patterns", "finalize_response")
        workflow.add_edge("finalize_response", END)

        # Compile the workflow
        self.workflow = workflow.compile(checkpointer=self.checkpointer)

    def _should_write_file(self, state: WorkflowState) -> str:
        """Determine if generated code should be written to a file."""
        user_query = state["user_query"].lower()
        intent = state["code_context"].get("intent", "")
        
        # Check if user wants to create a file
        create_indicators = ["create file", "write file", "save to file", "make a file", "create a", "write to"]
        
        if intent == "create_file" or any(indicator in user_query for indicator in create_indicators):
            return "write_file"
        else:
            return "skip_write"

    async def _write_file(self, state: WorkflowState) -> WorkflowState:
        """Write generated code to a file."""
        print("📝 Writing generated code to file...")
        
        try:
            generated_code = state["code_context"].get("generated_code", "")
            user_query = state["user_query"]
            
            if not generated_code:
                state["error_state"] = "No code to write"
                return state
            
            # Clean the generated code from markdown syntax
            clean_code = self._clean_code_from_markdown(generated_code)
            
            # Extract filename from user query
            filename = self._extract_filename(user_query)
            if not filename:
                # Generate a default filename based on the code content
                language = state["code_context"].get("language", "python")
                if "def hello_world" in clean_code:
                    filename = "hello_world.py"
                elif "class " in clean_code:
                    # Extract class name for filename
                    import re
                    class_match = re.search(r'class\s+(\w+)', clean_code)
                    if class_match:
                        filename = f"{class_match.group(1).lower()}.py"
                    else:
                        filename = "generated_class.py"
                else:
                    filename = f"generated_code.{self._get_file_extension(language)}"
            
            # Use file manager tool to write the file
            file_tool = self.tool_registry.get_tool("file_manager")
            if file_tool:
                result = await file_tool.execute({
                    "operation": "write",
                    "file_path": filename,
                    "content": clean_code
                })
                
                if result.success:
                    state["code_context"]["written_file"] = filename
                    print(f"✅ Code written to {filename}")
                else:
                    state["error_state"] = f"Failed to write file: {result.error}"
            else:
                state["error_state"] = "File manager tool not available"
                
        except Exception as e:
            print(f"❌ Error writing file: {e}")
            state["error_state"] = f"File writing failed: {e}"
        
        state["workflow_step"] = "file_written"
        return state

    def _clean_code_from_markdown(self, generated_code: str) -> str:
        """Remove markdown syntax from generated code."""
        import re
        
        # Remove triple backticks and language specifications
        # Pattern to match ```python, ```javascript, ```java, etc.
        code = re.sub(r'```\w*\n?', '', generated_code)
        code = re.sub(r'```', '', code)
        
        # Remove any remaining markdown code blocks
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are just backticks or language specifiers
            if line.strip() in ['```', '```python', '```javascript', '```java', '```cpp', '```go', '```rust']:
                continue
            cleaned_lines.append(line)
        
        # Join back and strip any leading/trailing whitespace
        cleaned_code = '\n'.join(cleaned_lines).strip()
        
        return cleaned_code

    def _extract_filename(self, query: str) -> str:
        """Extract filename from user query."""
        import re
        
        # Look for patterns like "create file main.py", "write to utils.js", etc.
        patterns = [
            r'(?:create|write|save).*?(?:file|to)\s+([a-zA-Z0-9_\-\.]+\.[a-zA-Z]+)',
            r'(?:file|name|called)\s+([a-zA-Z0-9_\-\.]+\.[a-zA-Z]+)',
            r'([a-zA-Z0-9_\-\.]+\.[a-zA-Z]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return ""

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for programming language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "go": "go",
            "rust": "rs",
            "php": "php"
        }
        return extensions.get(language, "txt")

    async def _analyze_query(self, state: WorkflowState) -> WorkflowState:
        """Analyze the user query to understand intent and requirements."""
        user_query = state["user_query"]

        self._vprint(f"🔍 Analyzing query: {user_query[:50]}...")

        # Simple intent analysis (can be enhanced with ML)
        intent_keywords = {
            "code_generation": ["write", "create", "generate", "build", "implement"],
            "create_file": ["create file", "write file", "save to file", "make a file"],
            "file_operations": ["read", "write", "delete", "list", "this project", "README", "about this", "what is this"],
            "git_operations": ["commit", "push", "pull", "branch", "status"],
            "code_analysis": ["analyze", "explain", "review", "understand"],
            "help": ["help", "how to", "what is", "explain"],
        }

        detected_intent = "general"
        # Check for create_file first (more specific)
        if any(keyword in user_query.lower() for keyword in intent_keywords["create_file"]):
            detected_intent = "create_file"
        else:
            for intent, keywords in intent_keywords.items():
                if intent != "create_file" and any(keyword in user_query.lower() for keyword in keywords):
                    detected_intent = intent
                    break

        # Update state
        state["code_context"] = {
            "intent": detected_intent,
            "requires_tools": detected_intent in ["file_operations", "git_operations"],
            "requires_code_gen": detected_intent in ["code_generation", "create_file"],
            "requires_file_write": detected_intent == "create_file",
            "language": self._detect_language(user_query),
        }
        state["workflow_step"] = "query_analyzed"

        return state

    async def _retrieve_patterns(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant patterns from memory store."""
        user_query = state["user_query"]
        language = state["code_context"].get("language", "python")

        self._vprint("🧠 Retrieving relevant patterns from memory...")

        try:
            if not self.memory_store:
                print("⚠️ Memory store not available - skipping pattern retrieval")
                state["memory_patterns"] = []
                state["workflow_step"] = "patterns_retrieved"
                return state
                
            # Find similar patterns
            similar_patterns = await self.memory_store.find_similar_patterns(
                query=user_query, top_k=3, language=language
            )

            # Format patterns for context
            memory_patterns = []
            for pattern, score in similar_patterns:
                memory_patterns.append(
                    {
                        "id": pattern.pattern_id,
                        "type": pattern.pattern_type,
                        "description": pattern.description,
                        "code": pattern.code_snippet,
                        "similarity": score,
                        "usage_count": pattern.usage_count,
                    }
                )

            state["memory_patterns"] = memory_patterns
            self._vprint(f"📚 Found {len(memory_patterns)} relevant patterns")

        except Exception as e:
            print(f"⚠️ Error retrieving patterns: {e}")
            state["memory_patterns"] = []

        state["workflow_step"] = "patterns_retrieved"
        return state

    async def _route_to_agent(self, state: WorkflowState) -> WorkflowState:
        """Route to appropriate agent based on intent."""
        intent = state["code_context"]["intent"]

        if intent in ["file_operations", "git_operations"]:
            state["current_agent"] = "llama"
        elif intent == "code_generation":
            state["current_agent"] = "gemini"
        else:
            state["current_agent"] = "llama"  # Default

        self._vprint(f"🎯 Routing to {state['current_agent']} agent for {intent}")
        state["workflow_step"] = "agent_routed"
        return state

    def _should_use_tools(self, state: WorkflowState) -> str:
        """Determine if tools should be used."""
        requires_tools = state["code_context"].get("requires_tools", False)
        requires_code_gen = state["code_context"].get("requires_code_gen", False)
        intent = state["code_context"].get("intent", "")

        if requires_tools:
            return "use_tools"
        elif intent == "create_file" or requires_code_gen:
            return "generate_code"
        else:
            return "direct_response"

    async def _execute_tools(self, state: WorkflowState) -> WorkflowState:
        """Execute tools using LLaMA agent."""
        if not self.llama_agent:
            state["error_state"] = "LLaMA agent not available"
            return state

        self._vprint("🔧 Executing tools...")

        try:
            # Create message for LLaMA agent
            message = AgentMessage(
                sender="workflow",
                recipient="llama_agent",
                message_type="user_input",
                content={"text": state["user_query"]},
                timestamp=datetime.now().isoformat(),
            )

            # Process with LLaMA agent
            response = await self.llama_agent.process_message(message)

            # Extract tool results
            tool_results = response.content.get("tool_results", {})
            state["tool_results"] = tool_results
            
            # If LLM response contains useful information from tools, store it as agent response
            response_text = response.content.get("text", "")
            if response_text and any(keyword in response_text.lower() for keyword in ["symbiote", "project", "based on", "readme"]):
                state["agent_response"] = response_text

            print(f"✅ Executed {len(tool_results)} tool operations")

        except Exception as e:
            print(f"❌ Error executing tools: {e}")
            state["error_state"] = f"Tool execution failed: {e}"

        state["workflow_step"] = "tools_executed"
        return state

    async def _generate_code(self, state: WorkflowState) -> WorkflowState:
        """Generate code using Gemini agent with context."""
        if not self.gemini_agent:
            state["error_state"] = "Gemini agent not available"
            return state

        self._vprint("💻 Generating code with context...")

        try:
            # Build enhanced context
            context = {
                "user_query": state["user_query"],
                "tool_results": state.get("tool_results", {}),
                "memory_patterns": state.get("memory_patterns", []),
                "language": state["code_context"].get("language", "python"),
            }

            # Create message for Gemini agent
            message = AgentMessage(
                sender="workflow",
                recipient="gemini_agent",
                message_type="code_generation_request",
                content={
                    "user_request": state["user_query"],
                    "context": context,
                    "sass_level": 4,
                },
                timestamp=datetime.now().isoformat(),
            )

            # Process with Gemini agent
            response = await self.gemini_agent.process_message(message)

            # Store generated code
            state["code_context"]["generated_code"] = response.content.get(
                "generated_code", ""
            )
            state["code_context"]["explanation"] = response.content.get(
                "explanation", ""
            )

            self._vprint("✅ Code generated successfully")

        except Exception as e:
            print(f"❌ Error generating code: {e}")
            state["error_state"] = f"Code generation failed: {e}"

        state["workflow_step"] = "code_generated"
        return state

    async def _get_agent_response(self, state: WorkflowState) -> WorkflowState:
        """Get a direct response from the appropriate agent for simple queries."""
        current_agent = state["current_agent"]
        user_query = state["user_query"]

        self._vprint(f"💬 Getting direct response from {current_agent} agent...")

        try:
            if current_agent == "llama" and self.llama_agent:
                # Create message for LLaMA agent
                message = AgentMessage(
                    sender="workflow",
                    recipient="llama_agent",
                    message_type="user_input",
                    content={"text": user_query},
                    timestamp=datetime.now().isoformat(),
                )

                # Process with LLaMA agent
                response = await self.llama_agent.process_message(message)
                state["agent_response"] = response.content.get("text", "No response generated")

            elif current_agent == "gemini" and self.gemini_agent:
                # Create message for Gemini agent
                message = AgentMessage(
                    sender="workflow",
                    recipient="gemini_agent",
                    message_type="user_input",
                    content={"text": user_query},
                    timestamp=datetime.now().isoformat(),
                )

                # Process with Gemini agent
                response = await self.gemini_agent.process_message(message)
                state["agent_response"] = response.content.get("text", "No response generated")

            else:
                state["agent_response"] = "I'm here to help! What would you like me to do?"

            self._vprint("✅ Got agent response")

        except Exception as e:
            print(f"❌ Error getting agent response: {e}")
            state["error_state"] = f"Agent response failed: {e}"

        state["workflow_step"] = "agent_response_received"
        return state

    async def _review_and_refine(self, state: WorkflowState) -> WorkflowState:
        """Review and refine the generated output."""
        self._vprint("🔍 Reviewing and refining output...")

        # Simple quality checks
        generated_code = state["code_context"].get("generated_code", "")
        if generated_code:
            # Check for basic syntax (very basic)
            quality_score = len(generated_code) / 100.0  # Simple heuristic
            state["code_context"]["quality_score"] = min(quality_score, 1.0)

            # Add refinement suggestions
            refinements = []
            if (
                "def " not in generated_code
                and "function" in state["user_query"].lower()
            ):
                refinements.append("Consider adding function definitions")
            if (
                "class " not in generated_code
                and "class" in state["user_query"].lower()
            ):
                refinements.append("Consider adding class structure")

            state["code_context"]["refinements"] = refinements

        state["workflow_step"] = "reviewed_and_refined"
        return state

    async def _learn_patterns(self, state: WorkflowState) -> WorkflowState:
        """Learn from the interaction and store patterns."""
        self._vprint("🧠 Learning from interaction...")

        try:
            generated_code = state["code_context"].get("generated_code", "")
            language = state["code_context"].get("language", "python")

            if generated_code:
                # Learn from this interaction
                pattern_id = await self.memory_store.learn_from_interaction(
                    user_query=state["user_query"],
                    code_generated=generated_code,
                    language=language,
                )

                if pattern_id:
                    print(f"📚 Learned new pattern: {pattern_id}")
                    state["code_context"]["learned_pattern_id"] = pattern_id

        except Exception as e:
            print(f"⚠️ Error learning patterns: {e}")

        state["workflow_step"] = "patterns_learned"
        return state

    async def _finalize_response(self, state: WorkflowState) -> WorkflowState:
        """Finalize the response to the user."""
        self._vprint("📝 Finalizing response...")

        # Build final response
        response_parts = []

        if state.get("error_state"):
            response_parts.append(f"❌ Error: {state['error_state']}")
        else:
            # Check if we have a direct response from an agent
            if state.get("agent_response"):
                response_parts.append(state["agent_response"])
            
            # Process tool results to extract useful information
            tool_results = state.get("tool_results", {})
            if tool_results:
                successful_tools = [r for r in tool_results.values() if r.get("success")]
                if successful_tools:
                    # Check if any tool returned file content for project questions
                    for tool_name, result in tool_results.items():
                        if result.get("success") and "file_manager" in tool_name:
                            data = result.get("data", {})
                            if "content" in data:
                                file_content = data["content"]
                                file_path = data.get("file_path", "")
                                
                                # If it's a README file, extract the relevant information
                                if "readme" in file_path.lower():
                                    # Extract the main description from README
                                    lines = file_content.split('\n')
                                    description_lines = []
                                    in_description = False
                                    
                                    for line in lines[:50]:  # First 50 lines should contain the main info
                                        line = line.strip()
                                        if line.startswith('# '):
                                            description_lines.append(f"**{line[2:]}**")
                                            in_description = True
                                        elif line.startswith('> ') and in_description:
                                            description_lines.append(line[2:])
                                        elif line and not line.startswith('#') and in_description:
                                            description_lines.append(line)
                                            if len(description_lines) > 5:  # Enough description
                                                break
                                    
                                    if description_lines:
                                        response_parts.append("Based on the README file, here's what this project is about:\n\n" + 
                                                            "\n".join(description_lines))
                                    else:
                                        response_parts.append(f"I found the README file. Here's a summary of the project:\n\n{file_content[:500]}...")
                                else:
                                    # For other files, provide a brief summary
                                    response_parts.append(f"📄 Content from {file_path}:\n{file_content[:300]}...")
                    
                    # Add tool summary if no file content was processed
                    if not any("📄" in part or "Based on" in part for part in response_parts):
                        response_parts.append(f"🔧 Executed {len(successful_tools)} tool operations successfully")

            # Add generated code
            generated_code = state["code_context"].get("generated_code", "")
            if generated_code:
                explanation = state["code_context"].get("explanation", "")
                response_parts.append(
                    f"💻 Generated code:\n```{state['code_context'].get('language', 'python')}\n{generated_code}\n```"
                )
                if explanation:
                    response_parts.append(f"📖 {explanation}")
                
                # Check if file was written
                written_file = state["code_context"].get("written_file")
                if written_file:
                    response_parts.append(f"📁 Code saved to file: `{written_file}`")

            # Add memory insights
            memory_patterns = state.get("memory_patterns", [])
            if memory_patterns:
                response_parts.append(
                    f"🧠 Used {len(memory_patterns)} patterns from memory"
                )

        # If no response content was generated, provide a helpful default
        if not response_parts:
            if "help" in state["user_query"].lower() or "what" in state["user_query"].lower():
                response_parts.append("I'd be happy to help! Could you be more specific about what you need assistance with?")
            else:
                response_parts.append("✅ Task completed")

        state["final_response"] = "\n\n".join(response_parts)
        state["workflow_step"] = "completed"

        return state

    def _detect_language(self, query: str) -> str:
        """Detect programming language from query."""
        language_keywords = {
            "python": ["python", "py", "django", "flask", "pandas"],
            "javascript": ["javascript", "js", "node", "react", "vue"],
            "typescript": ["typescript", "ts", "angular"],
            "java": ["java", "spring", "maven"],
            "go": ["go", "golang"],
            "rust": ["rust", "cargo"],
            "cpp": ["c++", "cpp", "cmake"],
        }

        query_lower = query.lower()
        for language, keywords in language_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return language

        return "python"  # Default

    async def process_query(self, user_query: str, thread_id: str = "default") -> str:
        """
        Process a user query through the complete workflow.
        """
        self._vprint(f"🚀 Starting workflow for: {user_query[:50]}...")

        # Ensure workflow is built
        if self.workflow is None:
            if self.verbose:
                print("⚠️ Workflow not compiled, rebuilding...")
            self._build_workflow()

        if self.workflow is None:
            return "❌ Failed to initialize workflow"

        # Initialize state
        initial_state = WorkflowState(
            messages=[HumanMessage(content=user_query)],
            user_query=user_query,
            current_agent="",
            tool_results={},
            code_context={},
            memory_patterns=[],
            workflow_step="initialized",
            error_state=None,
            final_response=None,
            agent_response=None,
        )
        try:
            # Run the workflow
            config = RunnableConfig(configurable={"thread_id": thread_id})
            final_state = await self.workflow.ainvoke(initial_state, config=config)

            self._vprint("✅ Workflow completed successfully")
            return final_state.get("final_response", "No response generated")

        except Exception as e:
            print(f"❌ Workflow error: {e}")
            return f"❌ Workflow failed: {str(e)}"

    def get_workflow_state(self, thread_id: str = "default") -> Optional[WorkflowState]:
        """Get the current workflow state for a thread."""
        try:
            config = RunnableConfig(configurable={"thread_id": thread_id})
            # This would need to be implemented with proper state retrieval
            return None
        except Exception:
            return None
