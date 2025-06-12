#!/usr/bin/env python3
"""
🧬 Symbiote - Your Code. Your Style. Your AI Shadow.

Main entry point for the Symbiote AI coding assistant.
This intelligent, adaptive coding assistant learns how you code and becomes
your snarky, ever-evolving sidekick.

Usage:
    python main.py --mode learn --path ./your/codebase
    python main.py --mode review --diff ./path/to/diff
    python main.py --mode generate --prompt "Create a function that..."
    python main.py --mode chat --path ./  # Interactive chat mode
    python main.py --mode sass --enable

Author: Vivek Sharma
License: MIT
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import readline  # For better input handling
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.code_parser import CodeParser, CodeAnalysisResult
from graph.style_graph import SymbioteStyleGraph
from agents.style_learner import StyleLearnerAgent
from utils.tools import SymbioteTools, create_symbiote_tools
from utils.tool_executor import AutonomousToolExecutor, create_autonomous_tool_executor
from utils.context_manager import SmartContextManager, create_smart_context_manager
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()


class SymbioteCore:
    """
    Core Symbiote application orchestrator.
    Manages the different modes and coordinates between components.
    """

    def __init__(self, debug: bool = False, sass_level: int = 5):
        self.debug = debug
        self.sass_level = sass_level

        # Initialize API keys and configurations
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("⚠️  Warning: GEMINI_API_KEY not found in environment variables.")
            print("   Some features may not work properly.")

        self.cache_enabled = (
            os.getenv("SYMBIOTE_CACHE_ENABLED", "true").lower() == "true"
        )

        # Initialize workspace path
        self.workspace_path = Path.cwd()

        # Initialize tools system
        self.tools = create_symbiote_tools(
            gemini_api_key=self.gemini_api_key,
            debug=debug,
            workspace_path=self.workspace_path
        )

        # Initialize autonomous tool executor
        self.tool_executor = create_autonomous_tool_executor(
            tools=self.tools,
            debug=debug,
            workspace_path=self.workspace_path
        )

        # Initialize smart context manager
        self.context_manager = create_smart_context_manager(
            workspace_path=self.workspace_path,
            debug=debug
        )

        # Initialize code parser with API key
        self.code_parser = CodeParser(gemini_api_key=self.gemini_api_key, debug=debug)

        # Initialize advanced components
        try:
            if self.gemini_api_key:
                self.style_graph = SymbioteStyleGraph(
                    gemini_api_key=self.gemini_api_key, debug=debug
                )
                self.style_agent = StyleLearnerAgent(
                    gemini_api_key=self.gemini_api_key, debug=debug
                )
                self.advanced_mode = True
                if self.debug:
                    print("🧠 Advanced AI components initialized")
            else:
                self.style_graph = None
                self.style_agent = None
                self.advanced_mode = False
                if self.debug:
                    print("⚠️  Advanced features disabled (no API key)")
        except Exception as e:
            print(f"⚠️  Advanced features unavailable: {e}")
            self.style_graph = None
            self.style_agent = None
            self.advanced_mode = False

        if self.debug:
            print(f"🐛 Debug mode enabled")
            print(f"😤 Sass level: {self.sass_level}/10")
            print(f"🗄️  Cache enabled: {self.cache_enabled}")
            print(f"🚀 Advanced mode: {self.advanced_mode}")
            print(f"🛠️  Tools available: {len(self.tools.tools)}")
            print(f"🤖 Tool executor ready: {self.tool_executor is not None}")
            print(f"🧠 Context manager ready: {self.context_manager is not None}")

    def learn_mode(
        self, codebase_path: str, output_file: Optional[str] = None
    ) -> CodeAnalysisResult:
        """
        Learn mode: Analyze a codebase and build/update the code style graph.

        Args:
            codebase_path: Path to the codebase to analyze
            output_file: Optional path to save the analysis results

        Returns:
            CodeAnalysisResult containing the analysis
        """
        print(f"🧠 Learning from codebase: {codebase_path}")

        if not Path(codebase_path).exists():
            raise FileNotFoundError(f"Codebase path not found: {codebase_path}")

        # Parse the codebase
        result = self.code_parser.parse_codebase(codebase_path)

        # Use advanced analysis if available
        if self.advanced_mode and self.style_graph:
            print("🚀 Running advanced style analysis...")
            try:
                # Run LangGraph analysis
                enhanced_result = self.style_graph.analyze_codebase(result)
                if enhanced_result:
                    print("✅ Advanced analysis completed")
                    # Add enhanced insights to display
                    if self.debug:
                        print(
                            f"📊 Enhanced insights: {len(enhanced_result.get('insights', []))}"
                        )
                        print(
                            f"💡 Recommendations: {len(enhanced_result.get('recommendations', []))}"
                        )
            except Exception as e:
                print(f"⚠️ Advanced analysis failed, using basic results: {e}")
                if self.debug:
                    import traceback

                    traceback.print_exc()

        # Display learning summary
        self._display_learning_summary(result)

        # Save results if requested
        if output_file:
            self.code_parser.export_to_json(result, output_file)
            print(f"💾 Analysis saved to: {output_file}")
        else:
            # Default output file
            default_output = f"symbiote_analysis_{Path(codebase_path).name}.json"
            self.code_parser.export_to_json(result, default_output)
            print(f"💾 Analysis saved to: {default_output}")

        return result

    def review_mode(self, diff_path: str):
        """
        Review mode: Analyze a diff/PR and provide feedback based on learned patterns.

        Args:
            diff_path: Path to the diff file to review
        """
        print(f"🔍 Reviewing diff: {diff_path}")

        if not Path(diff_path).exists():
            raise FileNotFoundError(f"Diff file not found: {diff_path}")

        # Read the diff file
        with open(diff_path, "r", encoding="utf-8") as f:
            diff_content = f.read()

        if self.advanced_mode and self.style_agent:
            print("🚀 Running AI-powered diff analysis...")
            try:
                # For now, use basic analysis - we'll enhance this later
                print("📊 Analyzing diff content...")

                # Basic diff statistics
                lines = diff_content.split("\n")
                added_lines = [
                    l for l in lines if l.startswith("+") and not l.startswith("+++")
                ]
                removed_lines = [
                    l for l in lines if l.startswith("-") and not l.startswith("---")
                ]

                print(f"\n📈 Diff Analysis:")
                print(f"   • Lines added: {len(added_lines)}")
                print(f"   • Lines removed: {len(removed_lines)}")
                print(f"   • Net change: {len(added_lines) - len(removed_lines)}")

                # Analyze code quality in added lines
                if added_lines:
                    print(f"\n🔍 Code Quality Check:")
                    code_content = "\n".join(
                        line[1:] for line in added_lines
                    )  # Remove + prefix

                    # Simple quality metrics
                    has_comments = any(
                        "//" in line or "#" in line for line in added_lines
                    )
                    avg_line_length = sum(len(line) for line in added_lines) / len(
                        added_lines
                    )

                    print(f"   • Contains comments: {'✅' if has_comments else '❌'}")
                    print(f"   • Average line length: {avg_line_length:.1f} chars")

                    if avg_line_length > 120:
                        print(
                            "   ⚠️  Some lines are quite long - consider breaking them up"
                        )

                # Add sass if enabled
                if self.sass_level > 6:
                    self._add_review_sass(len(added_lines), len(removed_lines))

            except Exception as e:
                print(f"⚠️ Diff analysis failed: {e}")
                if self.debug:
                    import traceback

                    traceback.print_exc()
        else:
            print("🚧 Basic diff analysis:")
            print("   - AI-powered review requires GEMINI_API_KEY")
            print("   - Showing basic diff statistics...")

            # Basic diff analysis
            lines = diff_content.split("\n")
            added_lines = len([l for l in lines if l.startswith("+")])
            removed_lines = len([l for l in lines if l.startswith("-")])

            print(f"   📊 Lines added: {added_lines}")
            print(f"   📊 Lines removed: {removed_lines}")
            print(f"   📊 Net change: {added_lines - removed_lines}")

            if self.sass_level > 5:
                if added_lines > removed_lines * 3:
                    print(
                        "   😤 Adding a lot more than removing... code bloat much? 🎈"
                    )
                elif removed_lines > added_lines * 2:
                    print("   ✂️ Nice cleanup! Marie Kondo would be proud. ✨")

    def generate_mode(self, prompt: str, style_file: Optional[str] = None):
        """
        Generate mode: Generate code based on learned style and user prompt.

        Args:
            prompt: The code generation prompt
            style_file: Optional path to specific style analysis file
        """
        print(f"✨ Generating code for: {prompt}")

        if not self.gemini_api_key:
            print("❌ Code generation requires GEMINI_API_KEY")
            print("   Please set your API key to use this feature.")
            return

        # Load style preferences
        style_context = ""
        analysis_file = style_file or "symbiote_analysis_.json"

        if Path(analysis_file).exists():
            try:
                with open(analysis_file, "r") as f:
                    analysis_data = json.load(f)

                # Extract style preferences
                style_prefs = analysis_data.get("style_preferences", {})
                naming_convs = analysis_data.get("naming_conventions", [])

                style_context = f"""
Based on your coding style:
- Language: {analysis_data.get('language', 'unknown')}
- Naming conventions: {', '.join(set(conv.get('convention_type', '') for conv in naming_convs[:3]))}
- Style preferences: {', '.join(f"{k}: {v}" for k, v in list(style_prefs.items())[:5])}
"""
                print("📚 Using learned style preferences from analysis")
            except Exception as e:
                print(f"⚠️ Could not load style preferences: {e}")
        else:
            print("📝 No style analysis found - using default style")

        # Generate code using Gemini
        try:
            print("🤖 Generating personalized code...")

            from google import genai

            client = genai.Client(api_key=self.gemini_api_key)

            generation_prompt = f"""
You are a coding assistant that generates code matching the user's personal style.

{style_context}

User Request: {prompt}

Generate code that:
1. Fulfills the user's request
2. Matches their established coding style and naming conventions
3. Includes appropriate comments and documentation
4. Follows best practices for the language

Provide only the code with brief explanations, no extra formatting.
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20", contents=generation_prompt
            )

            if response and response.text:
                print("\n🎨 Generated Code:")
                print("=" * 50)
                print(response.text)
                print("=" * 50)

                # Add sass if enabled
                if self.sass_level > 7:
                    print(f"\n😤 Symbiote's Take:")
                    print(
                        "   • Hope this code actually works better than your last attempt! 😏"
                    )
                    print("   • Don't forget to test it... unlike last time! 🧪")

                # Save to file if requested
                if style_file and style_file.endswith(".py"):
                    output_file = style_file.replace(".json", "_generated.py")
                    with open(output_file, "w") as f:
                        f.write(response.text)
                    print(f"\n💾 Code saved to: {output_file}")

            else:
                print("❌ No code generated - try rephrasing your request")

        except Exception as e:
            print(f"❌ Code generation failed: {e}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def sass_mode(self, enable: bool = True):
        """
        Toggle sass mode for more... entertaining feedback.

        Args:
            enable: Whether to enable or disable sass mode
        """
        if enable:
            print("😤 Sass mode ACTIVATED!")
            print("   Hope you're ready for some honest feedback about your code...")
            print("   (Don't say I didn't warn you! 🔥)")
        else:
            print("😐 Sass mode disabled.")
            print("   Back to boring, professional feedback. *sigh*")

    def chat_mode(self, workspace_path: str = "./"):
        """
        Interactive chat mode with AI assistant that can execute commands.

        Args:
            workspace_path: Path to the workspace directory
        """
        if not self.gemini_api_key:
            print("❌ Chat mode requires GEMINI_API_KEY")
            print("   Please set your API key to use this feature.")
            return

        print("🧬 Welcome to Symbiote Chat Mode!")
        print("   Your AI coding assistant is ready to help.")
        print("   Type 'help' for commands, 'exit' to quit.\n")

        # Initialize the workspace context
        workspace_path_obj = Path(workspace_path).resolve()
        if not workspace_path_obj.exists():
            print(f"❌ Workspace path not found: {workspace_path}")
            return

        print(f"📁 Workspace: {workspace_path}")

        # Analyze the workspace for context
        print("🔍 Analyzing workspace for context...")
        try:
            analysis_result = self.code_parser.parse_codebase(str(workspace_path))
            context = self._prepare_workspace_context(
                analysis_result, workspace_path_obj
            )
            print("✅ Workspace analysis complete!")
        except Exception as e:
            print(f"⚠️ Workspace analysis failed: {e}")
            context = {"error": str(e)}

        # Initialize chat session - use existing tools from SymbioteCore
        chat_session = SymbioteChatSession(
            workspace_path=workspace_path_obj,
            workspace_context=context,
            tools=self.tools,
            tool_executor=self.tool_executor,
            context_manager=self.context_manager,
            gemini_api_key=self.gemini_api_key,
            debug=self.debug,
            sass_level=self.sass_level,
        )

        # Start interactive loop
        try:
            chat_session.start_interactive_session()
        except KeyboardInterrupt:
            print("\n\n👋 Chat session ended. Goodbye!")
        except Exception as e:
            print(f"\n❌ Chat session error: {e}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def _prepare_workspace_context(
        self, analysis_result: CodeAnalysisResult, workspace_path: Path
    ) -> Dict[str, Any]:
        """Prepare workspace context for the chat session."""
        context = {
            "workspace_path": str(workspace_path),
            "language": analysis_result.language.value,
            "total_files": analysis_result.total_files,
            "total_lines": analysis_result.total_lines,
            "functions": [f.name for f in analysis_result.functions],
            "classes": [c.name for c in analysis_result.classes],
            "file_structure": [],
            "recent_files": [],
        }

        # Get file structure (top-level)
        try:
            for item in workspace_path.iterdir():
                if item.is_file() and item.suffix in [
                    ".py",
                    ".js",
                    ".ts",
                    ".java",
                    ".cpp",
                    ".c",
                    ".h",
                ]:
                    context["file_structure"].append(
                        {"name": item.name, "type": "file", "extension": item.suffix}
                    )
                elif item.is_dir() and not item.name.startswith("."):
                    context["file_structure"].append(
                        {"name": item.name, "type": "directory"}
                    )
        except Exception as e:
            context["file_structure_error"] = str(e)

        # Get recently modified files
        try:
            files = []
            for pattern in ["*.py", "*.js", "*.ts", "*.java"]:
                files.extend(workspace_path.glob(f"**/{pattern}"))

            # Sort by modification time and get recent ones
            recent_files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[
                :10
            ]
            context["recent_files"] = [
                str(f.relative_to(workspace_path)) for f in recent_files
            ]
        except Exception as e:
            context["recent_files_error"] = str(e)

        return context

    def _display_learning_summary(self, result: CodeAnalysisResult):
        """Display a summary of what was learned from the codebase."""
        print(f"\n📊 Learning Summary:")
        print(f"   Language: {result.language.value.title()}")
        print(f"   Files analyzed: {result.total_files}")
        print(f"   Lines of code: {result.total_lines:,}")
        print(f"   Functions discovered: {len(result.functions)}")
        print(f"   Classes discovered: {len(result.classes)}")

        if result.naming_conventions:
            print(f"\n🏷️  Naming Patterns Detected:")
            for conv in result.naming_conventions[:5]:  # Show top 5
                print(
                    f"   • {conv.context}: {conv.convention_type} "
                    f"({conv.frequency} occurrences)"
                )

        if result.style_preferences:
            print(f"\n🎨 Style Preferences:")
            for key, value in result.style_preferences.items():
                clean_key = key.replace("python_", "").replace("_", " ").title()
                print(f"   • {clean_key}: {value}")

        if result.api_usage:
            print(f"\n🔧 Most Used APIs:")
            sorted_apis = sorted(
                result.api_usage.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for api, count in sorted_apis:
                print(f"   • {api}: {count} times")

        # Display AI insights if available
        if hasattr(result, "ai_insights") and result.ai_insights:
            print(f"\n🤖 AI Insights ({len(result.ai_insights)}):")
            for insight in result.ai_insights[:3]:  # Show top 3
                print(f"   • {insight.insight_type}: {insight.description}")
                if insight.confidence > 0.8:
                    print(f"     ✅ High confidence ({insight.confidence:.2f})")

        # Add some sass if enabled
        if self.sass_level > 5:
            self._add_sass_commentary(result)

    def _add_sass_commentary(self, result: CodeAnalysisResult):
        """Add some sassy commentary based on the analysis."""
        print(f"\n😤 Symbiote's Honest Take:")

        # Commentary based on complexity
        if result.complexity_metrics.get("avg_function_complexity", 0) > 10:
            print("   • Your functions are more complex than a soap opera plot! 📺")

        # Commentary based on naming conventions
        snake_case_count = sum(
            1
            for conv in result.naming_conventions
            if conv.convention_type == "snake_case"
        )
        camel_case_count = sum(
            1
            for conv in result.naming_conventions
            if conv.convention_type == "camelCase"
        )

        if snake_case_count > camel_case_count:
            print("   • At least you're consistent with snake_case. *chef's kiss* 🐍")
        elif camel_case_count > 0:
            print(
                "   • Mixing camelCase with Python? Someone's feeling rebellious! 🏴‍☠️"
            )

        # Commentary based on code volume
        if result.total_lines > 50000:
            print("   • Wow, that's a LOT of code. Hope it actually works! 🤞")
        elif result.total_lines < 100:
            print("   • Cute little codebase! Don't worry, we all start somewhere. 👶")

    def _add_review_sass(self, added_lines: int, removed_lines: int):
        """Add some sassy commentary based on diff analysis."""
        print(f"\n😤 Symbiote's Review Take:")

        if added_lines > removed_lines * 3:
            print(
                "   • Adding way more than removing? That's some serious code bloat! 🎈"
            )
            print("   • Maybe consider if all that new code is really necessary? 🤔")
        elif removed_lines > added_lines * 2:
            print(
                "   • Nice cleanup! Marie Kondo would be proud of this decluttering! ✨"
            )
        elif added_lines == 0 and removed_lines == 0:
            print("   • No changes? Really? Did you just run git diff for fun? 😏")
        elif added_lines > 100:
            print("   • That's a big diff! Hope you tested it thoroughly... 🤞")
        elif added_lines < 5:
            print("   • Small and focused changes - I like your style! 👌")


class SymbioteChatSession:
    """
    Interactive chat session with AI assistant and tool execution capabilities.
    """

    def __init__(
        self,
        workspace_path: Path,
        workspace_context: Dict[str, Any],
        tools=None,
        tool_executor=None,
        context_manager=None,
        gemini_api_key: Optional[str] = None,
        debug: bool = False,
        sass_level: int = 5,
    ):
        self.workspace_path = workspace_path
        self.workspace_context = workspace_context
        self.tools = tools  # This will be a SymbioteTools instance
        self.tool_executor = tool_executor  # This will be an AutonomousToolExecutor instance
        self.context_manager = context_manager  # This will be a SmartContextManager instance
        self.gemini_api_key = gemini_api_key
        self.debug = debug
        self.sass_level = sass_level

        # Initialize conversation history
        self.conversation_history = []

        # Initialize Gemini client
        try:
            from google import genai

            self.gemini_client = genai.Client(api_key=gemini_api_key)
        except Exception as e:
            print(f"❌ Failed to initialize Gemini client: {e}")
            self.gemini_client = None

        # Initialize pending operations
        self.pending_operations = []

    def start_interactive_session(self):
        """Start the interactive chat session."""
        print("\n🚀 Chat session started!")
        print(
            "💬 You can ask me to analyze code, find bugs, make changes, or anything else!"
        )
        print("📋 Available commands: help, exit, clear, context, files")
        print("-" * 60)

        while True:
            try:
                # Get user input
                user_input = input("\n🧬 You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == "help":
                    self._show_help()
                    continue
                elif user_input.lower() == "clear":
                    self.conversation_history = []
                    print("🗑️ Conversation history cleared!")
                    continue
                elif user_input.lower() == "context":
                    self._show_context()
                    continue
                elif user_input.lower() == "files":
                    self._show_files()
                    continue

                # Process user query with AI
                print("\n🤖 Symbiote: ", end="", flush=True)
                self._process_user_query(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Chat session interrupted!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                if self.debug:
                    import traceback

                    traceback.print_exc()

    def _process_user_query(self, user_input: str):
        """Process user query with AI and tool execution."""
        if not self.gemini_client:
            print("❌ AI assistant not available (no API key)")
            return

        try:
            # Check for direct file operation commands first
            if self._handle_direct_file_operations(user_input):
                return

            # Add user input to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Create context-aware prompt
            system_prompt = self._create_system_prompt()
            conversation_context = self._format_conversation_history()

            full_prompt = f"""{system_prompt}

Current workspace: {self.workspace_path}
Recent conversation:
{conversation_context}

User query: {user_input}

Please analyze the query and provide a helpful response. If you need to examine or modify files, explain what you're doing step by step. 

IMPORTANT: When the user asks you to modify, fix, or add something to a file:
1. First read the current file content if needed
2. Generate the complete modified file content
3. Show a diff of the changes you would make
4. CLEARLY state that you're ready to apply the changes
5. The user can then confirm with "yes", "apply", "ok", or similar

Example response format when suggesting changes:
"I can add the requested function to your file. Here's what I would change:
[show diff]
I'm ready to apply these changes. Just say 'yes' or 'apply' to proceed."

Be specific about file operations and always show what you're going to change."""

            # Get AI response
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20", contents=full_prompt
            )

            if response and response.text:
                ai_response = response.text
                
                # Use autonomous tool executor to process the AI response
                if self.tool_executor:
                    enhanced_response, tool_results = self.tool_executor.process_llm_response(ai_response)
                    
                    if tool_results:
                        # Tools were executed - show the enhanced response with results
                        print(enhanced_response)
                        
                        # Track successful tool executions if context manager is available
                        if self.context_manager:
                            tools_used = [r.tool_call.tool_name for r in tool_results if r.success]
                            self.context_manager.track_interaction(
                                query=user_input,
                                intent=self.context_manager.infer_user_intent(user_input) if hasattr(self.context_manager, 'infer_user_intent') else 'analysis',
                                files_accessed=[],  # Could be enhanced to track actual files
                                tools_used=tools_used,
                                success=any(r.success for r in tool_results)
                            )
                    else:
                        # No tools executed - show original response
                        print(ai_response)
                else:
                    # Fallback - no tool executor available
                    print(ai_response)

                # Add AI response to conversation history
                self.conversation_history.append(
                    {"role": "assistant", "content": ai_response}
                )

                # Check if AI wants to execute tools or file operations
                if self._should_auto_execute_tools(user_input, ai_response):
                    self._handle_enhanced_tool_execution(user_input, ai_response)

                # Check for terminal command execution requests
                self._handle_terminal_commands(ai_response)

                # Add sass if enabled
                if self.sass_level > 6 and "error" in ai_response.lower():
                    print(
                        f"\n😤 (Well, that's not ideal... but hey, we'll figure it out! 🔧)"
                    )

            else:
                print("🤔 I didn't quite catch that. Could you rephrase?")

        except Exception as e:
            print(f"❌ Error: {e}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def _should_auto_execute_tools(self, user_query: str, ai_response: str) -> bool:
        """Determine if tools should be auto-executed based on user query and AI response patterns."""
        user_lower = user_query.lower()
        ai_lower = ai_response.lower()
        
        # Auto-execute for common analysis patterns in user query
        analysis_patterns = [
            "what does this code do",
            "how does this work", 
            "explain this code",
            "analyze this",
            "debug this",
            "fix this",
            "improve this",
            "optimize this",
            "show me the code",
            "check the code",
            "look at the file",
            "examine the file",
            "read the file",
        ]
        
        # Auto-execute if AI mentions it will examine/check/read files
        ai_action_patterns = [
            "let me check",
            "let me read", 
            "let me examine",
            "i'll read",
            "i'll check",
            "i'll examine",
            "i need to read",
            "i need to check", 
            "i should read",
            "i should check",
            "let me look at",
            "i'll look at",
            "let me see",
            "i'll investigate",
            "let me analyze",
            "i should examine",
            "i'll take a look",
        ]
        
        return (
            any(pattern in user_lower for pattern in analysis_patterns) or
            any(pattern in ai_lower for pattern in ai_action_patterns)
        )

    def _handle_direct_file_operations(self, user_input: str) -> bool:
        """Handle direct file operation commands from user."""
        user_lower = user_input.lower().strip()

        # Check for confirmation responses
        confirmation_words = [
            "yes",
            "apply",
            "confirm",
            "do it",
            "go ahead",
            "ok",
            "okay",
            "sure",
            "proceed",
            "continue",
        ]
        if any(word in user_lower for word in confirmation_words):
            if self.debug:
                print(f"🐛 DEBUG - Detected confirmation: '{user_input}'")
                self._debug_conversation_context()

            # Look for recent AI responses that suggested file changes
            if len(self.conversation_history) >= 2:
                last_ai_response = None
                for msg in reversed(self.conversation_history):
                    if msg["role"] == "assistant":
                        last_ai_response = msg["content"]
                        break

                if last_ai_response and any(
                    keyword in last_ai_response.lower()
                    for keyword in [
                        "add",
                        "function",
                        "modify",
                        "change",
                        "update",
                        "create",
                        "insert",
                        "def ",
                        "class ",
                        "import ",
                        "will add",
                        "let me add",
                        "i'll add",
                        "here's the",
                        "here is the",
                        "diff",
                        "changes",
                    ]
                ):
                    if self.debug:
                        print(
                            f"🐛 DEBUG - AI suggested changes detected in: '{last_ai_response[:100]}...'"
                        )
                    # Try to detect what file they want to modify
                    target_file = self._extract_target_file_from_context()
                    if target_file:
                        if self.debug:
                            print(f"🐛 DEBUG - Target file detected: {target_file}")
                        return self._apply_ai_suggested_changes(target_file, user_input)
                    else:
                        if self.debug:
                            print("🐛 DEBUG - No target file detected from context")

        # Check for direct file modification commands
        operations = (
            self.tools.smart_command_parser(user_input, "", self.workspace_context)
            if self.tools
            else []
        )
        for op in operations:
            if op["confidence"] > 0.7:
                if op["operation"] == "read":
                    result = (
                        self.tools.read_file_with_lines(op["file_path"])
                        if self.tools
                        else f"❌ File operations not available"
                    )
                    print(f"\n📄 {op['file_path']}:\n{result}")
                    return True
                elif op["operation"] == "modify":
                    return (
                        self.tools.interactive_file_modification(
                            op["file_path"], user_input, auto_confirm=False
                        )
                        if self.tools
                        else False
                    )
                elif op["operation"] == "create":
                    return self._handle_file_creation(op["file_path"], user_input)

        return False

    def _extract_target_file_from_context(self) -> Optional[str]:
        """Extract target file from recent conversation context."""
        # Look through recent conversation for file references
        for msg in reversed(self.conversation_history[-6:]):  # Check last 6 messages
            content = msg["content"].lower()
            for file_ext in [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"]:
                if file_ext in content:
                    words = content.split()
                    for word in words:
                        if file_ext in word and not word.startswith("http"):
                            clean_word = word.strip('.,!?()[]{}":;')
                            if self._file_exists(clean_word):
                                return clean_word

        # Check workspace files that might be implied
        workspace_files = self.workspace_context.get("recent_files", [])
        for file_path in workspace_files:
            if any(
                file_path.lower() in msg["content"].lower()
                for msg in self.conversation_history[-4:]
            ):
                return file_path

        return None

    def _apply_ai_suggested_changes(
        self, file_path: str, user_confirmation: str
    ) -> bool:
        """Apply changes suggested by AI to the specified file."""
        try:
            # Get the last AI response that suggested changes
            last_ai_response = None
            last_user_query = None

            for i in range(len(self.conversation_history) - 1, -1, -1):
                msg = self.conversation_history[i]
                if msg["role"] == "assistant" and not last_ai_response:
                    last_ai_response = msg["content"]
                elif msg["role"] == "user" and last_ai_response and not last_user_query:
                    last_user_query = msg["content"]
                    break

            if not last_ai_response or not last_user_query:
                print("❌ Could not find the suggested changes to apply.")
                return False

            print(f"\n🔧 Applying suggested changes to {file_path}...")

            # Use the interactive file modification with AI assistance, auto-confirming since user already said yes
            return (
                self.tools.interactive_file_modification(
                    file_path, last_user_query, auto_confirm=True
                )
                if self.tools
                else False
            )

        except Exception as e:
            print(f"❌ Error applying changes: {e}")
            return False

    def _handle_file_creation(self, file_path: str, user_request: str) -> bool:
        """Handle creation of a new file."""
        try:
            full_path = self.workspace_path / file_path

            if full_path.exists():
                print(f"❌ File {file_path} already exists!")
                return False

            # Use AI to generate file content
            if self.gemini_client:
                creation_prompt = f"""
Create a new file based on this request: {user_request}

File path: {file_path}

Generate appropriate content for this file based on its name and the user's request.
Return ONLY the file content without any explanations or markdown formatting.
"""

                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash-preview-05-20", contents=creation_prompt
                )

                if response and response.text:
                    content = response.text.strip()

                    print(f"\n📝 Generated content for {file_path}:")
                    print("=" * 50)
                    print(content[:500] + ("..." if len(content) > 500 else ""))
                    print("=" * 50)

                    confirmation = (
                        input(f"\n🤔 Create {file_path} with this content? (yes/no): ")
                        .strip()
                        .lower()
                    )

                    if confirmation in ["yes", "y"]:
                        return (
                            self.tools.create_file(file_path, content)
                            if self.tools
                            else False
                        )
                    else:
                        print("❌ File creation cancelled")
                        return False

            print("❌ AI file generation requires Gemini API key")
            return False

        except Exception as e:
            print(f"❌ Error creating file: {e}")
            return False

    def _file_exists(self, file_path: str) -> bool:
        """Check if a file exists in the workspace."""
        try:
            full_path = self.workspace_path / file_path
            return full_path.exists()
        except Exception:
            return False

    def _debug_conversation_context(self):
        """Debug method to show recent conversation context."""
        if self.debug:
            print("\n🐛 DEBUG - Recent conversation:")
            for i, msg in enumerate(self.conversation_history[-4:]):
                role = msg["role"]
                content = (
                    msg["content"][:200] + "..."
                    if len(msg["content"]) > 200
                    else msg["content"]
                )
                print(f"  {i+1}. [{role.upper()}]: {content}")
            print()

    def _create_system_prompt(self) -> str:
        """Create a context-aware system prompt."""
        files_context = ""
        if self.workspace_context.get("file_structure"):
            files = [f["name"] for f in self.workspace_context["file_structure"][:10]]
            files_context = f"Key files: {', '.join(files)}"

        return f"""You are Symbiote, an intelligent AI coding assistant with deep knowledge of the user's codebase.

Workspace Context:
- Language: {self.workspace_context.get('language', 'Unknown')}
- Total files: {self.workspace_context.get('total_files', 0)}
- Total lines: {self.workspace_context.get('total_lines', 0)}
- Functions: {len(self.workspace_context.get('functions', []))}
- Classes: {len(self.workspace_context.get('classes', []))}
{files_context}

You can help with:
1. Code analysis and debugging
2. Finding and fixing bugs
3. Code review and suggestions
4. File exploration and navigation
5. Explaining code functionality
6. Generating code solutions
7. **MODIFYING FILES** - You can read, edit, and create files!
8. **TERMINAL COMMANDS** - You can execute shell commands!

File Operations You Can Perform:
- Read files: "read file main.py" or "show me utils/helper.py"
- Modify files: "modify main.py" or "update the login function in auth.py"
- Create files: "create test_utils.py" or "make a new config file"

Terminal Commands You Can Execute:
When users ask you to run terminal/shell commands, use this format:
EXECUTE_COMMAND: [command]

Examples:
- EXECUTE_COMMAND: ls -la
- EXECUTE_COMMAND: git status
- EXECUTE_COMMAND: npm install
- EXECUTE_COMMAND: python test.py
- EXECUTE_COMMAND: rm unwanted_file.txt

IMPORTANT: Always use "EXECUTE_COMMAND: " prefix when you want to run a shell command.
Do NOT simulate command output - actually execute the command when users request it.

When modifying files:
- I'll show you the current content first
- I'll generate a diff to show exactly what changes
- I'll ask for your confirmation before applying changes
- I'll create backups automatically

When analyzing issues:
- Be specific about what you're checking
- Show relevant code snippets when helpful
- Explain problems clearly
- Propose concrete solutions with code changes
- Use file operations to implement fixes

Response Format:
- Explain what you're going to do
- Use natural language to describe file operations
- Show code snippets and diffs when relevant
- Ask for confirmation before making changes

Respond in a helpful, conversational manner. Be direct but friendly."""

    def _format_conversation_history(self) -> str:
        """Format recent conversation history for context."""
        if not self.conversation_history:
            return "No previous conversation."

        # Get last 3 exchanges
        recent = self.conversation_history[-6:]  # 3 user + 3 assistant messages
        formatted = []

        for msg in recent:
            role = "You" if msg["role"] == "user" else "Symbiote"
            content = (
                msg["content"][:200] + "..."
                if len(msg["content"]) > 200
                else msg["content"]
            )
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _handle_tool_execution(self, user_query: str, ai_response: str):
        """Handle tool execution based on AI response."""
        if not self.tools:
            return

        # Simple tool execution triggers
        if "let me check" in ai_response.lower() or "examine" in ai_response.lower():
            print("\n🔧 Executing analysis tools...")

            # Try to identify what file/function the user is asking about
            potential_targets = self._extract_code_targets(user_query)

            for target in potential_targets:
                if target in self.workspace_context.get("functions", []):
                    print(f"🔍 Analyzing function: {target}")
                    # Here you could call specific analysis tools
                elif any(
                    target in f for f in self.workspace_context.get("recent_files", [])
                ):
                    print(f"📄 Examining file: {target}")
                    # Here you could call file analysis tools

    def _extract_code_targets(self, query: str) -> List[str]:
        """Extract potential function names, file names from user query."""
        # Simple extraction - look for common patterns
        words = query.split()
        targets = []

        for word in words:
            # Remove common punctuation
            clean_word = word.strip('.,!?()[]{}":;')

            # Look for function-like patterns
            if clean_word.endswith("()") or "function" in query.lower():
                targets.append(clean_word.replace("()", ""))
            # Look for file patterns
            elif "." in clean_word and any(
                ext in clean_word for ext in [".py", ".js", ".ts", ".java"]
            ):
                targets.append(clean_word)
            # Look for CamelCase or snake_case identifiers
            elif len(clean_word) > 3 and (
                "_" in clean_word or any(c.isupper() for c in clean_word[1:])
            ):
                targets.append(clean_word)

        return targets[:3]  # Limit to first 3 targets

    def _show_help(self):
        """Show help information."""
        print(
            """
🆘 Symbiote Chat Help:

Commands:
  help     - Show this help message
  exit     - Exit chat mode
  clear    - Clear conversation history
  context  - Show workspace context
  files    - Show recent files

What you can ask:
  📖 Code Analysis:
    • "Check the generateImage function"
    • "Why is my API call failing?"
    • "Find bugs in auth.py"
    • "How does the login system work?"
  
  📝 File Operations:
    • "Show me main.py"
    • "Read the utils/helper.py file"
    • "Modify the login function in auth.py"
    • "Create a new test file"
    • "Fix the bug in line 25 of config.py"
  
  🔧 Code Improvements:
    • "Review my latest changes"
    • "Optimize this function"
    • "Add error handling to the API call"
    • "Refactor the database connection code"

I can analyze code, debug issues, explain functionality, suggest improvements, 
and **directly modify your files** with your permission!
"""
        )

    def _show_context(self):
        """Show current workspace context."""
        print(f"\n📋 Workspace Context:")
        print(f"   Path: {self.workspace_path}")
        print(f"   Language: {self.workspace_context.get('language', 'Unknown')}")
        print(f"   Files: {self.workspace_context.get('total_files', 0)}")
        print(f"   Lines: {self.workspace_context.get('total_lines', 0)}")

        if self.workspace_context.get("functions"):
            funcs = self.workspace_context["functions"][:5]
            print(
                f"   Functions: {', '.join(funcs)}{'...' if len(self.workspace_context['functions']) > 5 else ''}"
            )

        if self.workspace_context.get("classes"):
            classes = self.workspace_context["classes"][:3]
            print(
                f"   Classes: {', '.join(classes)}{'...' if len(self.workspace_context['classes']) > 3 else ''}"
            )

    def _show_files(self):
        """Show recent files in workspace."""
        print(f"\n📁 Recent Files:")
        recent_files = self.workspace_context.get("recent_files", [])
        if recent_files:
            for i, file_path in enumerate(recent_files[:10], 1):
                print(f"   {i:2}. {file_path}")
        else:
            print("   No recent files found.")

        print(f"\n📂 File Structure:")
        file_structure = self.workspace_context.get("file_structure", [])
        if file_structure:
            for item in file_structure[:15]:
                icon = "📄" if item["type"] == "file" else "📁"
                print(f"   {icon} {item['name']}")
            if len(file_structure) > 15:
                print(f"   ... and {len(file_structure) - 15} more items")
        else:
            print("   No file structure available.")

    def _handle_enhanced_tool_execution(self, user_query: str, ai_response: str):
        """Enhanced tool execution with file operations."""
        if not self.tools:
            # Try to execute file operations even without tools
            commands = (
                self.tools.parse_ai_file_commands(ai_response) if self.tools else []
            )
            for cmd in commands:
                print(f"\n🔧 Executing: {cmd['operation']} {cmd['file_path']}")
                result = (
                    self.tools.execute_file_operation(
                        cmd["operation"], cmd["file_path"]
                    )
                    if self.tools
                    else "❌ File operations not available"
                )
                print(result)
            return

        # Enhanced tool execution logic
        if any(
            keyword in ai_response.lower()
            for keyword in [
                "let me check",
                "examine",
                "analyze",
                "read file",
                "show file",
            ]
        ):
            print("\n🔧 Executing analysis tools...")

            # Parse potential file targets
            potential_targets = self._extract_code_targets(
                user_query + " " + ai_response
            )

            for target in potential_targets:
                if any(ext in target for ext in [".py", ".js", ".ts", ".java"]):
                    # This is likely a file
                    print(f"📄 Reading file: {target}")
                    result = (
                        self.tools.execute_file_operation("read", target)
                        if self.tools
                        else "❌ File operations not available"
                    )
                    print(result)
                elif target in self.workspace_context.get("functions", []):
                    print(f"🔍 Analyzing function: {target}")
                    # Could use tools here for function analysis
                elif any(
                    target in str(f)
                    for f in self.workspace_context.get("recent_files", [])
                ):
                    print(f"📄 Examining file: {target}")
                    result = (
                        self.tools.execute_file_operation("read", target)
                        if self.tools
                        else "❌ File operations not available"
                    )
                    print(result)

    def _handle_terminal_commands(self, ai_response: str):
        """Handle terminal command execution requests from AI responses."""
        if self.tools:
            self.tools.handle_terminal_commands(ai_response)


def main():
    """Main entry point for Symbiote."""
    parser = argparse.ArgumentParser(
        description="🧬 Symbiote - Your Code. Your Style. Your AI Shadow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode learn --path ./my_project
  %(prog)s --mode review --diff ./changes.patch
  %(prog)s --mode generate --prompt "Create a REST API endpoint"
  %(prog)s --mode chat --path ./my_project
  %(prog)s --mode sass --enable
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["learn", "review", "generate", "sass", "chat"],
        required=True,
        help="Operation mode",
    )

    parser.add_argument("--path", help="Path to codebase (for learn mode)")

    parser.add_argument("--diff", help="Path to diff file (for review mode)")

    parser.add_argument("--prompt", help="Code generation prompt (for generate mode)")

    parser.add_argument("--output", help="Output file path")

    parser.add_argument("--enable", action="store_true", help="Enable sass mode")

    parser.add_argument("--disable", action="store_true", help="Disable sass mode")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument(
        "--sass-level",
        type=int,
        default=5,
        choices=range(0, 11),
        help="Sass level (0=vanilla, 10=tsundere meltdown)",
    )

    args = parser.parse_args()

    # Initialize Symbiote
    symbiote = SymbioteCore(debug=args.debug, sass_level=args.sass_level)

    print("🧬 Symbiote activated!")
    print("   Your Code. Your Style. Your AI Shadow.\n")

    try:
        if args.mode == "learn":
            if not args.path:
                parser.error("--path is required for learn mode")
            symbiote.learn_mode(args.path, args.output)

        elif args.mode == "review":
            if not args.diff:
                parser.error("--diff is required for review mode")
            symbiote.review_mode(args.diff)

        elif args.mode == "generate":
            if not args.prompt:
                parser.error("--prompt is required for generate mode")
            symbiote.generate_mode(args.prompt, args.output)

        elif args.mode == "sass":
            if args.enable:
                symbiote.sass_mode(True)
            elif args.disable:
                symbiote.sass_mode(False)
            else:
                parser.error("--enable or --disable is required for sass mode")

        elif args.mode == "chat":
            if not args.path:
                parser.error("--path is required for chat mode")
            symbiote.chat_mode(args.path)

    except KeyboardInterrupt:
        print("\n\n👋 Symbiote interrupted. See you later!")
        sys.exit(0)

    except Exception as e:
        if args.debug:
            import traceback

            traceback.print_exc()
        else:
            print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
