#!/usr/bin/env python3
"""
Symbiote - Advanced AI coding assistant

"""

import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.orchestrator import SymbioteOrchestrator
from src.agents.base_agent import AgentConfig, SassLevel

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


async def main():
    """Main CLI entry point."""
    print("🧬 Welcome to Symbiote - Your AI Coding Assistant")
    print("Type '/help' for commands or just tell me what you want to do!")
    print("Type 'exit' or 'quit' to leave.\n")

    # Check API keys
    groq_key = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if not groq_key:
        print("⚠️  GROQ_API_KEY not found. Set it in .env file for LLaMA functionality.")
    if not google_key:
        print(
            "⚠️  GOOGLE_API_KEY not found. Set it in .env file for Gemini code generation."
        )

    if not groq_key and not google_key:
        print("❌ No API keys found. Please set up your .env file with API keys.")
        print("See .env.example for the required format.\n")

    # Initialize orchestrator (quiet mode after initialization)
    orchestrator = SymbioteOrchestrator(use_langgraph=True, verbose=True)

    # Initialize with default configs
    success = await orchestrator.initialize()
    if not success:
        print("❌ Failed to initialize Symbiote. Exiting.")
        return

    # Switch to quiet mode after initialization
    orchestrator.set_verbose(False)
    print("✅ Symbiote is ready! (Use /help for commands)\n")

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("🧬 You: ").strip()

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye! Thanks for using Symbiote!")
                break

            if not user_input:
                continue

            # Process input
            print("🤖 Symbiote: ", end="", flush=True)
            response = await orchestrator.process_user_input(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye! Thanks for using Symbiote!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again or type 'exit' to quit.\n")

    # Cleanup
    await orchestrator.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
