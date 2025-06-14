# 🧬 Symbiote

> _Your Code. Your Style. Your AI Shadow._

**Symbiote** is an intelligent, adaptive coding assistant that _learns how you code_—and then becomes your snarky, ever-evolving sidekick. It observes, remembers, critiques (sometimes too honestly), and writes code in your _exact_ style, almost like your second brain... if your second brain had better linting skills.

Powered by **LangGraph**, **LangChain**, **LLaMA 3.3**, and **Gemini API**, Symbiote builds a personal graph of your habits, quirks, and coding rituals to deliver personalized suggestions, witty corrections, and hauntingly familiar code snippets.

---

## ✨ Features

-   🧠 **Advanced Code Style Analysis**
    Uses LangGraph + LangChain to build intelligent graphs of your coding patterns, API preferences, and architectural decisions.
-   🤖 **AI-Powered Interactive Chat**
    Chat with your codebase! Ask questions, execute commands, modify files, and get intelligent suggestions.
-   🛠️ **Intelligent Tool System**
    Autonomous tools with descriptive names (FileExplorer, GitManager, CodeAnalyzer) that chain operations and make intelligent decisions.
-   🧩 **Multi-Language Code Parsing**
    Supports Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more with deep syntax analysis.
-   📊 **Persistent Pattern Memory**
    FAISS vector store remembers your coding patterns across sessions and projects.
-   💬 **Sass-Enabled Feedback Mode™**
    Adjustable snark levels (0-10) for entertaining code reviews and commentary. 😤
-   📚 **Personalized Code Generation**
    Gemini-powered code generation that matches your exact style, naming conventions, and preferences.
-   🔍 **Intelligent Diff Analysis**
    AI-powered PR and diff reviews based on your historical coding behavior.
-   🦾 **Real-Time Tool Execution**
    Execute shell commands, read/write files, and perform git operations through the chat interface.

---

## 🏗️ Multi-Agent Architecture

Symbiote uses a multi-agent system for maximum flexibility and intelligence:

-   **LLaMA 3.3 Agent (Orchestrator)**

    -   Handles user input, intent detection, and tool selection.
    -   Calls tools (file ops, git, search, etc.) as needed.
    -   Passes code generation requests to Gemini.

-   **Gemini Agent (Code Generator)**

    -   Receives context, requirements, and data from LLaMA.
    -   Generates code snippets, explanations, or refactors as needed.
    -   Returns code/results to LLaMA.

-   **Tool System**

    -   Modular, extensible tool interface (e.g., for file, git, shell, search, etc.).
    -   Tools can be called by LLaMA as functions or via a tool registry.

-   **Communication**
    -   Agents communicate via structured messages (JSON or similar).
    -   LLaMA manages the workflow and delegates code generation to Gemini.

---

## 🚦 Planned Workflow

1. **User** asks for a coding task.
2. **LLaMA** analyzes, calls tools, gathers context.
3. **LLaMA** sends code generation request to **Gemini**.
4. **Gemini** generates code and returns it.
5. **LLaMA** finalizes the task and responds to the user.

---

## 🛠️ Tool Handling System

-   Modular tool registry for easy extension.
-   Tools for file operations, git, shell, search, and more.
-   LLaMA can chain tool calls and handle tool outputs.
-   Error handling, logging, and extensibility for new tools.

---

## 🎯 Advanced Features

### Pattern Learning & Memory

-   **Persistent Vector Storage**: FAISS-based pattern memory that persists across sessions
-   **Multi-Language Support**: Analyzes Python, JavaScript, TypeScript, Java, C++, Go, Rust
-   **Style Preference Detection**: Learns naming conventions, indentation, API usage patterns
-   **Architectural Pattern Recognition**: Identifies design patterns and code organization preferences

### Interactive Chat Mode

-   **File Operations**: Read, modify, and create files through natural language
-   **Terminal Integration**: Execute shell commands directly from chat
-   **Context-Aware Assistance**: Understands your entire codebase for better suggestions
-   **Git Integration**: Analyze diffs, commits, and repository history

### AI-Powered Code Analysis

-   **LangGraph Workflows**: Multi-agent analysis pipelines for comprehensive insights
-   **Complexity Metrics**: Function complexity, code quality assessment
-   **Documentation Analysis**: Comment patterns and documentation preferences
-   **Import/Dependency Mapping**: Understanding of your library and framework usage

### Sass Mode™

Adjustable personality levels (0-10 scale):

-   **0-3**: Professional and helpful
-   **4-6**: Friendly with occasional humor
-   **7-8**: Sarcastic but constructive
-   **9-10**: Full tsundere mode (use at your own risk! 😤)

> **Personality is fully adjustable!**
> Use the `/sass <level>` command, config file, or CLI flag to set your preferred sass level. All agents (LLaMA and Gemini) will adapt their tone and style accordingly for every interaction.

---

## 🧪 Future Features

### Planned Enhancements

-   **VSCode Extension**: Integrated IDE experience with real-time suggestions
-   **GitHub App**: Automated PR commenting and code review workflows
-   **Visual Style Dashboard**: Interactive graph visualization of your coding patterns
-   **Team Mode**: Multi-developer style blending and team preference learning
-   **Custom Model Support**: Integration with local LLMs and other AI providers
-   **Code Refactoring Agent**: Automated code improvement suggestions
-   **Documentation Generator**: Auto-generate docs matching your writing style

### Experimental Ideas

-   **"Clone Me" Mode**: Train a model to write code exactly like you
-   **Code Archaeology**: Deep analysis of coding evolution over time
-   **Style Conflict Resolution**: Help teams reconcile different coding preferences
-   **Smart Code Templates**: Generate boilerplate matching your exact patterns

---

## 🧰 Tech Stack

-   **LangGraph** — Dynamic multi-agent workflows for code analysis
-   **LangChain** — Agent orchestration, memory management & tool integration
-   **LLaMA 3.3** — Orchestration, tool use, and workflow management
-   **Gemini API** — Advanced code understanding, generation & intelligent insights
-   **FAISS** — Vector storage for persistent pattern memory
-   **Google GenAI** — Embeddings and natural language processing
-   **Python AST** — Deep code structure analysis and pattern extraction

---

## 📦 Requirements

See `requirements.txt` for all dependencies. Core stack includes:
- langgraph, langchain, langchain-community, langchain-core, langchain-google-genai
- google-genai, faiss-cpu, httpx
- pydantic, typing-extensions, python-dotenv, pathlib, dataclasses-json

---

## 🚀 Quick Start

1. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd Symbiote
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # GROQ_API_KEY=your_groq_api_key_here
   # GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **Run Symbiote:**
   ```bash
   python main.py
   ```

4. **Start coding with your AI assistant!**
   ```
   🧬 You: create a hello world function
   🤖 Symbiote: *generates clean Python code*
   
   🧬 You: /sass 8
   🤖 Symbiote: Fine, I'll be more sarcastic... 😏
   ```

---

## 👨‍💻 Author

Made by [Vivek Sharma](https://ifsvivek.in) — the type of dev whose AI assistant may become _too powerful_.

---

## 📝 License

MIT — Use it, break it, improve it, sass it. Just don't sell it as _your_ symbiote. 😤
