# 🧬 Symbiote

> _Your Code. Your Style. Your AI Shadow._

**Symbiote** is an intelligent, adaptive coding assistant that _learns how you code_—and then becomes your snarky, ever-evolving sidekick. It observes, remembers, critiques (sometimes too honestly), and writes code in your _exact_ style, almost like your second brain... if your second brain had better linting skills.

Powered by **LangGraph**, **LangChain**, and **Gemini API**, Symbiote builds a personal graph of your habits, quirks, and coding rituals to deliver personalized suggestions, witty corrections, and hauntingly familiar code snippets.

---

## ✨ Features

-   🧠 **Advanced Code Style Analysis**
    Uses LangGraph + LangChain to build intelligent graphs of your coding patterns, API preferences, and architectural decisions.

-   🤖 **AI-Powered Interactive Chat**
    Chat with your codebase! Ask questions, execute commands, modify files, and get intelligent suggestions.

-   🛠 **Multi-Language Code Parsing**
    Supports Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more with deep syntax analysis.

-   📊 **Persistent Pattern Memory**
    FAISS vector store remembers your coding patterns across sessions and projects.

-   💬 **Sass-Enabled Feedback Mode™**
    Adjustable snark levels (0-10) for entertaining code reviews and commentary. 😤

-   📚 **Personalized Code Generation**
    Gemini-powered code generation that matches your exact style, naming conventions, and preferences.

-   🔍 **Intelligent Diff Analysis**
    AI-powered PR and diff reviews based on your historical coding behavior.

-   🧪 **Real-Time Tool Execution**
    Execute shell commands, read/write files, and perform git operations through the chat interface.

---

## 🛠 Tech Stack

-   **LangGraph** — Dynamic multi-agent workflows for code analysis
-   **LangChain** — Agent orchestration, memory management & tool integration
-   **Gemini API** — Advanced code understanding, generation & intelligent insights
-   **FAISS** — Vector storage for persistent pattern memory
-   **Google GenAI** — Embeddings and natural language processing
-   **Python AST** — Deep code structure analysis and pattern extraction

---

## 📁 Repo Structure

```bash
symbiote/
├── agents/
│   └── style_learner.py       # LangChain agent that learns coding patterns & preferences
├── graph/
│   └── style_graph.py         # LangGraph-based style analysis graph with AI agents
├── utils/
│   ├── code_parser.py         # Multi-language code analysis & pattern extraction
│   └── tools.py               # Comprehensive tool suite for AI agents
├── symbiote_memory/
│   └── pattern_vectors/       # FAISS vector store for learned patterns
├── DOC/                       # Documentation and setup guides
├── main.py                    # Main orchestration & CLI entry point
├── requirements.txt           # Dependencies (LangChain, LangGraph, Gemini)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+ (tested with Python 3.13)
-   Google Gemini API key
-   Git (for repository analysis)

### Installation

1. **Clone this Symbiote**

```bash
git clone https://github.com/ifsvivek/Symbiote.git
cd Symbiote
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure Gemini API**

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SYMBIOTE_CACHE_ENABLED=true
```

Or set environment variables directly:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

4. **Run your symbiotic buddy**

```bash
# Learn from your codebase
python main.py --mode learn --path ./your/codebase

# Start interactive chat
python main.py --mode chat --path ./your/project

# Enable debug mode for detailed output
python main.py --mode learn --path ./ --debug
```

---

## 🤖 Modes

| Mode       | Description                                     | Status         |
| ---------- | ----------------------------------------------- | -------------- |
| `learn`    | Builds/updates your LangGraph style profile     | ✅ Implemented |
| `review`   | Reviews diffs/PRs with AI-powered analysis      | ✅ Implemented |
| `generate` | Generates code matching your personal style     | ✅ Implemented |
| `chat`     | Interactive AI assistant with file operations   | ✅ Implemented |
| `sass`     | Enables/disables snarky commentary (0-10 scale) | ✅ Implemented |

### Usage Examples

```bash
# Learn your coding style from a codebase
python main.py --mode learn --path ./my-project

# Review a git diff with AI analysis
python main.py --mode review --diff changes.diff

# Generate code in your personal style
python main.py --mode generate --prompt "Create a REST API handler"

# Start interactive chat mode
python main.py --mode chat --path ./

# Enable maximum sass level
python main.py --mode sass --enable --sass-level 10
```

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

## ⚙️ Configuration & Troubleshooting

### Environment Variables

```bash
GEMINI_API_KEY=your_api_key          # Required: Your Google Gemini API key
SYMBIOTE_CACHE_ENABLED=true          # Optional: Enable/disable caching (default: true)
```

### Common Issues

**🔑 API Key Issues**

```bash
# Verify your API key is set
echo $GEMINI_API_KEY

# Test API connectivity
python -c "from google import genai; genai.configure(api_key='your_key'); print('✅ API key works!')"
```

**📦 Dependencies**

```bash
# If you encounter import errors, try reinstalling dependencies
pip install -r requirements.txt --force-reinstall

# For development, install with verbose output
pip install -r requirements.txt -v
```

**🧠 Memory Issues**

```bash
# Clear pattern memory if needed
rm -rf symbiote_memory/pattern_vectors/

# Disable caching for troubleshooting
export SYMBIOTE_CACHE_ENABLED=false
```

### Performance Tips

-   Use `--debug` flag for detailed logging
-   Large codebases (>10k files) may take several minutes to analyze
-   Pattern memory improves accuracy over time - don't clear it unless necessary
-   Chat mode works best with focused, project-specific conversations

---

## 🤝 Contributing

Symbiote is actively developed and welcomes contributions! Here's how you can help:

### Development Setup

```bash
git clone https://github.com/ifsvivek/Symbiote.git
cd Symbiote
pip install -r requirements.txt
export GEMINI_API_KEY="your_api_key"
python main.py --mode learn --path ./ --debug
```

### Areas for Contribution

-   **New Language Support**: Add parsers for additional programming languages
-   **Agent Improvements**: Enhance the LangGraph workflows and LangChain agents
-   **UI/UX**: Build web interfaces or IDE integrations
-   **Documentation**: Improve docs, examples, and tutorials
-   **Testing**: Add comprehensive test coverage
-   **Performance**: Optimize analysis speed and memory usage

### Code Style

Symbiote follows the same coding patterns it learns from you! But generally:

-   Use meaningful variable names and comments
-   Follow PEP 8 for Python code
-   Add docstrings to functions and classes
-   Include type hints where possible

---

## 👨‍💻 Author

Made by [Vivek Sharma](https://ifsvivek.in) — the type of dev whose AI assistant may become _too powerful_.

---

## 📝 License

MIT — Use it, break it, improve it, sass it. Just don't sell it as _your_ symbiote. 😤
