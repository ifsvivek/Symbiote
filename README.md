# 🧬 Symbiote

> *Your Code. Your Style. Your AI Shadow.*

**Symbiote** is an intelligent, adaptive coding assistant that *learns how you code*—and then becomes your snarky, ever-evolving sidekick. It observes, remembers, critiques (sometimes too honestly), and writes code in your *exact* style, almost like your second brain... if your second brain had better linting skills.

Powered by **LangGraph**, **LangChain**, and **Gemini API**, Symbiote builds a personal graph of your habits, quirks, and coding rituals to deliver personalized suggestions, witty corrections, and hauntingly familiar code snippets.

---

## ✨ Features

* 🧠 **Code Style Graphing**
  Constructs a LangGraph of your syntax patterns, preferred APIs, folder structures, and logic flow.

* 🛠 **Self-Updating Codex**
  Continuously refines itself using LangChain agents to stay in sync with your evolving style. It's basically stalking your repo. (Not creepy, promise.)

* 💬 **Sass-Enabled Feedback Mode™**
  Optional snark included. Turn on “Tsundere Commentary” if you like your AI with a side of emotional damage. 😤

* 📚 **Contextual Code Generation (Gemini-Powered)**
  Generates code that actually looks like you wrote it—bugs and all (but cleaner).

* 🧪 **Persona-Based PR Reviews**
  Gemini agent reviews your pull requests based on your past coding behavior. Might even roast your old self.

---

## 🛠 Tech Stack

* **LangGraph** — builds dynamic code pattern knowledge graphs.
* **LangChain** — handles multi-agent orchestration & prompt routing.
* **Gemini API** — does the heavy lifting for code understanding & generation.

---

## 📁 Repo Structure

```bash
symbiote/
├── agents/
│   ├── style_learner.py       # Learns & updates your code style graph
│   └── reviewer_agent.py      # Reviews PRs based on your coding patterns
├── graph/
│   └── style_graph.py         # Constructs LangGraph from your codebase
├── prompts/
│   └── personality.txt        # Sass levels: 0 (vanilla) to 10 (tsundere meltdown)
├── utils/
│   └── code_parser.py         # Tokenizes & analyzes code features
├── main.py                    # Orchestration entry point
└── README.md
```

---

## 🚀 Getting Started

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

```env
GEMINI_API_KEY=your_secret_key
```

4. **Run your symbiotic buddy**

```bash
python main.py --mode learn --path ./your/codebase
```

---

## 🤖 Modes

| Mode       | Description                           |
| ---------- | ------------------------------------- |
| `learn`    | Builds/updates your LangGraph profile |
| `review`   | Reviews a given diff/PR               |
| `generate` | Suggests code matching your style     |
| `sass`     | Enables snarky commentary             |

---

## 🧪 Future Features

* VSCode Plugin
* GitHub App for PR auto-commenting
* Visual style graph dashboard
* “Clone Me” mode (your evil AI twin?)
* Support for team-wide personality blending

---

## 👨‍💻 Author

Made by [Vivek Sharma](https://ifsvivek.in) — the type of dev whose AI assistant may become *too powerful*.

---

## 📝 License

MIT — Use it, break it, improve it, sass it. Just don’t sell it as *your* symbiote. 😤
