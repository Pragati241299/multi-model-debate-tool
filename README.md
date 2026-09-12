# Multi-Model Debate Tool

A Generative AI debate application where **ChatGPT, Groq, and Gemini** independently argue a position on a user-provided question.

Their arguments can use a **calculator tool** and a **SQLite-backed fact lookup tool** to ground responses instead of relying purely on model-generated opinions.

A fourth AI call acts as a **neutral judge**, evaluates all three arguments, and produces a final verdict. The verdict can then be illustrated and converted into speech through image generation and text-to-speech.

## What It Covers

- **Multi-provider LLM setup** using the OpenAI client interface with different `base_url` configurations
- Different **debater and judge personas** using system prompts
- **Tool calling** with:
  - A safe calculator
  - A SQLite-backed fact lookup tool
- A proper **agent loop** that handles tool calls until each model produces its final answer
- **Multi-model debate orchestration**
- AI-generated **final verdict illustration**
- **Text-to-speech** for the final verdict
- A **Gradio interface** for interacting with the application



## How It Works

The application follows this workflow:

```text
User Question
     ↓
┌───────────────┐
│   ChatGPT     │
│    Debater    │
└───────┬───────┘
        │
┌───────▼───────┐
│     Groq      │
│    Debater    │
└───────┬───────┘
        │
┌───────▼───────┐
│    Gemini     │
│    Debater    │
└───────┬───────┘
        │
        ▼
   Neutral Judge
        │
        ▼
  Final Verdict
     ↙     ↘
Image       Audio
Generation  / TTS
        │
        ▼
   Gradio UI
```

Each debater receives the same question but uses its own model/provider configuration and persona.

The judge then receives the three responses and evaluates them before producing the final verdict.

## Tools



### Calculator

The calculator tool uses Python's `ast` module instead of `eval()`.

This allows the application to support basic arithmetic while avoiding arbitrary Python code execution.

### Fact Lookup

The fact lookup tool uses SQLite as a small demonstration database.

It is intended to demonstrate how an LLM can retrieve information from a structured data source through tool calling.

> **Note:** The sample database contains illustrative information and should not be treated as a verified source of truth.



## Tech Stack

- Python
- OpenAI API
- Groq
- Gemini
- Gradio
- SQLite
- Pillow
- Jupyter Notebook
- Python-dotenv



## Project Structure

```text
multi-model-debate-tool/
│
├── multi_model_debate_tool.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The `.env` file is intentionally excluded from Git because it contains API credentials.

## Setup



### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd multi-model-debate-tool
```



### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```



### 3. Install dependencies

```bash
pip install -r requirements.txt
```



### 4. Configure API keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

Never commit the `.env` file to GitHub.

The repository includes `.env.example` as a safe template.

## Running the Project

Open:

```text
multi_model_debate_tool.ipynb
```

Run the notebook cells from top to bottom.

The final cell launches the Gradio application.

Enter a question and click **Start Debate**.

The application will:

1. Send the question to the three debater models
2. Allow the models to use the available tools
3. Collect their arguments
4. Send the arguments to the neutral judge
5. Generate a final verdict
6. Generate an illustration for the verdict
7. Generate a spoken version of the verdict



## Example Questions

The tool can be used for questions such as:

- Is remote work better than working from an office?
- Should AI-generated content be regulated?
- Is cryptocurrency useful for mainstream payments?
- Should companies adopt a four-day work week?



## Generative AI Concepts Demonstrated

This project demonstrates several practical Generative AI concepts:

- LLM API integration
- Multi-model orchestration
- Prompt engineering
- System prompts and AI personas
- Tool/function calling
- Agent loops
- Structured data retrieval
- SQLite integration
- Multi-agent workflows
- LLM-based evaluation
- Image generation
- Text-to-speech
- Gradio-based AI application development



## Security Considerations

The calculator tool uses Python's `ast` module rather than `eval()` to restrict supported expressions.

API credentials are stored in `.env`, which is excluded from version control through `.gitignore`.

Never commit real API keys, tokens, or other credentials to the repository.

## Limitations

- The SQLite fact database is a demonstration dataset rather than a production knowledge source.
- Model responses can still contain inaccuracies or hallucinations.
- Different providers may have different model capabilities, rate limits, and API behavior.
- The quality of the final verdict depends on the quality of the three model responses.



## Future Improvements

Potential improvements include:

- Add more LLM providers
- Allow users to select specific models
- Add conversation history
- Store previous debates
- Add web search for real-time fact verification
- Add debate scoring and leaderboards
- Add streaming responses
- Add configurable debate rounds
- Deploy the application as a web application
- Add authentication and usage limits



## Learning Outcome

This project was built to explore how multiple LLMs can be combined into a single AI workflow rather than relying on one model for every task.

It demonstrates the transition from simple LLM API calls toward **tool-using, multi-agent Generative AI applications**.