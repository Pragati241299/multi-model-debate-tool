"""
Multi-Model Debate Tool

Ask one question, and three AI providers - ChatGPT, Groq, and Gemini - each take a clear
position on it, backed by real tools: a calculator for exact math and a fact-lookup tool
so arguments are grounded in evidence rather than pure opinion. A fourth call acts as a
neutral judge, weighing all three answers and giving a final verdict. That verdict then
gets turned into a generated illustration and read aloud - all running through a Gradio
interface.
"""

# imports
import os
import json
import sqlite3
import base64
import ast
import operator
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import gradio as gr

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

if not OPENAI_API_KEY:
    print("No OPENAI_API_KEY found - GPT calls will fail. Check your .env file.")
if not GROQ_API_KEY:
    print("No GROQ_API_KEY found - Groq calls will fail. Check your .env file.")
if not GEMINI_API_KEY:
    print("No GEMINI_API_KEY found - Gemini calls will fail. Check your .env file.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
gemini_client = OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)

DEBATERS = [
    ("ChatGPT", openai_client, "gpt-4.1-mini"),
    ("Groq",    groq_client,   "openai/gpt-oss-120b"),
    ("Gemini",  gemini_client, "gemini-3.1-flash-lite"),
]

JUDGE_CLIENT = openai_client
JUDGE_MODEL = "gpt-4.1-mini"

DEBATER_SYSTEM_PROMPT = """You are a sharp, opinionated debater.
When given a question or topic:
- Take a clear, direct position - don't hedge or just list pros and cons.
- Use the calculate or look_up_fact tools where they would genuinely strengthen your argument.
- Keep your answer to 3-4 sentences, not an essay.
"""

JUDGE_SYSTEM_PROMPT = """You are a fair, neutral debate moderator.
Summarize each side objectively, identify the strongest argument, and give a balanced final
verdict - don't simply agree with whichever answer came first.
"""

calculate_function = {
    "name": "calculate",
    "description": "Evaluate a basic mathematical expression. Use this whenever your argument involves "
                    "numbers, statistics, or a calculation, so the math is exact rather than estimated.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression to evaluate, e.g. '2500000 / 12' or '0.15 * 80000'"
            }
        },
        "required": ["expression"]
    }
}

look_up_fact_function = {
    "name": "look_up_fact",
    "description": "Look up a stored fact related to a topic, to support your argument with evidence "
                    "rather than opinion alone.",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "A keyword describing what to look up, e.g. 'remote work', 'career growth', 'AI'"
            }
        },
        "required": ["topic"]
    }
}

tools = [
    {"type": "function", "function": calculate_function},
    {"type": "function", "function": look_up_fact_function},
]

ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPS:
        return ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPS:
        return ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only basic arithmetic (+ - * / **) is allowed")


def calculate(expression):
    """Safely evaluate a math expression - no eval(), so it can't run arbitrary code."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


DB_PATH = "debate_facts.db"


def setup_database():
    """Create the facts table and seed it with sample facts. Re-running this resets the data."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS debate_facts")
    cur.execute("""
        CREATE TABLE debate_facts (
            id INTEGER PRIMARY KEY,
            topic TEXT,
            fact TEXT
        )
    """)

    sample_facts = [
        ("remote work", "Remote and hybrid work became far more common after 2020, and many companies now offer flexible arrangements."),
        ("career growth", "Employers consistently rank communication and adaptability among the top skills for career advancement."),
        ("artificial intelligence", "AI adoption in the workplace has grown rapidly, with many companies integrating AI tools into everyday workflows."),
        ("education", "Continuous learning throughout a career is increasingly viewed as essential given how quickly industries change."),
        ("productivity", "Research on productivity consistently shows focused, uninterrupted work blocks outperform constant multitasking."),
        ("teamwork", "Strong collaboration and communication skills are frequently cited as more predictive of career success than technical skill alone."),
        ("work-life balance", "Burnout from overwork has been linked to reduced long-term productivity, prompting many companies to rethink always-on culture."),
        ("technology", "New programming languages and frameworks emerge frequently, making adaptability a valuable skill in tech careers."),
    ]
    cur.executemany("INSERT INTO debate_facts (topic, fact) VALUES (?, ?)", sample_facts)
    conn.commit()
    conn.close()
    print(f"Database ready with {len(sample_facts)} facts")


setup_database()


def look_up_fact(topic):
    """Search the facts table for anything matching the given topic keyword."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT fact FROM debate_facts WHERE topic LIKE ?", (f"%{topic}%",))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"facts": [], "message": "No facts found on that topic."}
    return {"facts": [r[0] for r in rows]}


available_tools = {
    "calculate": calculate,
    "look_up_fact": look_up_fact,
}


def ask_model(client, model, question, system_prompt=None):
    """Ask a model a question. Loops on tool calls until it gives a final answer."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(model=model, messages=messages, tools=tools)

        while response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            messages.append(message)

            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                function = available_tools[function_name]
                result = function(**arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            response = client.chat.completions.create(model=model, messages=messages, tools=tools)

        return response.choices[0].message.content
    except Exception as e:
        return f"[Error calling {model}: {e}]"


def generate_debate_image(question):
    """Generate a simple illustrative image representing the debate topic."""
    image_response = openai_client.images.generate(
        model="gpt-image-1-mini",
        prompt=f"A simple, colorful, abstract illustration representing a debate about: {question}. No text.",
        size="1024x1024",
        n=1,
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def speak_verdict(verdict):
    """Turn the judge's verdict into spoken audio and save it to a file."""
    response = openai_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=verdict
    )
    audio_path = "debate_verdict.mp3"
    with open(audio_path, "wb") as f:
        f.write(response.content)
    return audio_path


def build_synthesis_prompt(question, answers):
    prompt = (
        "You are moderating a debate between several AI models that were all asked the same question.\n\n"
        f"Question: {question}\n\n"
    )
    for name, answer in answers.items():
        prompt += f"--- {name}'s answer ---\n{answer}\n\n"

    prompt += (
        "Please:\n"
        "1. Summarize where the models agreed and disagreed.\n"
        "2. Point out which argument(s) were strongest and why.\n"
        "3. Give your own synthesized answer that incorporates the best points.\n"
    )
    return prompt


def run_debate(question):
    """Terminal version of the debate - prints each step, saves the image and audio
    to disk instead of displaying them inline (which only works in a notebook)."""
    print(f"## Question\n{question}\n")
    answers = {}

    for name, client, model in DEBATERS:
        answer = ask_model(client, model, question, system_prompt=DEBATER_SYSTEM_PROMPT)
        answers[name] = answer
        print(f"### {name}\n{answer}\n")

    synthesis_prompt = build_synthesis_prompt(question, answers)
    verdict = ask_model(JUDGE_CLIENT, JUDGE_MODEL, synthesis_prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
    print(f"### Synthesis / Verdict (by {JUDGE_MODEL})\n{verdict}\n")

    image = generate_debate_image(question)
    image_path = "debate_illustration.png"
    image.save(image_path)
    print(f"Illustration saved to {image_path}")

    audio_path = speak_verdict(verdict)
    print(f"Spoken verdict saved to {audio_path}")

    return answers, verdict


def run_debate_ui(question):
    """Same logic as run_debate, but returns plain values for Gradio instead of printing."""
    answers = {}
    for name, client, model in DEBATERS:
        answers[name] = ask_model(client, model, question, system_prompt=DEBATER_SYSTEM_PROMPT)

    synthesis_prompt = build_synthesis_prompt(question, answers)
    verdict = ask_model(JUDGE_CLIENT, JUDGE_MODEL, synthesis_prompt, system_prompt=JUDGE_SYSTEM_PROMPT)

    gpt_display = f"### ChatGPT\n{answers['ChatGPT']}"
    groq_display = f"### Groq\n{answers['Groq']}"
    gemini_display = f"### Gemini\n{answers['Gemini']}"

    image = generate_debate_image(question)
    audio_path = speak_verdict(verdict)

    return gpt_display, groq_display, gemini_display, verdict, image, audio_path


def build_ui():
    with gr.Blocks(title="Multi-Model Debate Arena") as ui:
        gr.Markdown("# Multi-Model Debate Arena\nAsk one question, watch ChatGPT, Groq, and Gemini debate it live.")

        question_box = gr.Textbox(
            label="Question or topic",
            placeholder="What's the most important skill for a junior developer to learn?"
        )
        run_button = gr.Button("Start Debate", variant="primary")

        with gr.Row():
            gpt_output = gr.Markdown(value="### ChatGPT\n*Waiting for a question...*")
            groq_output = gr.Markdown(value="### Groq\n*Waiting for a question...*")
            gemini_output = gr.Markdown(value="### Gemini\n*Waiting for a question...*")

        verdict_output = gr.Markdown(label="Verdict")

        with gr.Row():
            image_output = gr.Image(height=300, label="Debate illustration")
            audio_output = gr.Audio(label="Spoken verdict", autoplay=False)

        run_button.click(
            run_debate_ui,
            inputs=question_box,
            outputs=[gpt_output, groq_output, gemini_output, verdict_output, image_output, audio_output]
        )

    return ui


if __name__ == "__main__":
    # Launches the Gradio app - open the printed local URL in your browser.
    # To run a single debate straight in the terminal instead, comment the
    # two lines below out and uncomment the run_debate(...) call.
    ui = build_ui()
    ui.launch(share=True)

    # question = "What's the single most important skill for a junior software engineer to develop in their first year?"
    # answers, verdict = run_debate(question)
