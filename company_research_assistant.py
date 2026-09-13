"""
Company Research & Brochure Assistant

Give it a company name and website URL. It scrapes the site, picks out the pages worth
reading (About, Careers, etc.), writes a short brochure about the company, and then lets
you ask follow-up questions about it - with real conversation memory, using GPT or Groq's
free tier.
"""

# imports
import os
import json
from dotenv import load_dotenv
from scraper import fetch_website_links, fetch_website_contents
from openai import OpenAI
import tiktoken

# Initialize and constants

load_dotenv(override=True)

api_key = os.getenv('OPENAI_API_KEY')
groq_key = os.getenv('GROQ_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key) > 10:
    print("OpenAI API key looks good so far")
else:
    print("There might be a problem with your OpenAI API key? Please visit the troubleshooting notebook!")

openai = OpenAI(api_key=api_key)
groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)

MODEL = "gpt-4.1-mini"
LINK_MODEL = "gpt-5-nano"

encoding = tiktoken.get_encoding("cl100k_base")

link_system_prompt = """
You are provided with a list of links found on a webpage.
Decide which links are most relevant to include in a brochure about the company,
such as an About page, Company page, or Careers/Jobs page.

IMPORTANT RULES:
- Only include links on the company's OWN domain. Ignore external sites such as
  LinkedIn, Twitter/X, GitHub, Discord, status pages, and external job boards.
- Do not include Terms of Service, Privacy, or email (mailto:) links.
- Return at most 5 links.
- Convert relative links such as "/about" into the full https URL.

Respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://full.url/careers"}
    ]
}
"""


def get_links_user_prompt(url):
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company, 
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt


def select_relevant_links(url):
    try:
        response = openai.chat.completions.create(
            model=LINK_MODEL,
            messages=[
                {"role": "system", "content": link_system_prompt},
                {"role": "user", "content": get_links_user_prompt(url)}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Link selection failed: {e}")
        return {"links": []}


def fetch_page_and_all_relevant_links(url):
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"

    for link in relevant_links['links']:
        try:
            page_content = fetch_website_contents(link["url"])
            result += f"\n\n### Link: {link['type']}\n{page_content}"
        except Exception as e:
            print(f"Could not fetch {link['url']}: {e}")

    return result


brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages
from a company website and creates a short brochure about the company.

The brochure is intended for:
- prospective customers
- investors
- potential employees

Include information such as:
- Company overview
- Products and services
- What makes the company different
- Customers or users
- Company culture
- Careers and job opportunities

Only use information provided in the website content.
Do not make up facts.

Respond in Markdown.
"""

MAX_TOKENS = 3_000  # keeps the prompt from getting too big or too expensive


def get_brochure_user_prompt(company_name, url):
    header = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    content = fetch_page_and_all_relevant_links(url)
    tokens = encoding.encode(content)[:MAX_TOKENS]
    print(f"Using {len(tokens)} tokens of content")
    return header + encoding.decode(tokens)


def create_brochure(company_name, url, client=openai, model=MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
    )
    result = response.choices[0].message.content
    print(result)
    return result


def stream_brochure(company_name, url, client=openai, model=MODEL):
    """Same as create_brochure, but prints the response as it streams in
    instead of waiting for the full reply - a terminal equivalent of the
    notebook's live-updating Markdown display."""
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ],
        stream=True
    )
    response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        response += content
        print(content, end="", flush=True)
    print()  # final newline after streaming finishes
    return response


def chat_about_company(brochure_text, client=openai, model=MODEL):
    """Ask follow-up questions and get answers that remember what you already asked."""
    messages = [
        {"role": "system", "content": f"Answer questions about the company based only on this brochure:\n\n{brochure_text}"}
    ]
    print("Ask anything. Type 'quit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": user_input})

        stream = client.chat.completions.create(model=model, messages=messages, stream=True)
        reply = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            reply += content
            print(content, end="", flush=True)
        print()

        messages.append({"role": "assistant", "content": reply})

    return messages


if __name__ == "__main__":
    # Change these to research any company you like
    company_name = "OpenAI"
    company_url = "https://openai.com"

    # To use Groq instead of OpenAI, pass: groq, "openai/gpt-oss-120b"
    brochure_text = stream_brochure(company_name, company_url)

    conversation = chat_about_company(brochure_text)
