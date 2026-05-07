"""
seo_agent.py
-----------------
Automatically generates SEO-optimized blog articles
from a company.txt file using the OpenAI API (gpt-4o).

Usage:
    pip install openai
    export OPENAI_API_KEY="sk-..."
    python blog_generator.py
"""

import os
import re
import json
import time
import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "❌  Module 'openai' not found. Install it with: pip install openai"
    )

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

COMPANY_FILE  = "company.txt"
BLOG_FOLDER   = "blog"
MODEL         = "gpt-4o"
RETRY_LIMIT   = 2
RETRY_DELAY   = 3   # seconds between retry attempts

# ─────────────────────────────────────────────────────────────
# OPENAI CLIENT INITIALIZATION
# ─────────────────────────────────────────────────────────────

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "❌  OPENAI_API_KEY environment variable not found.\n"
        "    Run: export OPENAI_API_KEY='sk-...'"
    )

client = OpenAI(api_key=api_key)

# Global token counter to estimate cost
total_tokens_used = {"prompt": 0, "completion": 0}


# ─────────────────────────────────────────────────────────────
# PARSE COMPANY.TXT
# ─────────────────────────────────────────────────────────────

def parse_company_file(filepath: str) -> dict:
    """
    Read company.txt and return a dictionary with company data.
    Each line must follow the format: KEY: value
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"❌  File '{filepath}' not found. "
            "Create it based on the example company.txt provided."
        )

    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignore empty lines and comments
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                data[key.strip().upper()] = value.strip()

    # Required fields
    required = ["COMPANY_NAME", "INDUSTRY", "PROBLEM", "SOLUTION",
                "TARGET_AUDIENCE", "KEYWORDS", "TONE", "NB_ARTICLES"]
    for field in required:
        if field not in data:
            raise ValueError(f"❌  Required field missing in company.txt: '{field}'")

    # Convert number of articles to integer
    try:
        data["NB_ARTICLES"] = int(data["NB_ARTICLES"])
    except ValueError:
        raise ValueError("❌  NB_ARTICLES must be an integer (ex: NB_ARTICLES: 8)")

    return data


# ─────────────────────────────────────────────────────────────
# OPENAI API CALL WITH ERROR HANDLING AND RETRY
# ─────────────────────────────────────────────────────────────

def call_openai(messages: list, temperature: float = 0.7) -> str:
    """
    Call OpenAI API with automatic retry on failure.
    Returns response text or None if all attempts failed.
    """
    for attempt in range(1, RETRY_LIMIT + 2):  # 1 normal attempt + RETRY_LIMIT retries
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
            )
            # Accumulate tokens used
            usage = response.usage
            total_tokens_used["prompt"]     += usage.prompt_tokens
            total_tokens_used["completion"] += usage.completion_tokens

            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt <= RETRY_LIMIT:
                print(f"    ⚠️  API Error (attempt {attempt}/{RETRY_LIMIT}): {e}")
                print(f"    ⏳ Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    ❌  Failed after {RETRY_LIMIT + 1} attempts: {e}")
                return None


# ─────────────────────────────────────────────────────────────
# GENERATE ARTICLE TOPICS
# ─────────────────────────────────────────────────────────────

def generate_topics(company: dict) -> list:
    """
    Ask GPT-4o to generate a list of SEO-optimized article topics
    for the company in JSON format.
    """
    print(f"\n📋 Generating {company['NB_ARTICLES']} article topics...")

    prompt = f"""You are an SEO and content marketing expert.

Here is company information:
- Name: {company['COMPANY_NAME']}
- Industry: {company['INDUSTRY']}
- Problem solved: {company['PROBLEM']}
- Solution offered: {company['SOLUTION']}
- Target audience: {company['TARGET_AUDIENCE']}
- Priority SEO keywords: {company['KEYWORDS']}
- Editorial tone: {company['TONE']}

Generate exactly {company['NB_ARTICLES']} SEO-optimized blog article topics \\
for this company.
Each topic must:
- Address a specific search intent (informational, commercial, or transactional)
- Target a main keyword from the provided list or a relevant long-tail keyword
- Be relevant to the target audience
- Be varied (do not repeat the same angle)

Respond ONLY with valid JSON array, no additional text, in this exact format:
[
  {{
    "title": "SEO-optimized article title",
    "main_keyword": "targeted keyword",
    "intent": "informational|commercial|transactional",
    "angle": "brief description of editorial angle (1 sentence)"
  }}
]"""

    response = call_openai([{"role": "user", "content": prompt}], temperature=0.8)

    if not response:
        raise RuntimeError("❌  Failed to generate article topics.")

    # Clean JSON (GPT may sometimes add backticks)
    cleaned = re.sub(r"```json\s*|\s*```", "", response).strip()

    try:
        topics = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"❌  Invalid JSON response for topics: {e}\n\nRaw response:\n{response}")

    print(f"    ✅  {len(topics)} topics generated successfully.")
    return topics


# ─────────────────────────────────────────────────────────────
# GENERATE COMPLETE ARTICLE
# ─────────────────────────────────────────────────────────────

def generate_article(topic: dict, company: dict) -> str:
    """
    Generate a complete blog article in Markdown for a given topic.
    """
    prompt = f"""You are an expert SEO writer. Write a complete blog article \\
optimized for SEO in English.

Company Information:
- Name: {company['COMPANY_NAME']}
- Industry: {company['INDUSTRY']}
- Problem solved: {company['PROBLEM']}
- Solution: {company['SOLUTION']}
- Target audience: {company['TARGET_AUDIENCE']}
- Tone: {company['TONE']}

Article Topic:
- Title: {topic.get('title', 'Article')}
- Main keyword: {topic.get('main_keyword', '')}
- Search intent: {topic.get('intent', '')}
- Editorial angle: {topic.get('angle', '')}

Generate the complete article in Markdown following EXACTLY this structure:

---
title: {topic.get('title', 'Article')}
meta_description: [max 155 characters, include main keyword]
keyword: {topic.get('main_keyword', '')}
date: {datetime.date.today().isoformat()}
---

# {topic.get('title', 'Article')}

[Engaging introduction of 150-200 words. Present the problem, why it matters to the reader, and what they will learn. Include the keyword naturally.]

## [H2 Title – Context or problem definition]

[200-300 words. Develop the subject with data or statistics if relevant.]

## [H2 Title – Main causes or challenges]

[200-300 words.]

## [H2 Title – Recommended solution or approach]

[200-300 words. Mention {company['COMPANY_NAME']} naturally if relevant.]

## [H2 Title – Practical implementation / concrete steps]

[200-300 words. Actionable advice, bullet points where useful.]

## [H2 Title – Common mistakes to avoid or FAQ]

[200-300 words.]

## Conclusion

[100-150 words. Summarize key points and include a clear call-to-action inviting readers to \\
discover {company['COMPANY_NAME']} or contact the team.]

---
*Article written by {company['COMPANY_NAME']}*

Rules to follow:
- Main keyword must appear in H1, meta description, introduction, and at least 2 H2 sections
- No keyword stuffing (natural density ~1-2%)
- Tone: {company['TONE']}
- Use internal links like [link text](#) if relevant
- Use bullet points to improve readability when appropriate"""

    response = call_openai([{"role": "user", "content": prompt}], temperature=0.7)
    return response


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """
    Convert a title to a SEO-friendly slug for filename.
    Ex: "How to optimize your CRM?" → "how-to-optimize-your-crm"
    """
    # Convert to lowercase
    text = text.lower()
    # Replace accented characters
    accents = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'î': 'i', 'ï': 'i',
        'ô': 'o', 'ö': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c', 'ñ': 'n'
    }
    for accented, replacement in accents.items():
        text = text.replace(accented, replacement)
    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    # Replace spaces with hyphens
    text = re.sub(r"\s+", "-", text.strip())
    # Remove multiple hyphens
    text = re.sub(r"-+", "-", text)
    # Limit to 80 characters
    return text[:80].rstrip("-")


def extract_meta_description(article_content: str) -> str:
    """
    Extract meta_description from article frontmatter.
    """
    match = re.search(r"meta_description:\s*(.+)", article_content)
    if match:
        return match.group(1).strip()
    return ""


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate cost in USD based on gpt-4o rates (May 2024).
    Price: $5 / 1M tokens input, $15 / 1M tokens output
    """
    cost_input  = (prompt_tokens     / 1_000_000) * 5.00
    cost_output = (completion_tokens / 1_000_000) * 15.00
    return cost_input + cost_output


# ─────────────────────────────────────────────────────────────
# GENERATE INDEX
# ─────────────────────────────────────────────────────────────

def generate_index(articles_metadata: list, company: dict) -> str:
    """
    Generate index.md file summarizing all produced articles.
    """
    today = datetime.date.today().strftime("%m/%d/%Y")
    lines = [
        f"# Blog Index – {company['COMPANY_NAME']}",
        f"\n_Generated on {today} · {len(articles_metadata)} articles_\n",
        "---\n",
    ]

    for i, meta in enumerate(articles_metadata, 1):
        lines.append(f"## {i}. [{meta['title']}]({meta['filename']})")
        lines.append(f"\n**Main Keyword:** `{meta['keyword']}`  ")
        lines.append(f"**Date:** {meta['date']}  ")
        if meta['meta_description']:
            lines.append(f"\n> {meta['meta_description']}")
        lines.append("\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# MAIN PROGRAM
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🚀  Blog Generator – Powered by GPT-4o")
    print("=" * 60)

    # 1. Read company.txt
    print(f"\n📂 Reading '{COMPANY_FILE}'...")
    company = parse_company_file(COMPANY_FILE)
    print(f"    ✅  Company: {company['COMPANY_NAME']}")
    print(f"    ✅  Industry: {company['INDUSTRY']}")
    print(f"    ✅  Articles: {company['NB_ARTICLES']}")

    # 2. Create blog/ directory
    blog_path = Path(BLOG_FOLDER)
    blog_path.mkdir(exist_ok=True)
    print(f"\n📁 Directory '{BLOG_FOLDER}/' ready.")

    # 3. Generate article topics
    topics = generate_topics(company)

    # 4. Generate each article
    articles_metadata = []
    errors = []

    print(f"\n✍️  Writing articles...\n")

    for i, topic in enumerate(topics, 1):
        title = topic.get("title", f"Article {i}")
        keyword = topic.get("main_keyword", "")

        print(f"  [{i}/{len(topics)}] {title[:65]}...")

        # Call API to generate article
        article_content = generate_article(topic, company)

        if article_content is None:
            print(f"    ⚠️  Article skipped (API failed).")
            errors.append(title)
            continue

        # SEO-friendly filename
        slug     = slugify(title)
        filename = f"{slug}.md"
        filepath = blog_path / filename

        # Save article
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(article_content)

        print(f"    ✅  Saved → {BLOG_FOLDER}/{filename}")

        # Metadata for index
        articles_metadata.append({
            "title":            title,
            "keyword":          keyword,
            "filename":         filename,
            "date":             datetime.date.today().isoformat(),
            "meta_description": extract_meta_description(article_content),
        })

        # Pause to avoid rate limiting
        if i < len(topics):
            time.sleep(1)

    # 5. Generate index
    if articles_metadata:
        print(f"\n📑 Generating index...")
        index_content = generate_index(articles_metadata, company)
        index_path    = blog_path / "index.md"

        with open(index_path, "w", encoding="utf-8") as f:
            f.write(index_content)

        print(f"    ✅  Index saved → {BLOG_FOLDER}/index.md")

    # 6. Final report
    print("\n" + "=" * 60)
    print("  📊  FINAL REPORT")
    print("=" * 60)
    print(f"  ✅  Articles generated: {len(articles_metadata)}/{len(topics)}")

    if errors:
        print(f"  ⚠️  Failed articles: {len(errors)}")
        for err in errors:
            print(f"      - {err}")

    # Cost estimation
    p_tokens = total_tokens_used["prompt"]
    c_tokens = total_tokens_used["completion"]
    cost     = estimate_cost(p_tokens, c_tokens)

    print(f"\n  🔢  Tokens used")
    print(f"      Prompt:     {p_tokens:,}")
    print(f"      Completion: {c_tokens:,}")
    print(f"      Total:      {p_tokens + c_tokens:,}")
    print(f"\n  💰  Estimated cost: ${cost:.4f} USD")
    print("=" * 60)
    print(f"\n✨  Done! Your articles are in the '{BLOG_FOLDER}/' folder.")


if __name__ == "__main__":
    main()
