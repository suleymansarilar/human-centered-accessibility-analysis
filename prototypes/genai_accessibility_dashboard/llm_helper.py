import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def generate_llm_summary(context, prompt=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if OpenAI is None or not api_key:
        return "LLM narrative is off. Install openai and set OPENAI_API_KEY."

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are an urban analytics helper. Be friendly and short. Mention one good thing and one idea to improve."
    )

    if prompt is None:
        user_prompt = (
            f"Neighborhood: {context.get('neighborhood', '')}\n"
            f"Accessibility score: {context.get('accessibility_score', '')}\n"
            f"Average path distance: {context.get('avg_path_distance', '')}\n"
            f"Transit nodes: {context.get('transit_nodes', '')}\n"
            f"Population: {context.get('population', '')}\n"
            f"Notes: {context.get('notes', '')}\n"
            "Write a short summary (max 120 words)."
        )
    else:
        user_prompt = prompt

    reply = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=220,
        temperature=0.6,
    )

    try:
        return reply.choices[0].message.content.strip()
    except Exception:
        return "There was a problem talking to the LLM."
