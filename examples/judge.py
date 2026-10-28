"""LLM-as-a-judge on a ChatGPT plan: label claims with gpt-5.6-luna, forcing a web search."""
import json

from chatgpt_as_api import OpenAI

client = OpenAI()
messages = [
    {"role": "system", "content": 'You are an evaluator. Return ONLY valid JSON: {"claims": [{"text": "<claim>", "label": 0 or 1}]}'},
    {"role": "user", "content": "User query: When did the Supplemental Security Income (SSI) program start?\n"
                                "Claims to label:\n1. SSI began paying benefits in January 1974.\n2. SSI is administered by the IRS."},
]
r = client.responses.create(model="gpt-5.6-luna", input=messages, tools=[{"type": "web_search"}], tool_choice="required")
text = r.output_text
print(text)
try:
    labels = json.loads(text[text.find("{"):text.rfind("}") + 1])  # judges sometimes wrap the JSON in prose or fences
    print("labels:", [c["label"] for c in labels["claims"]])
except (ValueError, KeyError):
    pass
print("searches:", sum(1 for item in r.output if item.type == "web_search_call"), "| usage:", r.usage.total_tokens)
