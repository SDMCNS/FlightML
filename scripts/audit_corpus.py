import json
import random
import re
from collections import Counter

path = 'data/llm_corpus/sft_chatml_train.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total lines in file: {len(lines)}")

random.seed(42)
sample_size = 400
sampled_indices = random.sample(range(len(lines)), sample_size)
samples = [json.loads(lines[i]) for i in sampled_indices]

dollar_contexts = []
nan_contexts = []
curly_contexts = []

for idx, s in enumerate(samples[:50]):
    text = json.dumps(s, ensure_ascii=False)
    if "$" in text:
        pos = text.find("$")
        dollar_contexts.append((idx, text[max(0, pos-30):pos+30]))
    if "nan" in text.lower():
        pos = text.lower().find("nan")
        # Check if it's the word nan or substring of something like "finance" or "maintenance"
        matches = re.finditer(r"\bnan\b", text, re.IGNORECASE)
        for m in matches:
            p = m.start()
            nan_contexts.append((idx, text[max(0, p-30):p+30]))
    if "{{" in text:
        curly_contexts.append(idx)

print(f"\nDollar contexts found ({len(dollar_contexts)}):")
for idx, ctx in dollar_contexts[:5]:
    print(f"  Sample {idx}: ...{ctx}...")

print(f"\nNaN contexts (word boundary) found ({len(nan_contexts)}):")
for idx, ctx in nan_contexts[:5]:
    print(f"  Sample {idx}: ...{ctx}...")

print(f"\nCurly brackets {{ found: {len(curly_contexts)}")
