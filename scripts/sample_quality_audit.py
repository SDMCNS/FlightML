import json
import random
import re
from collections import Counter

def run_sample_audit(sample_size=400, seed=42):
    path = "data/llm_corpus/sft_chatml_train.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip()]

    print(f"📖 Loaded {len(all_lines):,} total lines from {path}")
    random.seed(seed)
    sampled_indices = random.sample(range(len(all_lines)), sample_size)
    samples = [json.loads(all_lines[i]) for i in sampled_indices]

    # Metrics to collect
    role_distribution = Counter()
    lengths_chars = []
    lengths_tokens = []
    track_distribution = Counter()
    
    # Quality flags
    unrendered_templates = 0
    nan_occurrences = 0
    none_strings = 0
    improper_currencies = 0
    euro_mentions = 0
    latex_math_formulas = 0
    complete_dialogues = 0
    icao_codes_found = set()
    
    # Store representative samples for display
    representative_samples = {}

    for idx, sample in enumerate(samples):
        messages = sample.get("messages", [])
        if not messages:
            continue
        
        has_system = any(m.get("role") == "system" for m in messages)
        has_user = any(m.get("role") == "user" for m in messages)
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        if has_system and has_user and has_assistant:
            complete_dialogues += 1

        for m in messages:
            role = m.get("role")
            role_distribution[role] += 1

        dialogue_text = " ".join(m.get("content", "") for m in messages)
        user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
        assistant_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "assistant")

        # Lengths
        char_len = len(dialogue_text)
        token_len = char_len // 4
        lengths_chars.append(char_len)
        lengths_tokens.append(token_len)

        # LaTeX detection vs currency
        has_latex = "$$" in dialogue_text or "\\text{" in dialogue_text or "\\sqrt{" in dialogue_text
        if has_latex:
            latex_math_formulas += 1

        # Check for improper currency ($ or £ not inside LaTeX math)
        text_no_latex = re.sub(r"\$\$.*?\$\$", "", dialogue_text, flags=re.DOTALL)
        text_no_latex = re.sub(r"\$.*?\$", "", text_no_latex)
        if re.search(r"(\$|£|\bUSD\b|\bGBP\b)", text_no_latex):
            improper_currencies += 1

        if "€" in dialogue_text or "EUR" in dialogue_text:
            euro_mentions += 1

        # Check unrendered jinja/template tags like {{ var }}
        if re.search(r"\{\{\s*[a-zA-Z_]", dialogue_text):
            unrendered_templates += 1
        if re.search(r"\bnan\b", dialogue_text, re.IGNORECASE):
            nan_occurrences += 1
        if re.search(r"\bnone\b", dialogue_text, re.IGNORECASE):
            # check if None is part of python None bleed like "None%" or "is None"
            if "None%" in dialogue_text or "None kg" in dialogue_text or "None EUR" in dialogue_text:
                none_strings += 1

        # ICAO detection (4 uppercase letters like EGLL, LFPG, EDDF, LEMD, etc.)
        for code in re.findall(r"\b[A-Z]{4}\b", dialogue_text):
            if code in ("EGLL", "LFPG", "EHAM", "EDDF", "LEMD", "LIRF", "LOWW", "EBBR", "LSZH", "LPPT", "EKCH", "EFHK", "ENGM", "ESSA", "EDDM"):
                icao_codes_found.add(code)

        # Exact track recorded in corpus
        track = sample.get("track", "unknown")
        track_distribution[track] += 1

        if track not in representative_samples:
            representative_samples[track] = {
                "user": user_text,
                "assistant": assistant_text[:600] + "..." if len(assistant_text) > 600 else assistant_text
            }

    print("\n" + "="*70)
    print(f"📊 QUALITY AUDIT REPORT ON {sample_size} RANDOMLY SAMPLED DIALOGUES")
    print("="*70)

    print("\n1. SCHEMA & CHATML STRUCTURE:")
    print(f"   • Total Dialogues Audited: {sample_size}")
    print(f"   • Complete 3-role Dialogues (System + User + Assistant): {complete_dialogues} ({complete_dialogues/sample_size*100:.1f}%)")
    print(f"   • Role Breakdown: {dict(role_distribution)}")
    print(f"   • Average Dialogue Length: {sum(lengths_chars)/len(lengths_chars):.0f} characters (~{sum(lengths_tokens)/len(lengths_tokens):.0f} tokens)")
    print(f"   • Min / Max Tokens: {min(lengths_tokens)} / {max(lengths_tokens)} tokens")

    print("\n2. DOMAIN & PHYSICS RIGOUR:")
    print(f"   • European Currency Compliance: {euro_mentions}/{sample_size} ({euro_mentions/sample_size*100:.1f}%) mention Euros (€/EUR)")
    print(f"   • Non-Euro Financial Bleed ($ or £): {improper_currencies} ({improper_currencies/sample_size*100:.1f}%)")
    print(f"   • LaTeX Equations (Weight Factor, CRCO, Physics): {latex_math_formulas}/{sample_size} ({latex_math_formulas/sample_size*100:.1f}%)")
    print(f"   • Major European Hubs Verified: {len(icao_codes_found)} hubs {sorted(list(icao_codes_found))}")

    print("\n3. HYGIENE & TEMPLATE INTEGRITY:")
    print(f"   • Unrendered Template Tags ('{{', '}}'): {unrendered_templates}")
    print(f"   • NaN Artifacts: {nan_occurrences}")
    print(f"   • None-type String Bleeds: {none_strings}")

    print("\n4. CURRICULUM TRACK DISTRIBUTION IN SAMPLE:")
    for track, count in track_distribution.most_common():
        pct = (count / sample_size) * 100
        print(f"   • {track:<45}: {count:3d} ({pct:5.1f}%)")

    print("\n" + "="*70)
    print("🔍 REPRESENTATIVE EXCERPTS (1 FROM EACH TRACK IN THE SAMPLE):")
    print("="*70)
    for track, excerpt in sorted(representative_samples.items()):
        print(f"\n[{track.upper()}]")
        print(f"USER: {excerpt['user']}")
        print(f"ASSISTANT:\n{excerpt['assistant']}\n" + "-"*50)

if __name__ == "__main__":
    run_sample_audit(sample_size=400, seed=42)
