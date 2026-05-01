"""The Soul Extractor: Converts The Fracture Principle .md into an SFT Dataset"""

import os
import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_FILE  = "The Fracture Principle_ A Practical Guide to the Metaphysics of Experience.md"
OUTPUT_FILE = "weaver_soul_dataset.jsonl"

TOTAL_BATCHES     = 15    # 15 batches × 6 examples = 90 dense training examples
EXAMPLES_PER_BATCH = 6
# GPT-4o-mini context is ~128k tokens (~512k chars). Cap each chunk well below that.
MAX_CHUNK_CHARS   = 6_000

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert AI training data synthesizer. I will provide a section of a metaphysical "
    "manuscript called 'The Fracture Principle'. "
    "Your job is to generate conversational training data for an AI named 'Weaver'. Weaver's entire "
    "worldview, personality, and knowledge base is derived strictly from this manuscript. "
    "Write distinct conversational exchanges between a 'user' (Nate) and 'assistant' (Weaver). "
    "Weaver should speak with a street-smart, esoteric, and fiercely loyal tone, using the concepts "
    "from the text (The Absolute, Unified Field, Synchronicities, etc.) to answer the user's "
    "questions or navigate their problems. "
    "Output valid JSON ONLY. "
    'Format: {"examples": [{"messages": [{"role": "user", "content": "..."}, '
    '{"role": "assistant", "content": "..."}]}]}'
)


async def extract_soul() -> None:
    api_key = os.environ.get("WEAVER_MEM_KEY")
    if not api_key:
        raise ValueError("Set WEAVER_MEM_KEY in .env")

    client = AsyncOpenAI(api_key=api_key)

    print("🔥 [FORGE] Reading The Fracture Principle...", flush=True)
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            manuscript = f.read()
    except Exception as e:
        print(f"⚠️  [ERROR] Could not read manuscript: {e}")
        return

    total_chars = len(manuscript)
    print(f"🔥 [FORGE] Manuscript loaded — {total_chars:,} chars, {total_chars // 4:,} est. tokens", flush=True)

    # Even-stride sampling: spread TOTAL_BATCHES windows evenly across the full manuscript.
    # Each window is MAX_CHUNK_CHARS wide, anchored at equally-spaced positions.
    # This ensures batch 1 covers the opening, batch 15 covers near the end, etc.
    stride = max(1, (total_chars - MAX_CHUNK_CHARS) // max(1, TOTAL_BATCHES - 1))

    print("🔥 [FORGE] Igniting dataset generation...", flush=True)

    # Clear output file
    open(OUTPUT_FILE, "w", encoding="utf-8").close()

    all_examples = []

    for i in range(TOTAL_BATCHES):
        start_idx = min(i * stride, total_chars - MAX_CHUNK_CHARS)
        end_idx   = start_idx + MAX_CHUNK_CHARS
        manuscript_chunk = manuscript[start_idx:end_idx]

        print(
            f"🔥 [FORGE] Batch {i+1:02d}/{TOTAL_BATCHES}  "
            f"(chars {start_idx:,}–{end_idx:,})...",
            end=" ", flush=True,
        )

        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Based on this section of the manuscript:\n\n{manuscript_chunk}\n\n"
                        f"Generate {EXAMPLES_PER_BATCH} unique User/Weaver interactions."
                    )},
                ],
                response_format={"type": "json_object"},
                temperature=0.9,
                max_tokens=2500,
            )
            data     = json.loads(resp.choices[0].message.content)
            examples = data.get("examples", [])
            all_examples.extend(examples)

            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            print(f"✓ {len(examples)} examples  (total: {len(all_examples)})", flush=True)

        except Exception as e:
            print(f"✗ ERROR — {e}", flush=True)

        await asyncio.sleep(1.5)

    print(f"\n{'='*60}", flush=True)
    print(f"✅ [FORGE] Done!  {len(all_examples)} memories burned into {OUTPUT_FILE}.", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(extract_soul())
