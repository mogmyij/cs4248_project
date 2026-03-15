#!/usr/bin/env python3
"""Generate synthetic sarcastic headlines using DeepSeek-V3."""

import argparse
import asyncio
import json
import os
import sys

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from tqdm import tqdm

API_BASE_URL = "https://api.deepseek.com"
MODEL_NAME   = "deepseek-chat"
MAX_RETRIES  = 3
BASE_DELAY   = 1.0
TEMPERATURE  = 0.9
MAX_TOKENS   = 100

SYSTEM_PROMPT = """You are a headline writer for a satirical news publication in the style of The Onion.
Transform the given genuine news headline into a sarcastic, satirical version.

Rules:
- Keep the same general topic and named entities where possible
- Use deadpan, mock-serious tone — absurdity in content, not exclamation marks
- Do NOT use words like "sarcastic", "ironically", or "satirically"
- Similar length to input; one headline only
- Output ONLY the transformed headline — no explanation, no prefix"""


def load_dataset(path: str) -> list[str]:
    headlines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("is_sarcastic") == 0:
                    headlines.append(item["headline"])
            except (json.JSONDecodeError, KeyError):
                pass
    return headlines


def load_already_done(path: str) -> set[str]:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                done.add(item["input"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


async def call_api(client: AsyncOpenAI, headline: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": headline},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(BASE_DELAY * 2 ** attempt)
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < MAX_RETRIES - 1:
                await asyncio.sleep(BASE_DELAY * 2 ** attempt)
            else:
                raise
        except APIConnectionError:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(BASE_DELAY * 2 ** attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries for headline: {headline!r}")


async def process_headline(
    headline: str,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    f,
    pbar: tqdm,
    counters: dict,
) -> None:
    async with semaphore:
        try:
            sarcastic = await call_api(client, headline)
            f.write(json.dumps({"input": headline, "output": sarcastic}, ensure_ascii=False) + "\n")
            f.flush()
            counters["ok"] += 1
        except Exception as e:
            print(f"WARNING: failed for {headline!r}: {e}", file=sys.stderr)
            counters["failed"] += 1
        pbar.update(1)


async def run(todo: list[str], output_path: str, concurrency: int, api_key: str) -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    counters = {"ok": 0, "failed": 0}

    with open(output_path, "a") as f, tqdm(total=len(todo), desc="Generating") as pbar:
        await asyncio.gather(
            *[process_headline(h, semaphore, client, f, pbar, counters) for h in todo],
            return_exceptions=True,
        )

    print(f"\nDone: {counters['ok']} succeeded, {counters['failed']} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sarcastic headlines via DeepSeek-V3")
    parser.add_argument("--dataset", default="Sarcasm_Headlines_Dataset.json", help="Input JSONL dataset")
    parser.add_argument("--output", default="synthetic_sarcastic.jsonl", help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Process only N remaining headlines")
    parser.add_argument("--concurrency", type=int, default=5, help="Max simultaneous API calls")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    all_headlines = load_dataset(args.dataset)
    done_set = load_already_done(args.output)

    todo = [h for h in all_headlines if h not in done_set]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"Dataset: {len(all_headlines)} non-sarcastic headlines")
    print(f"Already done: {len(done_set)}")
    print(f"To process: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    asyncio.run(run(todo, args.output, args.concurrency, api_key))


if __name__ == "__main__":
    main()
